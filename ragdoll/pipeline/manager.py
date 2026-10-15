import asyncio
import logging
from pathlib import Path
from typing import Optional, Set, Dict, Any, List

from ragdoll.database import get_connection
from ragdoll.clients.embedding import EmbeddingClient
from ragdoll.clients.kev import KevClient
from ragdoll.pipeline.worker import process_asset_pipeline

logger = logging.getLogger("ragdoll.task_manager")


class BackgroundTaskManager:
    """
    Manages background asset ingestion tasks, enforcing a maximum concurrency limit
    to prevent overwhelming downstream services (Kev, EmbeddingGemma, SQLite, CPU/RAM).
    Tracks active and queued tasks, supports startup recovery, and enables graceful shutdown.
    """

    def __init__(self, max_concurrent_tasks: int = 2):
        self.max_concurrent_tasks = max(1, max_concurrent_tasks)
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._active_tasks: Set[str] = set()
        self._queued_tasks: Set[str] = set()
        self._tracked_asyncio_tasks: Set[asyncio.Task] = set()
        self._shutting_down: bool = False

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Lazily initialize semaphore on the current running loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        return self._semaphore

    async def process_asset(
        self,
        asset_id: str,
        db_path: Path,
        embedding_client: EmbeddingClient,
        kev_client: KevClient,
    ) -> None:
        """
        Executes the ingestion pipeline for an asset within the concurrency limit.
        If at capacity, the task waits gracefully in the queue.
        """
        current_task = asyncio.current_task()
        if current_task is not None:
            self._tracked_asyncio_tasks.add(current_task)
            current_task.add_done_callback(self._tracked_asyncio_tasks.discard)

        self._queued_tasks.add(asset_id)
        logger.info(
            f"Asset [{asset_id}] queued for ingestion. "
            f"(Queued: {len(self._queued_tasks)}, Active: {len(self._active_tasks)}/{self.max_concurrent_tasks})"
        )

        try:
            async with self.semaphore:
                self._queued_tasks.discard(asset_id)
                self._active_tasks.add(asset_id)
                logger.info(
                    f"Asset [{asset_id}] acquired ingestion slot. "
                    f"(Active: {len(self._active_tasks)}/{self.max_concurrent_tasks})"
                )
                try:
                    await process_asset_pipeline(
                        asset_id=asset_id,
                        db_path=db_path,
                        embedding_client=embedding_client,
                        kev_client=kev_client,
                    )
                finally:
                    self._active_tasks.discard(asset_id)
                    logger.info(
                        f"Asset [{asset_id}] released ingestion slot. "
                        f"(Active: {len(self._active_tasks)}/{self.max_concurrent_tasks})"
                    )
        finally:
            self._queued_tasks.discard(asset_id)
            if current_task is not None:
                self._tracked_asyncio_tasks.discard(current_task)

    async def recover_pending_assets(
        self,
        db_path: Path,
        embedding_client: EmbeddingClient,
        kev_client: KevClient,
    ) -> int:
        """
        Scans SQLite database for any assets stranded in 'queued' status
        (e.g. from an abrupt server reboot or power failure) and schedules
        them for background indexing.
        """
        conn = get_connection(db_path)
        try:
            rows = conn.execute(
                "SELECT id FROM assets WHERE status = 'queued' ORDER BY created_at ASC"
            ).fetchall()
            asset_ids = [r["id"] for r in rows]
        finally:
            conn.close()

        count = 0
        for asset_id in asset_ids:
            if asset_id not in self._queued_tasks and asset_id not in self._active_tasks:
                task = asyncio.create_task(
                    self.process_asset(
                        asset_id=asset_id,
                        db_path=db_path,
                        embedding_client=embedding_client,
                        kev_client=kev_client,
                    )
                )
                self._tracked_asyncio_tasks.add(task)
                task.add_done_callback(self._tracked_asyncio_tasks.discard)
                count += 1

        if count > 0:
            logger.info(f"Recovered {count} queued asset(s) from database for background indexing.")
        return count

    async def shutdown(self, timeout: float = 10.0) -> None:
        """
        Gracefully waits for active and queued background tasks to finish,
        up to `timeout` seconds. Cancels any remaining tasks if timeout expires.
        """
        self._shutting_down = True
        active_count = len(self._active_tasks)
        queued_count = len(self._queued_tasks)
        logger.info(
            f"Shutting down BackgroundTaskManager: {active_count} active task(s), "
            f"{queued_count} queued task(s). Waiting up to {timeout}s..."
        )

        if not self._tracked_asyncio_tasks:
            logger.info("No active background tasks to drain.")
            return

        tasks = list(self._tracked_asyncio_tasks)
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        logger.info(f"BackgroundTaskManager shutdown: {len(done)} task(s) completed, {len(pending)} pending.")

        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            logger.info(f"Cancelled {len(pending)} remaining background task(s).")

    def get_stats(self) -> Dict[str, Any]:
        """Returns snapshot statistics for monitoring and health reporting."""
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "active_tasks_count": len(self._active_tasks),
            "queued_tasks_count": len(self._queued_tasks),
            "active_asset_ids": list(self._active_tasks),
            "queued_asset_ids": list(self._queued_tasks),
            "shutting_down": self._shutting_down,
        }

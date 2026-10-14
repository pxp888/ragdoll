import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from ragdoll.config import settings

logger = logging.getLogger("ragdoll.clients.standard_reranker")


class StandardReranker:
    """
    Standard neural cross-encoder re-ranking client powered by FlashRank (ONNX).
    Provides ultra-low-latency (~5-15ms), CPU-friendly semantic re-ranking without
    requiring Kev or PyTorch.
    """

    def __init__(
        self,
        model_name: str = settings.reranker_model,
        cache_dir: Optional[str] = settings.reranker_cache_dir,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or "/tmp"
        self._ranker = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="flashrank")

    def _get_ranker(self):
        if self._ranker is None:
            from flashrank import Ranker

            logger.info(f"Initializing standard FlashRank ranker with model: {self.model_name}")
            self._ranker = Ranker(model_name=self.model_name, cache_dir=self.cache_dir)
        return self._ranker

    async def check_health(self) -> bool:
        """
        Verify that the FlashRank model can be instantiated and run.
        """
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._get_ranker)
            return True
        except Exception as e:
            logger.error(f"Standard reranker health check failed: {e}")
            return False

    async def rerank(
        self,
        query: str,
        passages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate passages against a query using FlashRank.
        Each passage in `passages` must have an 'id' and 'text'.
        Returns list of passages reordered by score in descending order.
        """
        if not passages:
            return []

        def _do_rerank() -> List[Dict[str, Any]]:
            from flashrank import RerankRequest

            ranker = self._get_ranker()
            req = RerankRequest(query=query, passages=passages)
            return ranker.rerank(req)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _do_rerank)

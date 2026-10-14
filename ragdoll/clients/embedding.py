from typing import List
import httpx
from ragdoll.config import settings


class EmbeddingClient:
    def __init__(self, base_url: str = settings.embed_url, model: str = settings.embed_model, timeout: float = settings.request_timeout):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def check_health(self) -> bool:
        """
        Verify connectivity to the embedding server.
        Fails fast if unreachable.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            # Verify the model is present or server is responding
            return "models" in data

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates 768-dim vector embeddings for a batch of strings.
        """
        if not texts:
            return []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": texts,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Expected {len(texts)} embeddings, but received {len(embeddings)}"
                )
            return embeddings

    async def embed_all(self, texts: List[str], batch_size: int = settings.batch_embed_size) -> List[List[float]]:
        """
        Embeds a list of texts in batches.
        """
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = await self.embed_batch(batch)
            all_embeddings.extend(emb)
        return all_embeddings

    async def embed_single(self, text: str) -> List[float]:
        results = await self.embed_batch([text])
        return results[0]

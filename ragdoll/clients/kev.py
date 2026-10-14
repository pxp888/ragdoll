from typing import Optional, Dict, Any, List
import httpx
from ragdoll.config import settings


class KevClient:
    def __init__(
        self,
        base_url: str = settings.kev_url,
        auth_token: str = settings.kev_auth_token,
        timeout: float = settings.request_timeout,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout

    @property
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    async def check_health(self) -> bool:
        """
        Verify connectivity to Kev 4B server.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{self.base_url}/v1/models",
                headers=self._headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return "models" in data

    async def decide_boundary(self, block_a: str, block_b: str) -> bool:
        """
        Queries Kev 4B systemone to decide if there is a semantic boundary
        between block_a and block_b.
        Returns True if a split should be created.
        """
        payload = {
            "state": f"Preceding text:\n{block_a}\n\nCandidate continuation:\n{block_b}",
            "questions": {
                "boundary": {
                    "type": "choice",
                    "instructions": "Does the subject matter shift significantly enough between these blocks to warrant a separate semantic chunk?",
                    "criteria": {
                        "continue": "The text continues the same topic or contextual narrative.",
                        "split": "The text begins a distinct topic, section, or logical transition.",
                    },
                }
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/v1/systemone",
                json=payload,
                headers=self._headers,
            )
            resp.raise_for_status()
            data = resp.json()
            boundary_ans = (
                data.get("answers", {}).get("boundary", {})
            )
            choice = boundary_ans.get("choice")
            probs = boundary_ans.get("probabilities", {})
            split_prob = probs.get("split", 0.0)

            # Split if choice is split or split probability is high
            return choice == "split" or split_prob >= 0.5

    async def score_relevance(self, query: str, passage: str) -> float:
        """
        Scores how relevant a passage is to the query using Kev 4B systemone.
        """
        payload = {
            "state": f"Query: {query}\nPassage: {passage}",
            "questions": {
                "relevance": {
                    "type": "score",
                    "instructions": "How relevant is this passage to the user query?",
                    "criteria": [
                        "irrelevant",
                        "tangentially related",
                        "directly answers or addresses query",
                    ],
                }
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/v1/systemone",
                json=payload,
                headers=self._headers,
            )
            resp.raise_for_status()
            data = resp.json()
            relevance_ans = data.get("answers", {}).get("relevance", {})
            return float(relevance_ans.get("score", 0.0))

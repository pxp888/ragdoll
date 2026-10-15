import asyncio
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

    async def _post_systemone(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST to /v1/systemone, retrying on transient network errors.
        The Kev server can stall under concurrent load; a fresh request
        usually gets served while the stalled one times out.
        shortcut: fixed 3 attempts / linear backoff; make configurable if we need per-caller tuning.
        """
        last_exc = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/v1/systemone",
                        json=payload,
                        headers=self._headers,
                    )
                    resp.raise_for_status()
                    return resp.json()
            except httpx.TransportError as e:  # timeouts + connection errors; HTTP status errors fail fast
                last_exc = e
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
        raise last_exc

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
                    "instructions": "Determine whether to place a chunk boundary before the candidate text.",
                    "criteria": {
                        "split": "Split: The candidate marks a natural break (such as a new paragraph, change of speaker, scene shift, or change of action).",
                        "continue": "Do not split: The candidate directly continues the same immediate thought, statement, or description.",
                    },
                }
            },
        }

        data = await self._post_systemone(payload)
        boundary_ans = (
            data.get("answers", {}).get("boundary", {})
        )
        choice = boundary_ans.get("choice")
        probs = boundary_ans.get("probabilities", {})
        split_prob = probs.get("split", 0.0)

        # Split if choice is split or split probability is high
        return choice == "split" or split_prob >= 0.38

    async def decide_breakpoint(
        self,
        preceding_text: str,
        candidates: List[str],
    ) -> int:
        """
        Queries Kev 4B systemone to choose the best semantic split point among multiple candidates.
        Returns the number of candidate units that should be included in the current chunk:
          0: Split before candidates[0] (chunk ends at preceding_text).
          1: Split after candidates[0] (include candidate 1).
          2: Split after candidates[1] (include candidates 1 and 2).
          ...
          k: Split after candidates[k-1] (include candidates 1..k).
          len(candidates): Continue all candidates without splitting.
        """
        if not candidates:
            return 0

        # Build candidate lines and criteria
        cand_text_lines = []
        criteria = {
            "before_1": "Split before [1]: [1] begins a new scene, speaker, action, or distinct thought.",
        }
        for idx, cand in enumerate(candidates, start=1):
            cand_text_lines.append(f"[{idx}] {cand.strip()}")
            if idx < len(candidates):
                criteria[f"after_{idx}"] = f"Split after [{idx}]: [{idx}] completes the current thought, and [{idx+1}] begins a transition."

        criteria["continue_all"] = "Keep together: all sentences belong to the same uninterrupted passage."

        next_text = "\n".join(cand_text_lines)
        payload = {
            "state": f"Preceding text:\n{preceding_text.strip()}\n\nNext sentences:\n{next_text}",
            "questions": {
                "boundary": {
                    "type": "choice",
                    "instructions": "Where is the most natural boundary to end this passage and begin a new retrieval chunk?",
                    "criteria": criteria,
                }
            },
        }

        data = await self._post_systemone(payload)
        boundary_ans = data.get("answers", {}).get("boundary", {})
        choice = boundary_ans.get("choice")

        if choice == "before_1":
            return 0
        if choice == "continue_all":
            return len(candidates)
        if choice and choice.startswith("after_"):
            try:
                k = int(choice.split("_")[1])
                return min(max(k, 1), len(candidates))
            except (ValueError, IndexError):
                return len(candidates)

        return len(candidates)

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

        data = await self._post_systemone(payload)
        relevance_ans = data.get("answers", {}).get("relevance", {})
        return float(relevance_ans.get("score", 0.0))

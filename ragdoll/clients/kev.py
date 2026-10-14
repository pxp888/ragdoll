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
            "before_1": "Between preceding text and [1].",
        }
        for idx, cand in enumerate(candidates, start=1):
            cand_text_lines.append(f"[{idx}] {cand.strip()}")
            if idx < len(candidates):
                criteria[f"after_{idx}"] = f"Between [{idx}] and [{idx+1}]."

        criteria["continue_all"] = "No significant shift; all sentences continue the topic."

        next_text = "\n".join(cand_text_lines)
        payload = {
            "state": f"Preceding text:\n{preceding_text.strip()}\n\nNext sentences:\n{next_text}",
            "questions": {
                "boundary": {
                    "type": "choice",
                    "instructions": "Where does a significant semantic shift occur to start a new chunk?",
                    "criteria": criteria,
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

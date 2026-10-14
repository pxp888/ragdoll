from datetime import datetime
from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class StoreAssetResponse(BaseModel):
    asset_id: str
    filename: str
    size_bytes: int
    status: str
    created_at: str


class AssetStatusResponse(BaseModel):
    asset_id: str
    status: str
    coarse_chunks: int
    fine_chunks: int
    error: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    stage: Literal["coarse", "fine"] = "fine"
    rerank: bool = False
    include_parent_context: bool = True


class SearchResultItem(BaseModel):
    chunk_id: int
    asset_id: str
    filename: str
    score: float
    text: str
    parent_context: Optional[str] = None
    created_at: str


class SearchResponse(BaseModel):
    results: List[SearchResultItem]


class HealthResponse(BaseModel):
    status: str
    embeddinggemma2: str
    kev_4b: str
    details: Optional[dict] = None

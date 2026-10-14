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
    rerank_method: Optional[Literal["kev", "standard", "none"]] = None
    include_parent_context: bool = True
    candidate_k: Optional[int] = None


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    stage: Literal["coarse", "fine"] = "fine"
    include_parent_context: bool = True
    candidate_k: Optional[int] = None


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
    rerank_method: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    embeddinggemma2: str
    kev_4b: str
    standard_reranker: Optional[str] = None
    details: Optional[dict] = None


class DeleteAssetResponse(BaseModel):
    deleted: bool = True
    asset_id: str
    message: str = "Asset deleted successfully"


class DeleteAllAssetsResponse(BaseModel):
    deleted_count: int
    apikey: str
    asset_ids: List[str]
    message: str

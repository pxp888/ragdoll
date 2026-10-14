from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # External services
    # kev_url: str = "http://192.168.0.12:8008"
    kev_url: str = "http://192.168.0.189:8080"
    kev_auth_token: str = "hellothere"
    embed_url: str = "http://192.168.0.12:11434"
    embed_model: str = "embeddinggemma-2:740m"

    # Standard reranker (FlashRank / cross-encoder)
    reranker_model: str = "ms-marco-TinyBERT-L-2-v2"
    reranker_cache_dir: Optional[str] = None

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Storage and database
    db_path: Path = Path("data/ragdoll.db")
    storage_path: Path = Path("data/uploads")

    # Pipeline tuning
    concurrency_limit: int = 4
    batch_embed_size: int = 32
    coarse_min_chars: int = 2000
    coarse_max_chars: int = 4000
    fine_min_chars: int = 300
    fine_max_chars: int = 1500

    # Network timeouts (seconds)
    request_timeout: float = 30.0

    # Test & probe toggles
    skip_startup_probe: bool = False

    model_config = SettingsConfigDict(
        env_prefix="RAGDOLL_",
        env_file=".env",
        extra="ignore"
    )


settings = Settings()

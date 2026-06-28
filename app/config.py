from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    app_env: str = "local"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    use_openai_llm: bool = False
    database_path: Path = Field(default=Path("data/app.db"))
    knowledge_base_path: Path = Field(default=Path("data/knowledge_base/raw/knowledge_base.json"))
    eval_dataset_path: Path = Field(default=Path("data/eval/eval_dataset.json"))
    retrieval_top_k: int = 4
    answer_threshold: float = 0.18
    clarify_threshold: float = 0.09

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Primary and fallback Gemini API keys
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_api_key_2: str = Field(default="", alias="GEMINI_API_KEY_2")
    gemini_api_key_3: str = Field(default="", alias="GEMINI_API_KEY_3")
    gemini_embed_model: str = Field(default="text-embedding-004", alias="GEMINI_EMBED_MODEL")
    gemini_gen_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_GEN_MODEL")

    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index: str = Field(default="document-clustering-lab", alias="PINECONE_INDEX")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")

    cache_db_path: str = Field(default="./data/embedding_cache.db", alias="CACHE_DB_PATH")
    cors_origins: str = Field(default="http://localhost:5173,http://127.0.0.1:5173", alias="CORS_ORIGINS")
    similarity_threshold: float = Field(default=0.65, alias="SIMILARITY_THRESHOLD")

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def get_available_gemini_keys(self) -> list[str]:
        """Return list of all configured Gemini API keys (non-empty)."""
        keys = []
        if self.gemini_api_key:
            keys.append(self.gemini_api_key)
        if self.gemini_api_key_2:
            keys.append(self.gemini_api_key_2)
        if self.gemini_api_key_3:
            keys.append(self.gemini_api_key_3)
        return keys

settings = Settings()

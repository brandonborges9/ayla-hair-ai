# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Supabase
    supabase_url: str
    supabase_service_role_key: str  # pour backend
    supabase_anon_key: Optional[str] = None  # pour frontend si besoin
    supabase_bucket: str = "images"          # bucket par défaut

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # ignore si Render envoie une var inutile
    )

settings = Settings()

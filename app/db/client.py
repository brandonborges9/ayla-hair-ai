# app/db/client.py

from supabase import create_client, Client
from datetime import datetime
import re

from app.core.config import settings

# Client Supabase (backend → on utilise la Service Role Key)
supabase: Client = create_client(
    settings.supabase_url,
    settings.supabase_service_role_key
)

def _safe_filename(name: str) -> str:
    name = name.strip().lower().replace(" ", "_")
    name = re.sub(r"[^a-z0-9._-]", "", name)
    return name or "upload.jpg"

def upload_image_to_supabase(file_bytes: bytes, filename: str, content_type: str = "image/jpeg") -> str:
    """
    Upload l'image dans le bucket défini (default = images) et retourne son URL publique.
    """
    safe_name = _safe_filename(filename or "upload.jpg")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    path = f"analyze_uploads/{timestamp}_{safe_name}"

    supabase.storage.from_(settings.supabase_bucket).upload(
        path,
        file_bytes,
        {"content-type": content_type or "image/jpeg"}
    )

    public_url = f"{settings.supabase_url}/storage/v1/object/public/{settings.supabase_bucket}/{path}"
    return public_url

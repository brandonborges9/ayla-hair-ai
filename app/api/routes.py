from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional

from app.models.inference import analyze_image, save_analysis_to_supabase
from app.db.client import upload_image_to_supabase
from app.core.config import settings

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "supabase_url": settings.supabase_url,
        "bucket": settings.supabase_bucket
    }

@router.post("/analyze-hair/")
async def analyze_hair(
    image: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
):
    # 1) Validation contenu
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    # 2) Lire les bytes UNE SEULE FOIS
    file_bytes = await image.read()

    # 3) Upload dans Supabase Storage
    image_url = upload_image_to_supabase(
        file_bytes=file_bytes,
        filename=image.filename or "upload.jpg",
        content_type=image.content_type,
    )

    # 4) Analyse (sur bytes, pas UploadFile)
    result = await analyze_image(file_bytes)

    # 5) Sauvegarde en base
    supabase_response = save_analysis_to_supabase(
        result_json=result,
        image_url=image_url,
        user_id=user_id,
    )

    return {
        "analysis": result,
        "image_url": image_url,
        "saved_in_supabase": supabase_response.data if hasattr(supabase_response, "data") else None,
    }

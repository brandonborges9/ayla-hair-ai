from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.inference import analyze_image

router = APIRouter()


@router.post("/analyze-hair/")
async def analyze_hair(image: UploadFile = File(...)):
    # Vérifie que le fichier est bien une image
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    # Appelle la fonction d'analyse
    result = await analyze_image(image)
    return result

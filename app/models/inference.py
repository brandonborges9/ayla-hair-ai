# app/models/inference.py

from typing import Dict, List, Optional
from PIL import Image
import io
from datetime import datetime
import uuid

from app.db.client import supabase  # pour l'enregistrement en BDD

# IA (stub)
import torch
from torchvision import transforms

# ---------- Préprocessing ----------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

HAIR_TYPES: List[str] = ["lisse", "ondulé", "bouclé", "frisé", "crépu"]
SCALP_ISSUES: List[str] = ["pellicules", "sécheresse", "excès de sébum", "rougeurs", "psoriasis"]


def stub_classify(tensor_image: torch.Tensor) -> Dict:
    """
    Classification très simple (stub) :
      - type : basé sur ratio (W/H) *à titre de démo*
      - problème : basé sur la moyenne du canal vert
    """
    _, h, w = tensor_image.shape
    hair_type = "lisse" if w > h else "bouclé"

    green_mean = tensor_image[1].mean().item()
    issues = []
    if green_mean < 0.4:
        issues.append("sécheresse")

    return {"type": hair_type, "problèmes": issues}


async def analyze_image(contents: bytes) -> Dict:
    """
    Reçoit des bytes d'image, exécute le préprocessing et renvoie un diagnostic stub.
    """
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    width, height = img.size

    tensor = preprocess(img)
    diag = stub_classify(tensor)

    recs = [f"Routine de base pour cheveux {diag['type']}"]
    for issue in diag["problèmes"]:
        if issue == "sécheresse":
            recs.append("Masque hydratant intensif – poser 30 min")
        elif issue == "pellicules":
            recs.append("Shampoing anti-pelliculaire – 2×/semaine")

    return {
        "type": diag["type"],
        "dimensions": {"width": width, "height": height},
        "problèmes": diag["problèmes"],
        "recommandations": recs,
    }

def save_analysis_to_supabase(result_json: dict, image_url: str, user_id: Optional[str] = None):
    # Valide et normalise un éventuel user_id en UUID
    uid: Optional[str] = None
    if user_id:
        try:
            uid = str(uuid.UUID(user_id))
        except Exception:
            uid = None  # invalide => on ignore

    data = {
        "result_json": result_json,
        "image_url": image_url,
        "created_at": datetime.utcnow().isoformat(),
    }
    if uid:
        data["user_id"] = uid  # on n’envoie la clé que si UUID valide

    return supabase.table("analyses").insert(data).execute()
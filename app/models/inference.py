# app/models/inference.py

from typing import Dict
from fastapi import UploadFile
from PIL import Image
import io

# Pour le préprocessing
import torch
from torchvision import transforms

# Définition du pipeline de transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalisation ImageNet (stub pour l’instant)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Mapping stub pour le type de cheveux et problèmes
HAIR_TYPES = ["lisse", "ondulé", "bouclé", "frisé", "crépu"]
SCALP_ISSUES = ["pellicules", "sécheresse", "excès de sébum", "rougeurs", "psoriasis"]


def stub_classify(tensor_image: torch.Tensor) -> Dict:
    """
    Classification simpliste :
     - type : selon ratio hauteur/largeur
     - problèmes : selon moyenne du pixel vert (stub)
    """
    _, h, w = tensor_image.shape
    # Type de cheveux basé sur ratio
    if w > h:
        hair_type = "lisse"
    else:
        hair_type = "bouclé"
    # Détecte un problème si la moyenne verte est faible
    green_mean = tensor_image[1].mean().item()
    issues = []
    if green_mean < 0.4:
        issues.append("sécheresse")
    return {
        "type": hair_type,
        "problèmes": issues
    }


async def analyze_image(image: UploadFile) -> Dict:
    """
    Lit l'image, préprocesse, applique un stub de classification,
    puis enrichit la réponse avec recommandations et dimensions.
    """
    # Lecture
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Dimensions originales
    width, height = img.size

    # Préprocessing et ajout batch dimension
    tensor = preprocess(img)

    # Classification stub
    diag = stub_classify(tensor)

    # Génération de recommandations basées sur le diagnostic
    recs = []
    # Reco de routine simple
    recs.append(f"Routine de base pour cheveux {diag['type']}")
    for issue in diag["problèmes"]:
        if issue == "sécheresse":
            recs.append("Masque hydratant intensif – poser 30 min")
        elif issue == "pellicules":
            recs.append("Shampoing anti-pelliculaire – 2×/semaine")

    # Emballer la réponse
    return {
        "type": diag["type"],
        "dimensions": {"width": width, "height": height},
        "problèmes": diag["problèmes"],
        "recommandations": recs
    }

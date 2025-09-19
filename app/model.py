import json, os, torch, torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# -----------------------------
# Réseau principal (backbone + 2 têtes)
# -----------------------------
class HairNet(nn.Module):
    def __init__(self, n_type=3, n_state=3):
        super().__init__()

        # 1) On charge MobileNetV3 pré-entraîné sur ImageNet
        #    -> il sait déjà extraire des features génériques sur des millions d’images
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # On récupère le nombre de neurones dans la dernière couche du backbone
        in_f = self.backbone.classifier[-1].in_features

        # On enlève la dernière couche (classif ImageNet) pour ne garder que l'extracteur de features
        self.backbone.classifier[-1] = nn.Identity()

        # 2) On ajoute deux "têtes" linéaires :
        #    - une pour la classification du type de cheveux
        #    - une pour la classification de l'état du cheveu
        self.head_type  = nn.Linear(in_f, n_type)
        self.head_state = nn.Linear(in_f, n_state)

    def forward(self, x):
        # Étape 1 : extraction de features avec le backbone
        features = self.backbone(x)

        # Étape 2 : chaque tête prédit sa tâche spécifique
        return {
            "type_logits":  self.head_type(features),   # sortie pour hair type
            "state_logits": self.head_state(features),  # sortie pour hair state
        }


# -----------------------------
# Classe utilitaire pour charger modèle + classes + prédire
# -----------------------------
class Predictor:
    def __init__(self, weights_path: str, classes_type_path: str, classes_state_path: str, device: str = "cpu"):
        self.device = device

        # Chargement des classes (labels) depuis fichiers JSON
        with open(classes_type_path) as f:
            self.type_classes = json.load(f)
        with open(classes_state_path) as f:
            self.state_classes = json.load(f)

        # Création du modèle
        model = HairNet(n_type=len(self.type_classes), n_state=len(self.state_classes))

        # Chargement des poids entraînés (fichier .pt)
        ckpt = torch.load(weights_path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt)   # supporte dict direct ou checkpoint complet
        model.load_state_dict(state, strict=False)
        model.eval().to(self.device)           # mode évaluation = pas de dropout, pas de grad
        self.model = model

        # Normalisation ImageNet : chiffres "moyenne" et "écart-type" par canal RGB
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    @torch.inference_mode()  # pas de gradients, plus rapide
    def predict(self, img_chw_float):
        """
        Args:
            img_chw_float : image au format Tensor CHW (3, H, W) valeurs [0,1]
        Returns:
            dict : prédictions hair_type + hair_state avec probabilités
        """

        # 1) On normalise comme ImageNet attendait (même stats que pré-entraînement)
        x = (img_chw_float.unsqueeze(0) - self.mean) / self.std
        x = x.to(self.device)

        # 2) Passage dans le réseau
        out = self.model(x)

        # 3) Softmax pour convertir logits -> probabilités (somme = 1)
        p_type  = torch.softmax(out["type_logits"], dim=1)[0].cpu().numpy()
        p_state = torch.softmax(out["state_logits"], dim=1)[0].cpu().numpy()

        # 4) On récupère l’index + la confiance max
        idx_t, conf_t = int(p_type.argmax()), float(p_type.max())
        idx_s, conf_s = int(p_state.argmax()), float(p_state.max())

        return {
            "type": {
                "label": self.type_classes[idx_t],
                "confidence": conf_t,
                "probs": {c: float(p_type[i]) for i,c in enumerate(self.type_classes)}
            },
            "state": {
                "label": self.state_classes[idx_s],
                "confidence": conf_s,
                "probs": {c: float(p_state[i]) for i,c in enumerate(self.state_classes)}
            }
        }

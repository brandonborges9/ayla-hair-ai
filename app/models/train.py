# models/train.py
import os, json, argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from torchvision.transforms import RandomErasing


def mixup(x, y, alpha=0.2):
    if alpha <= 0: return x, y, None
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x2, y2 = x[idx], y[idx]
    return lam * x + (1 - lam) * x2, (y, y2), lam


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__();
        self.weight = weight;
        self.gamma = gamma

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.softmax(logits, 1).gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
        return ((1 - pt) ** self.gamma * ce).mean()


# ---------------- Utils ----------------
def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda"), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


# --------------- Dataset ---------------
class HairDataset(Dataset):
    """
    CSV: relative_path, state (good/medium/bad)
    type est déduit du chemin: split/type/file.jpg
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file).copy()
        self.root_dir = root_dir
        self.transform = transform

        # --- Nettoyage de la colonne "state"
        self.df["state"] = (
            self.df["state"].astype(str).str.strip().str.lower()
            .replace({"nan": "", "none": ""})
        )

        # --- On ignore les lignes sans label (pour l’entraînement)
        self.df = self.df[self.df["state"].isin(["good", "medium", "bad"])].reset_index(drop=True)

        # --- Déduction du type (curly/wavy/straight)
        self.df["type"] = self.df["relative_path"].apply(lambda p: p.split("/")[1])

        # --- Classes et mappings
        STATE_ORDER = ["bad", "medium", "good"]  # ordre fixe pour la cohérence
        self.state_classes = [c for c in STATE_ORDER if c in set(self.df["state"])]
        self.type_classes = sorted(self.df["type"].unique().tolist())

        self.type2idx = {c: i for i, c in enumerate(self.type_classes)}
        self.state2idx = {c: i for i, c in enumerate(self.state_classes)}

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.root_dir, row["relative_path"])).convert("RGB")
        if self.transform: img = self.transform(img)
        y_t = self.type2idx[row["type"]]
        y_s = self.state2idx[row["state"]]
        return img, y_t, y_s


# --------------- Model -----------------
class ResNetMultiHead(nn.Module):
    def __init__(self, n_type, n_state, dropout_p=0.2):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout_p)
        self.head_type = nn.Linear(in_f, n_type)
        self.head_state = nn.Linear(in_f, n_state)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)  # régularisation
        return self.head_type(feat), self.head_state(feat)


# --------- Train/Val one epoch ---------
def run_epoch(model, loader, device, crit_type, crit_state, optimizer=None):
    if len(loader.dataset) == 0:
        return 0.0, float("nan"), np.array([]), np.array([])
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    tot_loss, n = 0.0, 0
    y_true_s, y_pred_s = [], []

    for x, y_t, y_s in loader:
        x, y_t, y_s = x.to(device), y_t.to(device), y_s.to(device)
        with torch.set_grad_enabled(train_mode):
            if train_mode:
                # --- MixUp : mélange d'images et labels pour régulariser ---
                x, (y_ta, y_tb), lam = mixup(x, y_t, alpha=0.2)
                logit_t, logit_s = model(x)
                loss = (
                        lam * crit_type(logit_t, y_ta)
                        + (1 - lam) * crit_type(logit_t, y_tb)
                        + crit_state(logit_s, y_s)
                )
            else:
                # --- Évaluation sans MixUp ---
                logit_t, logit_s = model(x)
                loss = crit_type(logit_t, y_t) + crit_state(logit_s, y_s)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        bs = x.size(0);
        tot_loss += loss.item() * bs;
        n += bs
        y_true_s.extend(y_s.detach().cpu().numpy())
        y_pred_s.extend(logit_s.argmax(1).detach().cpu().numpy())

    return tot_loss / max(1, n), f1_score(y_true_s, y_pred_s, average="macro"), \
        np.array(y_true_s), np.array(y_pred_s)


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--output", type=str, default="outputs/baseline")
    ap.add_argument("--csv", type=str, default=os.path.join("app", "data", "data", "labels_state.csv"))
    ap.add_argument("--root", type=str, default=os.path.join("app", "data", "data"))
    ap.add_argument("--weights", type=str, default="data/state_class_weights.json")
    ap.add_argument("--early-stop", type=int, default=5, help="patience (epochs) sur F1 val")
    ap.add_argument("--focal-gamma", type=float, default=1.5, help="gamma pour la FocalLoss (1.0=CE)")
    ap.add_argument("--save-miscls", action="store_true", help="sauver les erreurs de validation dans un dossier")

    args = ap.parse_args()

    # Création du dossier de sortie + sous-dossier pour erreurs (si activé)
    os.makedirs(args.output, exist_ok=True)
    ERR_DIR = os.path.join(args.output, "val_errors")
    if args.save_miscls:
        os.makedirs(ERR_DIR, exist_ok=True)

    device, dev_name = pick_device()
    print(f"[Device] PyTorch {torch.__version__} | Using: {dev_name}")
    try:
        torch.set_float32_matmul_precision("high")
    except:
        pass

    # ---------- Dataframes ----------
    df = pd.read_csv(args.csv)

    def _clean_col(s):
        return s.astype(str).str.strip().str.lower().replace({"nan": "", "none": ""})

    df["state"] = _clean_col(df["state"])
    is_labeled = df["state"].isin(["good", "medium", "bad"])

    df_train = df[df["relative_path"].str.startswith("train/") & is_labeled].reset_index(drop=True)
    df_val = df[df["relative_path"].str.startswith("val/") & is_labeled].reset_index(drop=True)
    if len(df_val) == 0:
        # crée une petite val (20% par classe, min 2) à partir du train labellisé
        parts = []
        for s, g in df_train.groupby("state"):
            n = max(2, int(0.2 * len(g)))
            parts.append(g.sample(n=min(n, len(g)), random_state=42))
        df_val = pd.concat(parts).drop_duplicates().reset_index(drop=True)
        # retire ces échantillons du train pour éviter les fuites
        df_train = pd.merge(df_train, df_val[["relative_path"]], on="relative_path", how="left", indicator=True)
        df_train = df_train[df_train["_merge"] == "left_only"].drop(columns=["_merge"]).reset_index(drop=True)

    df_train.to_csv(os.path.join(args.output, "train.csv"), index=False)
    df_val.to_csv(os.path.join(args.output, "val.csv"), index=False)
    print(f"[Counts] train={len(df_train)} | val={len(df_val)}")

    # ---------- Transforms ----------
    tf_train = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        RandomErasing(p=0.35, scale=(0.02, 0.08), ratio=(0.3, 3.3))
    ])
    tf_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = HairDataset(os.path.join(args.output, "train.csv"), args.root, tf_train)
    val_ds = HairDataset(os.path.join(args.output, "val.csv"), args.root, tf_val)

    pin = (dev_name == "cuda")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    # ---------- Model ----------
    model = ResNetMultiHead(len(train_ds.type_classes), len(train_ds.state_classes), dropout_p=0.4).to(device)

    # Freeze 2 premières époques pour stabiliser (puis on dégèle)
    for p in model.backbone.parameters(): p.requires_grad = False

    # Losses
    weights_path = args.weights if os.path.exists(args.weights) else os.path.join(args.root, "state_class_weights.json")
    with open(weights_path) as f:
        wdict = json.load(f)
    state_w = torch.tensor([float(wdict[c]) for c in train_ds.state_classes], dtype=torch.float32).to(device)
    crit_state = FocalLoss(weight=state_w, gamma=args.focal_gamma)

    # Optim & schedulers
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_f1, no_improve = -1.0, 0
    history = {"train_loss": [], "val_loss": [], "val_f1_state": []}

    for epoch in range(1, args.epochs + 1):
        # Dégel progressif à partir de l'epoch 3
        if epoch == 5:
            for p in model.backbone.parameters():
                p.requires_grad = True
            for g in optimizer.param_groups:
                g["lr"] = 2e-4

        print(f"\nEpoch {epoch}/{args.epochs}")
        crit_type = nn.CrossEntropyLoss(label_smoothing=0.08)

        tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, device, crit_type, crit_state, optimizer)
        va_loss, va_f1, y_true_s, y_pred_s = run_epoch(model, val_dl, device, crit_type, crit_state, optimizer=None)

        history["train_loss"].append(tr_loss);
        history["val_loss"].append(va_loss);
        history["val_f1_state"].append(va_f1)
        print(f"Train | loss={tr_loss:.4f}  (F1-state={tr_f1:.4f})")
        print(f"Val   | loss={va_loss:.4f}  F1-state={va_f1:.4f}")

        # Scheduler + early stop
        scheduler.step(va_f1)
        if va_f1 > best_f1:
            best_f1, no_improve = va_f1, 0
            torch.save({
                "state_dict": model.state_dict(),
                "type_classes": train_ds.type_classes,
                "state_classes": train_ds.state_classes,
            }, os.path.join(args.output, "model_best.pt"))
            print(f"  -> New best model saved (F1-state={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                print(f"Early stopping (no improve {args.early_stop} epochs).")
                break

    # ---- Reports / Plots ----
    cm = confusion_matrix(y_true_s, y_pred_s, labels=list(range(len(train_ds.state_classes))))
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues");
    plt.title("Confusion Matrix - State (val)")
    plt.xticks(range(len(train_ds.state_classes)), train_ds.state_classes, rotation=45)
    plt.yticks(range(len(train_ds.state_classes)), train_ds.state_classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout();
    fig.savefig(os.path.join(args.output, "cm_state_val.png"));
    plt.close(fig)

    with open(os.path.join(args.output, "report_state_val.txt"), "w") as f:
        f.write(classification_report(y_true_s, y_pred_s, target_names=train_ds.state_classes, digits=3))

    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump({"best_val_f1_state": float(best_f1),
                   "train_loss": history["train_loss"],
                   "val_loss": history["val_loss"],
                   "val_f1_state": history["val_f1_state"]}, f, indent=2)

    # Sauvegarde CSV des prédictions val
    val_df = val_ds.df.copy()
    val_df["y_true"] = y_true_s
    val_df["y_pred"] = y_pred_s
    val_df.to_csv(os.path.join(args.output, "val_preds.csv"), index=False)

    # Sauvegarde des images mal classées (optionnel avec --save-miscls)
    if args.save_miscls:
        from shutil import copy2
        mis = val_df[val_df["y_true"] != val_df["y_pred"]]
        # map index->label lisible
        idx2state = {i: c for c, i in val_ds.state2idx.items()}
        for _, r in mis.iterrows():
            rel = r["relative_path"]
            p = os.path.join(args.root, rel)
            y_t = idx2state[int(r["y_true"])]
            y_p = idx2state[int(r["y_pred"])]
            dst = os.path.join(ERR_DIR, f"{y_t}__as__{y_p}__{os.path.basename(rel)}")
            try:
                copy2(p, dst)
            except:
                pass

    print("\n✅ Training finished. Outputs in:", args.output)


if __name__ == "__main__":
    main()

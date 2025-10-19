# app/models/evaluate.py
import os, argparse, json
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from train import ResNetMultiHead

def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda"), "cuda"
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"

def save_confusion(cm, class_names, out_png, title):
    fig = plt.figure(figsize=(5,5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

@torch.no_grad()
def predict_paths(model, device, root, rel_paths, type_classes, state_classes, tta=False,
                  bias_vec=None, threshold=0.0):
    base_tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    out_json, y_pred_state = [], []
    for rel in rel_paths:
        img = Image.open(os.path.join(root, rel)).convert("RGB")
        views = [base_tf(img)]
        if tta: views.append(base_tf(TF.hflip(img)))
        x = torch.stack(views).to(device)

        logit_t, logit_s = model(x)
        pt = torch.softmax(logit_t,1).mean(0).cpu().numpy()
        ps = torch.softmax(logit_s,1).mean(0).cpu().numpy()

        # ---- BIAIS D’INFÉRENCE + SEUIL D’INCERTITUDE ----
        ps_adj = ps.copy()
        if bias_vec is not None:
            ps_adj = ps_adj * bias_vec
            ps_adj = ps_adj / ps_adj.sum()

        pred_idx = int(ps_adj.argmax())
        conf_adj = float(ps_adj[pred_idx])
        uncertain = conf_adj < threshold

        y_pred_state.append(pred_idx)
        out_json.append({
            "image": rel,
            "hair_type":  {"label": type_classes[int(pt.argmax())],  "confidence": float(pt.max())},
            "hair_state": {
                "label": state_classes[pred_idx],
                "confidence": conf_adj,
                "uncertain": bool(uncertain),
                "confidence_raw": float(ps.max())
            }
        })
    return out_json, np.array(y_pred_state)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv",  default="app/data/data/labels_state.csv")
    ap.add_argument("--root", default="app/data/data")
    ap.add_argument("--split", choices=["val","test","train"], default="test")
    ap.add_argument("--n", type=int, default=0, help="0=toutes les images du split")
    ap.add_argument("--tta", action="store_true", help="flip TTA")
    ap.add_argument("--bias", type=str, default="1.15,1.05,0.85",
                    help="poids multiplicatifs [bad,good,medium] appliqués aux proba (ex: '1.15,1.05,0.85')")
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="seuil d’incertitude sur la proba ajustée")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    device, dev = pick_device(); print("[Device]", dev)

    ckpt = torch.load(args.ckpt, map_location=device)

    # ⚠️ Garde exactement l’ordre appris pendant l’entraînement
    type_classes = list(ckpt["type_classes"])
    state_classes = list(ckpt["state_classes"])  # NE PAS trier

    model = ResNetMultiHead(len(type_classes), len(state_classes)).to(device)
    model.load_state_dict(ckpt["state_dict"]);
    model.eval()

    outdir = args.outdir or os.path.dirname(args.ckpt) or "."
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    sdf = df[df["relative_path"].str.startswith(f"{args.split}/")].copy()
    rels = sdf["relative_path"].tolist()
    if args.n > 0: rels = rels[:args.n]
    sdf = sdf[sdf["state"].isin(state_classes)].reset_index(drop=True)
    rels = sdf["relative_path"].tolist()
    if len(rels) == 0:
        raise ValueError(f"Aucune image annotée trouvée dans le split '{args.split}'. Vérifie labels_state.csv.")
    # y_true en indices
    state2idx = {s:i for i,s in enumerate(state_classes)}
    y_true = sdf["state"].map(state2idx).values[:len(rels)]


    # bias vector (accepte "bad=1.15,good=1.05,medium=0.85" OU "1.15,1.05,0.85" dans l'ordre du ckpt)
    if "=" in args.bias:
        pairs = [s.strip() for s in args.bias.split(",") if s.strip()]
        bias_map = {k.strip(): float(v) for k, v in (p.split("=") for p in pairs)}
        bias_vals = np.array([bias_map.get(c, 1.0) for c in state_classes], dtype=np.float32)
    else:
        bias_vals = np.array([float(x) for x in args.bias.split(",")], dtype=np.float32)
        if len(bias_vals) != len(state_classes):
            raise ValueError(f"--bias must have {len(state_classes)} values (ckpt order: {state_classes}).")

    # prédictions
    preds_json, y_pred = predict_paths(
        model, device, args.root, rels, type_classes, state_classes,
        tta=args.tta, bias_vec=bias_vals, threshold=args.threshold
    )

    # métriques
    f1m = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=state_classes, digits=3)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(state_classes))))
    save_confusion(cm, state_classes, os.path.join(outdir, f"cm_state_{args.split}.png"),
                   f"Confusion Matrix - State ({args.split})")

    # sauvegarde
    with open(os.path.join(outdir, f"preds_{args.split}.json"), "w", encoding="utf-8") as f:
        json.dump(preds_json, f, indent=2, ensure_ascii=False)
    with open(os.path.join(outdir, f"metrics_{args.split}.json"), "w") as f:
        json.dump({"f1_macro_state": float(f1m), "n_samples": int(len(rels)),
                   "tta": bool(args.tta), "bias": args.bias, "threshold": args.threshold}, f, indent=2)
    with open(os.path.join(outdir, f"report_state_{args.split}.txt"), "w") as f:
        f.write(report)

    print(json.dumps(preds_json[:min(10,len(preds_json))], indent=2, ensure_ascii=False))
    print(f"\nF1(macro) state [{args.split}]: {f1m:.4f}")
    print("Saved metrics/report/CM to:", outdir)

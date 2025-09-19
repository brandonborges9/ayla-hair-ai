# app/models/evaluate.py
import os, argparse, json
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from train import ResNetMultiHead  # même modèle que train.py

def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda"), "cuda"
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"

@torch.no_grad()
def predict_paths(model, device, root, rel_paths, type_classes, state_classes):
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    out = []
    for rel in rel_paths:
        p = os.path.join(root, rel)
        img = Image.open(p).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        logit_t, logit_s = model(x)
        pt, ps = torch.softmax(logit_t,1)[0].cpu().numpy(), torch.softmax(logit_s,1)[0].cpu().numpy()
        out.append({
            "image": rel,
            "hair_type":  {"label": type_classes[pt.argmax()],  "confidence": float(pt.max())},
            "hair_state": {"label": state_classes[ps.argmax()], "confidence": float(ps.max())},
        })
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default="app/data/data/labels_state.csv")
    ap.add_argument("--root", default="app/data/data")
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--n", type=int, default=30, help="nb d’images à évaluer (0 = toutes)")
    args = ap.parse_args()

    device, name = pick_device()
    print("[Device]", name)

    ckpt = torch.load(args.ckpt, map_location=device)
    type_classes, state_classes = ckpt["type_classes"], ckpt["state_classes"]
    model = ResNetMultiHead(len(type_classes), len(state_classes)).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    df = pd.read_csv(args.csv)
    rels = df[df["relative_path"].str.startswith(f"{args.split}/")]["relative_path"].tolist()
    if args.n > 0: rels = rels[:args.n]

    res = predict_paths(model, device, args.root, rels, type_classes, state_classes)
    print(json.dumps(res, indent=2, ensure_ascii=False))

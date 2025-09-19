import os, io, csv, argparse, hashlib
import pandas as pd
from PIL import Image
import requests
from tqdm import tqdm

ROOT = "app/data/data"                     # dossier racine des images
CSV_LABELS = os.path.join(ROOT, "labels_state.csv")

VALID_SPLITS = {"train","val","test"}
VALID_TYPES  = {"curly","straight","wavy"}
VALID_STATES = {"good","medium","bad"}
VALID_CT = {"image/jpeg":".jpg","image/jpg":".jpg","image/png":".png","image/webp":".webp","image/bmp":".bmp"}

def safe_filename(url, fallback_prefix="img"):
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"{fallback_prefix}_{h}.jpg"

def download_to_path(url, out_path):
    r = requests.get(url, timeout=20, stream=True)
    r.raise_for_status()
    ct = r.headers.get("Content-Type","").split(";")[0].strip().lower()
    ext = VALID_CT.get(ct, os.path.splitext(url)[1].lower() or ".jpg")
    data = r.content
    # Convertit en JPEG RGB (uniformise le format)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, format="JPEG", quality=92)
    return out_path

def main(args):
    df = pd.read_csv(args.csv)
    required = {"url","split","type","state"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # normalise colonnes
    cols = {c.lower():c for c in df.columns}
    df = df.rename(columns=cols)
    for col in ["split","type","state"]:
        df[col] = df[col].str.lower().str.strip()

    # filtres
    if args.only_bad:
        df = df[df["state"] == "bad"]

    # validations simples
    bad_rows = df[~df["split"].isin(VALID_SPLITS) | ~df["type"].isin(VALID_TYPES) | ~df["state"].isin(VALID_STATES)]
    if not bad_rows.empty:
        raise ValueError("Invalid rows detected (split/type/state). Fix your CSV first.")

    rows_new = []
    errors = []

    print(f"Downloading {len(df)} images …")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        split, typ, state = row["split"], row["type"], row["state"]
        url = row["url"].strip()
        # destination
        fname = str(row.get("filename", "")).strip()
        if not fname or fname == "nan":
            fname = safe_filename(url, fallback_prefix=f"{typ}_{state}")
        # force .jpg (on convertit tout en JPEG)
        base, _ = os.path.splitext(fname)
        fname = base + ".jpg"

        rel_path = os.path.join(split, typ, fname).replace("\\","/")
        out_path = os.path.join(ROOT, rel_path)

        if os.path.exists(out_path) and not args.overwrite:
            # déjà présent → on ajoute juste la ligne de label si absente
            rows_new.append({"relative_path": rel_path, "state": state})
            continue

        try:
            download_to_path(url, out_path)
            rows_new.append({"relative_path": rel_path, "state": state})
        except Exception as e:
            errors.append((url, str(e)))

    # met à jour labels_state.csv
    if rows_new:
        os.makedirs(os.path.dirname(CSV_LABELS), exist_ok=True)
        if os.path.exists(CSV_LABELS):
            df_lab = pd.read_csv(CSV_LABELS)
        else:
            df_lab = pd.DataFrame(columns=["relative_path","state"])
        df_out = pd.concat([df_lab, pd.DataFrame(rows_new)], ignore_index=True)
        df_out.drop_duplicates(subset=["relative_path"], inplace=True)
        df_out.to_csv(CSV_LABELS, index=False)
        print(f"labels_state.csv updated → {CSV_LABELS} (+{len(rows_new)} rows merged)")

    if errors:
        print("\nSome downloads failed:")
        for u, msg in errors[:20]:
            print(" -", u, "->", msg)
        if len(errors) > 20:
            print(f"… and {len(errors)-20} more")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="app/data/new_images.csv")
    ap.add_argument("--overwrite", action="store_true", help="ré-écrit les fichiers existants")
    ap.add_argument("--only-bad", action="store_true", help="ne traite que les lignes state=bad")
    args = ap.parse_args()
    main(args)

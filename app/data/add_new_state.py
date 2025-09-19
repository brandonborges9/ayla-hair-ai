# app/data/add_new_state.py
import os
import argparse
import pandas as pd

CSV  = "app/data/data/labels_state.csv"
ROOT = "app/data/data/train"
VALID_STATES = {"good", "medium", "bad"}
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, choices=VALID_STATES,
                    help="état à assigner aux nouvelles images trouvées dans train/")
    ap.add_argument("--pattern", default="new_",
                    help="motif que le nom du fichier doit contenir (défaut: 'new_')")
    ap.add_argument("--root", default=ROOT, help="racine à scanner (par défaut: train/)")
    ap.add_argument("--csv",  default=CSV,  help="labels_state.csv à mettre à jour")
    args = ap.parse_args()

    # Charge CSV existant
    df = pd.read_csv(args.csv) if os.path.exists(args.csv) else pd.DataFrame(columns=["relative_path","state"])
    existing = set(df["relative_path"].values)

    new_rows = []
    for r, _, files in os.walk(args.root):
        for f in files:
            if not f.lower().endswith(VALID_EXT):
                continue
            if args.pattern and args.pattern not in f:
                continue  # ne prend que les fichiers correspondant au pattern
            rel = os.path.join(r.replace("app/data/data/", ""), f).replace("\\", "/")
            if rel not in existing:
                new_rows.append({"relative_path": rel, "state": args.state})

    if new_rows:
        df2 = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df2.drop_duplicates(subset=["relative_path"], inplace=True)
        df2.to_csv(args.csv, index=False)
        print(f"✅ Ajouté {len(new_rows)} nouvelles images (state={args.state}) contenant '{args.pattern}' dans {args.csv}")
    else:
        print("Aucune nouvelle image correspondant au motif trouvé.")

if __name__ == "__main__":
    main()

# app/data/augment_balance.py
import os, random, argparse
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import pandas as pd
from collections import Counter

CSV  = "app/data/data/labels_state.csv"
ROOT = "app/data/data"
VALID_STATES = {"good","medium","bad"}

def aug_generic(img, target):
    if random.random()<0.5: img = ImageOps.mirror(img)
    if random.random()<0.5: img = img.rotate(random.uniform(-8,8), resample=Image.BICUBIC, expand=False)
    if target == "bad":
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.80,0.95))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(1.05,1.25))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.6,0.9))
        if random.random()<0.35: img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3,0.9)))
    elif target == "good":
        img = ImageEnhance.Brightness(img).enhance(random.uniform(1.00,1.10))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(1.00,1.15))
        img = ImageEnhance.Color(img).enhance(random.uniform(1.00,1.20))
    else:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9,1.05))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.95,1.10))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.9,1.1))
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", choices=list(VALID_STATES), required=True, help="classe à augmenter")
    ap.add_argument("--target-ratio", type=float, default=1.0, help="ratio objectif vs classe max (1.0 => 1:1:1)")
    ap.add_argument("--max-new-per-image", type=int, default=2, help="cap par image source")
    ap.add_argument("--pattern", default="", help="motif à filtrer sur le NOM de fichier source (ex: new_). Vide = aucun filtre")
    ap.add_argument("--dry-run", action="store_true", help="n’ajoute rien, affiche seulement le plan")
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    train = df[df["relative_path"].str.startswith("train/")].copy()
    cnt = Counter(train["state"])
    if not cnt:
        print("No train data found."); return

    goal = max(cnt.values())
    goal_map = {s: int((args.target_ratio if s==args.state else 1.0)*goal) for s in VALID_STATES}
    need = {s: max(0, goal_map[s]-cnt.get(s,0)) for s in goal_map}
    print(f"Current: {cnt} | Goal: {goal_map} | Need: {need}")

    if need[args.state] <= 0:
        print(f"No need to augment {args.state}."); return

    # sources (optionnellement filtrées par pattern sur le basename)
    src_rows = train[train["state"] == args.state].copy()
    if args.pattern:
        src_rows = src_rows[src_rows["relative_path"].apply(lambda p: args.pattern in os.path.basename(p))]
    if src_rows.empty:
        print(f"No source images for state={args.state} with pattern='{args.pattern}' in train/."); return
    print(f"Using {len(src_rows)} source images (pattern='{args.pattern}' ).")

    new_rows, remaining = [], need[args.state]
    for _, r in src_rows.iterrows():
        if remaining <= 0: break
        src_rel = r["relative_path"]
        src = os.path.join(ROOT, src_rel)
        try:
            img = Image.open(src).convert("RGB")
        except Exception as e:
            print("skip", src_rel, e); continue

        base, _ = os.path.splitext(os.path.basename(src))
        parent = os.path.dirname(src)
        kmax = min(args.max_new_per_image, remaining)
        for _k in range(kmax):
            out = os.path.join(parent, f"{base}_aug{random.randint(1000,9999)}.jpg")
            if not args.dry_run:
                aug_generic(img, args.state).save(out, quality=92)
            rel_new = os.path.join(os.path.dirname(src_rel), os.path.basename(out)).replace("\\","/")
            new_rows.append({"relative_path": rel_new, "state": args.state})
            remaining -= 1
            if remaining <= 0: break

    if args.dry_run:
        print(f"[DRY-RUN] Would add {len(new_rows)} samples of state={args.state}."); return

    if new_rows:
        df2 = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df2.drop_duplicates(subset=["relative_path"], inplace=True)
        df2.to_csv(CSV, index=False)
        print(f"✅ Added {len(new_rows)} {args.state} samples. CSV updated.")
    else:
        print("Nothing added.")

if __name__ == "__main__":
    main()

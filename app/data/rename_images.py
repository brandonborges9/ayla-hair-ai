# scripts/rename_images.py
import os
from glob import glob

ROOT = "train"  # <-- TON dossier actuel
TYPES = ["curly", "straight", "wavy"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for hair_type in TYPES:
    folder = os.path.join(ROOT, hair_type)
    if not os.path.isdir(folder):
        print(f"Skip {folder}, not found.")
        continue

    files = sorted([p for p in glob(os.path.join(folder, "*"))
                    if os.path.splitext(p)[1].lower() in VALID_EXT])
    print(f"\nRenaming {len(files)} images in {folder}...")

    for i, old in enumerate(files,  start=1):
        ext = os.path.splitext(old)[1].lower()
        new = os.path.join(folder, f"{i}{ext}")
        if os.path.abspath(old) == os.path.abspath(new):
            continue
        if os.path.exists(new):
            os.remove(new)
        os.rename(old, new)
        print(os.path.basename(old), "->", os.path.basename(new))

print("\nDone.")

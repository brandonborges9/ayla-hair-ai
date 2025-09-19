# scripts/check_dataset.py
import csv, os, sys
from glob import glob

ROOT = "data"
CSV_PATH = os.path.join(ROOT, "labels_state.csv")
VALID_EXT = (".jpg",".jpeg",".png",".bmp",".webp")

all_imgs = {p.replace("\\","/") for p in glob(os.path.join(ROOT, "*", "*", "*"))
            if p.lower().endswith(VALID_EXT) and os.path.isfile(p)}
state_map = {}
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rel = os.path.join(ROOT, row["relative_path"]).replace("\\","/")
        state_map[rel] = row["state"].strip().lower()

missing_label = [p for p in all_imgs if p not in state_map]
missing_file  = [p for p in state_map.keys() if p not in all_imgs]

print("Total images:", len(all_imgs))
print("Missing state labels:", len(missing_label))
print("Missing files listed in CSV:", len(missing_file))
if missing_label or missing_file:
    sys.exit(1)
print("Dataset OK ✅")

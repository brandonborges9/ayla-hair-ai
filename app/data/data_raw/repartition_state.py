import csv, os
from collections import Counter, defaultdict

CSV = "data/labels_state.csv"
by_split = defaultdict(Counter)
total = Counter()

with open(CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rel = row["relative_path"]
        split = rel.split("/")[0]
        state = row["state"].strip().lower()
        by_split[split][state] += 1
        total[state] += 1

print("TOTAL:", dict(total))
for s in ["train","val","test"]:
    print(s.upper(), dict(by_split[s]))

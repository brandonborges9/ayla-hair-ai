import pandas as pd, json, os

ROOT = "data"
CSV_PATH = os.path.join(ROOT, "labels_state.csv")
OUT_PATH = os.path.join(ROOT, "state_class_weights.json")

labels = pd.read_csv(CSV_PATH)
counts = labels["state"].value_counts()
N_total = len(labels)
K = counts.shape[0]

weights = {c: round(N_total / (K * counts[c]), 4) for c in counts.index}
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

with open(OUT_PATH, "w") as f:
    json.dump(weights, f, indent=2)

print("Class weights saved to", OUT_PATH)
print(weights)
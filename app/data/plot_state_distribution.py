import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

CSV = "app/data/data/labels_state.csv"
OUT = "app/data/state_distribution.png"

def main():
    if not os.path.exists(CSV):
        raise FileNotFoundError(f"{CSV} introuvable")

    df = pd.read_csv(CSV)
    # Garde uniquement train (pas val/test)
    train_df = df[df["relative_path"].str.startswith("train/")]
    counts = Counter(train_df["state"])

    # Tri par ordre logique good/medium/bad si existants
    order = ["good", "medium", "bad"]
    states = [s for s in order if s in counts]
    values = [counts[s] for s in states]

    print("📊 Distribution (train) :", counts)

    # Plot
    plt.figure(figsize=(5, 3))
    plt.bar(states, values, color=["#4CAF50", "#FFC107", "#F44336"])
    plt.title("Distribution des états (train)")
    plt.xlabel("State")
    plt.ylabel("Nombre d'images")
    for i, v in enumerate(values):
        plt.text(i, v + 1, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT, dpi=150)
    plt.close()
    print(f"✅ Graphe sauvegardé : {OUT}")

if __name__ == "__main__":
    main()

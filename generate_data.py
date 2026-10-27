"""Builds data/synthetic_sessions.csv from the rules in rules.py."""
import sys
import numpy as np
import pandas as pd
from config import *
from rules import label_row


def draw_candidates(rng, n):
    mat = rng.choice(MATERIALS, size=n, p=[0.30, 0.10, 0.30, 0.20, 0.10])
    geo = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.15, 0.15, 0.30, 0.25, 0.15])
    bad = (geo == 2) & ~np.isin(mat, ["Aluminum", "Steel"])
    while bad.any():  # throw the row away and draw again
        k = int(bad.sum())
        mat[bad] = rng.choice(MATERIALS, size=k, p=[0.30, 0.10, 0.30, 0.20, 0.10])
        geo[bad] = rng.choice([1, 2, 3, 4, 5], size=k, p=[0.15, 0.15, 0.30, 0.25, 0.15])
        bad = (geo == 2) & ~np.isin(mat, ["Aluminum", "Steel"])
    size = rng.choice([1, 2, 3], size=n, p=[0.4, 0.4, 0.2])
    des = np.where(rng.random(n) < 0.7, 2, 1)
    cols = {"material_type": mat, "part_geometry": geo, "part_size": size, "design_status": des}
    for name, hi in [("strength_level", 4), ("surface_finish", 3), ("tolerance_level", 3),
                     ("batch_size_cat", 5), ("urgency_level", 4), ("budget_per_part", 4)]:
        cols[name] = rng.integers(1, hi + 1, size=n)
    return pd.DataFrame(cols)[FEATURE_COLUMNS]


def main():
    rng = np.random.default_rng(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    df = draw_candidates(rng, N_CANDIDATES)
    df[RULE_TARGET] = [label_row(r) for r in df.to_dict("records")]

    counts = df[RULE_TARGET].value_counts()
    short = {c: int(counts.get(c, 0)) for c in CLASS_LABELS if counts.get(c, 0) < PER_CLASS}
    if short:
        sys.exit(f"STOP: classes with fewer than {PER_CLASS} candidates: {short}. Rules not edited.")

    parts = [g.sample(PER_CLASS, random_state=RANDOM_STATE) for _, g in df.groupby(RULE_TARGET)]
    bal = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    bal[TARGET] = bal[RULE_TARGET]
    idx = rng.choice(len(bal), size=int(round(NOISE_FRAC * len(bal))), replace=False)
    for i in idx:
        others = [c for c in CLASS_LABELS if c != bal.at[i, RULE_TARGET]]
        bal.at[i, TARGET] = others[rng.integers(len(others))]

    DATA_PATH.parent.mkdir(exist_ok=True)
    bal.to_csv(DATA_PATH, index=False)
    print("Candidate counts per class:\n", counts.to_string(), "\n")
    print("Class counts in saved dataset (rule_label):\n", bal[RULE_TARGET].value_counts().to_string())
    print(f"\nTotal rows: {len(bal)}; noisy labels: {(bal[TARGET] != bal[RULE_TARGET]).sum()}")
    print(f"Saved {DATA_PATH}")


if __name__ == "__main__":
    main()

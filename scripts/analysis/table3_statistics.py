"""Recompute the rate columns, Wilson intervals and exact McNemar p-values of Table III
from the per-group outcomes of the confirmatory run."""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


POLICIES = {"MICA": "MICA", "Horizon control": "horizon max-min control",
            "Unfiltered base": "utilitarian (unfiltered base)"}


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=root / "results" / "main" / "table3_per_group_outcomes.csv")
    args = ap.parse_args()
    d = pd.read_csv(args.csv)
    rows = []
    for city, sub in d.groupby("city", sort=False):
        mica = sub[sub.policy == POLICIES["MICA"]].set_index("group_id")
        for label, name in POLICIES.items():
            s = sub[sub.policy == name].set_index("group_id")
            n = len(s)
            k = int(s.irrecoverable.sum())
            lo, hi = wilson(k, n)
            interv = s.n_overrides.sum() / s.n_decisions.sum()
            p = ""
            if label != "MICA":
                a = s.irrecoverable.reindex(mica.index)
                b = mica.irrecoverable
                n01 = int(((a == 1) & (b == 0)).sum())
                n10 = int(((a == 0) & (b == 1)).sum())
                p = (f"{binomtest(n01, n01 + n10, 0.5).pvalue:.4f}" if n01 + n10 else "no discordant pairs")
            rows.append(dict(city=city, policy=label, n=n, irrecoverability=round(k / n, 3),
                             wilson_low=round(lo, 3), wilson_high=round(hi, 3),
                             intervention=round(interv, 3),
                             base_loss=round(s.base_score_loss.mean(), 3), p_vs_MICA=p))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

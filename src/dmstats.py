"""Cleaning, anonymization and small stats helpers for the cold-DM reply analysis.

Build the anonymized dataset from the raw sheet export:
    python -m src.dmstats data/raw/colddms.csv data/processed/dms.csv
"""
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_ztest,
)

REPLY_STATUSES = {"Responded", "Shot and Uploaded", "Delivered", "Booked", "Followed Up", "Cancelled"}
SHOT_STATUSES = {"Shot and Uploaded", "Delivered"}

# Checked in order; first match wins (multi-car owners get their first-listed / most exotic car).
SEGMENT_RULES = [
    ("Italian exotic", r"ferrari|lambo|huracan|revuelto|aventador|gallardo|murcielago|\bsto\b|458|812|720s|testarossa"),
    ("Porsche", r"gt3|gt4|911|carrera|\b9(44|91|92|93|97)\b|718|3rs|922"),
    ("BMW", r"bmw|\b[efg]\d{2}\b|\bm[2-8]\b|3\.0 cs"),
]


def segment(car: str) -> str:
    car = car.lower()
    return next((name for name, pat in SEGMENT_RULES if re.search(pat, car)), "Other")


def anonymize(raw: pd.DataFrame) -> pd.DataFrame:
    """Sent DMs only, no handles/notes/model strings. Note text is used only to derive flags, then dropped."""
    df = raw[raw["Date"].notna() & (raw["Response"] != "Fill In")].copy()
    notes = df["Notes"].fillna("").str.lower()
    date = pd.to_datetime(df["Date"], format="mixed")
    out = pd.DataFrame({
        "owner_id": [f"O{i:03d}" for i in range(1, len(df) + 1)],
        "date_sent": date.dt.date,
        "weekday": date.dt.day_name(),
        "period": np.where(date < "2026-08-01", "May-Jul", "Aug-Oct"),
        "segment": df["Car"].map(segment),
        "multi_car": df["Car"].str.contains(r"/|&|\+", regex=True).astype(int),
        "thread_missing": notes.str.contains("no thread").astype(int),
        # "Dead" rows whose notes show the owner did write back (e.g. "replied once") still count as a reply.
        "replied": (df["Response"].isin(REPLY_STATUSES) | notes.str.contains(r"\breplied\b|\bsaid\b")).astype(int),
        "shot": df["Response"].isin(SHOT_STATUSES).astype(int),
    })
    return out.reset_index(drop=True)


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    return proportion_confint(k, n, alpha=alpha, method="wilson")


def rate_table(df: pd.DataFrame, by: str, y: str = "replied") -> pd.DataFrame:
    """Reply rate per group with 95% Wilson CIs."""
    t = df.groupby(by)[y].agg(n="count", k="sum")
    t["rate"] = t.k / t.n
    t["lo"], t["hi"] = zip(*(wilson(k, n) for k, n in zip(t.k, t.n)))
    return t.sort_values("rate", ascending=False)


def chi2_perm(df: pd.DataFrame, by: str, y: str = "replied", reps: int = 20_000, seed: int = 0) -> dict:
    """Chi-square test of independence plus a permutation p-value (valid when expected counts are small)."""
    table = pd.crosstab(df[by], df[y])
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    # Null distribution: shuffle the outcome; group sizes and the overall rate stay fixed.
    rng = np.random.default_rng(seed)
    yv = df[y].to_numpy()
    onehot = pd.get_dummies(df[by]).to_numpy(float)
    n_g, n, k = onehot.sum(0), len(yv), yv.sum()
    yes = rng.permuted(np.tile(yv, (reps, 1)), axis=1) @ onehot          # reps x groups
    exp_yes, exp_no = n_g * k / n, n_g * (n - k) / n
    null = ((yes - exp_yes) ** 2 / exp_yes + ((n_g - yes) - exp_no) ** 2 / exp_no).sum(1)
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {"chi2": chi2, "dof": dof, "p_asymptotic": p, "p_permutation": (null >= chi2).mean(),
            "cramers_v": cramers_v, "min_expected": expected.min()}


def chi2_cluster_perm(df: pd.DataFrame, by: str, cluster: str, y: str = "replied",
                      reps: int = 20_000, seed: int = 0) -> float:
    """Permutation p-value for a group label that is constant within clusters (e.g. weekday within send date).

    Shuffles the label across clusters, so same-day DMs move together instead of being treated as independent.
    """
    per_cluster = df.groupby(cluster)[by].first()
    assert (df.groupby(cluster)[by].nunique() == 1).all(), f"{by} varies within {cluster}"
    idx = df[cluster].map({c: i for i, c in enumerate(per_cluster.index)}).to_numpy()
    yv = df[y].to_numpy()
    stat = lambda labels: stats.chi2_contingency(pd.crosstab(labels, yv), correction=False)[0]
    observed = stat(df[by].to_numpy())
    rng = np.random.default_rng(seed)
    labels = per_cluster.to_numpy()
    # ponytail: crosstab per rep (~2ms); fine for 17 clusters, vectorize like chi2_perm if reps or n grow
    null = np.array([stat(rng.permutation(labels)[idx]) for _ in range(reps)])
    return (null >= observed - 1e-9).mean()


def cohens_h(p1: float, p2: float) -> float:
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def two_prop(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Two-proportion z-test with risk difference (Newcombe CI) and Cohen's h."""
    z, p = proportions_ztest([k1, k2], [n1, n2])
    lo, hi = confint_proportions_2indep(k1, n1, k2, n2, method="newcomb")
    return {"p1": k1 / n1, "p2": k2 / n2, "diff": k1 / n1 - k2 / n2, "diff_lo": lo, "diff_hi": hi,
            "h": cohens_h(k1 / n1, k2 / n2), "z": z, "p": p,
            "fisher_p": stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]}


def each_vs_rest(df: pd.DataFrame, by: str, y: str = "replied") -> pd.DataFrame:
    """Each group vs. everyone else, Holm-corrected across the family of comparisons."""
    rows = []
    for g in df[by].unique():
        a, b = df[df[by] == g][y], df[df[by] != g][y]
        rows.append({"group": g, **two_prop(a.sum(), len(a), b.sum(), len(b))})
    out = pd.DataFrame(rows).set_index("group")
    out["p_holm"] = multipletests(out.p, method="holm")[1]
    return out.sort_values("diff", ascending=False)


def demo():
    assert segment("BMW G82 M4") == "BMW"
    assert segment("991.2 GT3 RS") == "Porsche"
    assert segment("Huracan STO / M8 Comp / G63") == "Italian exotic"
    assert segment("A80 Supra") == "Other"
    raw = pd.DataFrame({
        "Date": ["5/12/2026", "10/1/26", "07/10/26", None],
        "IG Tag": ["@a", "@b", "@c", "@d"],
        "Car": ["E92 M3", "GT3 / 458", "GTR", "F40"],
        "Response": ["Dead", "Delivered", "Dead", "Fill In"],
        "Notes": ["replied once. no reply since", None, "old lead, no thread", None],
    })
    a = anonymize(raw)
    assert len(a) == 3 and "IG Tag" not in a and "Notes" not in a
    assert a.replied.tolist() == [1, 1, 0] and a.shot.tolist() == [0, 1, 0]
    assert a.thread_missing.tolist() == [0, 0, 1] and a.multi_car.tolist() == [0, 1, 0]
    assert a.weekday[0] == "Tuesday" and a.period.tolist() == ["May-Jul", "Aug-Oct", "May-Jul"]
    lo, hi = wilson(0, 10)
    assert lo == 0 and 0.25 < hi < 0.35
    assert abs(cohens_h(0.5, 0.5)) < 1e-12
    strong = pd.DataFrame({"g": ["a"] * 30 + ["b"] * 30, "replied": [1] * 25 + [0] * 5 + [1] * 5 + [0] * 25})
    null = pd.DataFrame({"g": ["a", "b"] * 30, "replied": [1, 1, 0, 0] * 15})
    assert chi2_perm(strong, "g", reps=2000)["p_permutation"] < 0.001
    assert chi2_perm(null, "g", reps=2000)["p_permutation"] > 0.5
    print("dmstats self-check OK")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        anonymize(pd.read_csv(sys.argv[1])).to_csv(sys.argv[2], index=False)
        print(f"wrote {sys.argv[2]}")
    else:
        demo()

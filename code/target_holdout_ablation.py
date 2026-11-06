"""Constraint ablations on the target-held-out rows.

All arms are scored on the SAME full joint criterion (contact >= 0.6 and the
property/SA/novelty constraints). Arms differ only in which constraints are used
to SELECT the top-1 candidate, so a dropped constraint can cause a miss.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

OUT = os.environ.get("OUT_DIR", "results/target_holdout")
FEATS = ["mcs_atoms", "scaf_sim_ref", "dlogp", "dtpsa", "dhbd", "dhba", "dmw",
         "sa", "conserved", "sa_ok", "k"]
FULL = ["conserved", "sa_ok", "novel"]
SELECT = {
    "full": FULL,
    "drop_novelty": ["conserved", "sa_ok"],
    "drop_sa": ["conserved", "novel"],
    "drop_both": ["conserved"],
    "ranker_only": [],
}
HIT = 0.6


def hit(row):
    if row is None:
        return 0
    return int(row.contact_recall >= HIT and all(row[c] == 1 for c in FULL))


def pick(g, col, masks):
    sub = g
    for c in masks:
        sub = sub[sub[c] == 1]
    if len(sub) == 0:
        return None
    return sub.sort_values(col, ascending=False).iloc[0]


def main():
    df = pd.read_csv(f"{OUT}/rows.csv")
    tr = df[df.split == "train"]
    te = df[df.split == "test"].copy()
    reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                        max_leaf_nodes=31, l2_regularization=1.0,
                                        random_state=0)
    reg.fit(tr[FEATS].fillna(0).values, tr.contact_recall.values)
    te["pred_contact"] = np.clip(reg.predict(te[FEATS].fillna(0).values), 0, 1)
    ybin = (te.contact_recall >= HIT).astype(int)
    res = {"n_test_pairs": int(te.pair.nunique()),
           "spearman": float(spearmanr(te.contact_recall, te.pred_contact).statistic),
           "auc": float(roc_auc_score(ybin, te.pred_contact)) if ybin.nunique() > 1 else None,
           "arms": {}}
    rows = []
    for pid, g in te.groupby("pair"):
        rec = {"pair": pid,
               "property": hit(g.sort_values("prop", ascending=False).iloc[0]),
               "sim2d": hit(g.sort_values("scaf_sim_ref", ascending=False).iloc[0]),
               "random_expected": float((g.contact_recall >= HIT).mul(
                   (g[FULL] == 1).all(axis=1)).mean())}
        for arm, masks in SELECT.items():
            rec[arm] = hit(pick(g, "pred_contact", masks))
        rows.append(rec)
    ad = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    for arm in ["property", "sim2d", "random_expected"] + list(SELECT):
        v = ad[arm].values.astype(float)
        bs = v[rng.integers(0, len(v), size=(4000, len(v)))].mean(1)
        res["arms"][arm] = {"mean": float(v.mean()),
                            "ci_lo": float(np.percentile(bs, 2.5)),
                            "ci_hi": float(np.percentile(bs, 97.5))}
    diffs = (ad["full"].values.astype(float) - ad["property"].values.astype(float))
    bs = diffs[rng.integers(0, len(diffs), size=(4000, len(diffs)))].mean(1)
    res["full_minus_property"] = {
        "mean": float(diffs.mean()),
        "ci_lo": float(np.percentile(bs, 2.5)),
        "ci_hi": float(np.percentile(bs, 97.5)),
        "p_le_0": float((bs <= 0).mean()),
    }
    fam = te.drop_duplicates("pair").set_index("pair")["family"]
    ad["family"] = ad["pair"].map(fam)
    res["by_family"] = {
        f: {a: float(g[a].mean()) for a in ["property", "sim2d", "full"]}
        for f, g in ad.groupby("family")}
    json.dump(res, open(f"{OUT}/ablation.json", "w"), indent=2)
    ad.to_csv(f"{OUT}/ablation_pairs.csv", index=False)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

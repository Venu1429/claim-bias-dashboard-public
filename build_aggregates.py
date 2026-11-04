"""Build aggregates.json from the raw claims file. Run locally; the raw file is never published.
Usage: python build_aggregates.py insurance_claims.csv
"""
import itertools, json, sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

SRC = sys.argv[1] if len(sys.argv) > 1 else "insurance_claims.csv"
OTHER = "Other (small groups)"
FLOOR = 15
CONTROLS = ["EARLY_NON", "MEDICAL_NONMED", "sa_band", "PAYMENT_MODE", "age_band"]
ATTRS = ["PI_GENDER", "age_band", "PI_STATE", "ZONE", "PI_OCCUPATION", "PAYMENT_MODE", "EARLY_NON", "MEDICAL_NONMED", "sa_band", "REASON_FOR_CLAIM"]
TEST_ATTRS = ["PI_GENDER", "age_band", "PI_STATE", "ZONE", "PI_OCCUPATION"]
MIN_NS = list(range(15, 85, 5))

d = pd.read_csv(SRC)
d["rep"] = (d.POLICY_STATUS == "Repudiate Death").astype(int)
d["sa_band"] = pd.qcut(d.SUM_ASSURED.rank(method="first"), 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]).astype(str)
d["age_band"] = pd.cut(d.PI_AGE, [-1, 40, 50, 60, 70, 120], labels=["40 or under", "41-50", "51-60", "61-70", "over 70"]).astype(str)
d["PI_OCCUPATION"] = d.PI_OCCUPATION.fillna("Not recorded")
d["REASON_FOR_CLAIM"] = d.REASON_FOR_CLAIM.fillna("Not recorded")


def design(df, cols):
    if not cols:
        return pd.DataFrame(index=df.index)
    return pd.get_dummies(df[list(cols)].astype(str), drop_first=True).astype(float)


def loglik(X, y):
    if X.shape[1] == 0:
        p = np.full(len(y), y.mean())
    else:
        p = LogisticRegression(C=1e8, max_iter=20000).fit(X, y).predict_proba(X)[:, 1]
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def lump(s, m):
    vc = s.value_counts()
    return s.where(s.isin(vc[vc >= m].index), OTHER)


subsets = [tuple(c) for r in range(len(CONTROLS) + 1) for c in itertools.combinations(sorted(CONTROLS), r)]
key = lambda s: "|".join(s) if s else "none"
out = {"groups": {}, "lr": {}, "gender_or": {}}
for s in subsets:
    X = design(d, s)
    exp = np.full(len(d), d.rep.mean()) if X.shape[1] == 0 else LogisticRegression(C=1.0, max_iter=5000).fit(X, d.rep).predict_proba(X)[:, 1]
    dd = d.assign(exp=exp, vr=exp * (1 - exp))
    for a in ATTRS:
        g = dd.assign(g=lump(dd[a], FLOOR)).groupby("g").agg(n=("rep", "size"), obs=("rep", "sum"), exp=("exp", "sum"), var=("vr", "sum")).reset_index()
        g = g[(g.g != OTHER) | (g.n >= FLOOR)]
        out["groups"][f"{a}::{key(s)}"] = [[r.g, int(r.n), int(r.obs), round(float(r.exp), 4), round(float(r["var"]), 4)] for _, r in g.iterrows()]
    # gender odds ratio
    Xg = design(d, ("PI_GENDER",) + s); Xg.insert(0, "c", 1.0)
    m = LogisticRegression(C=1e8, max_iter=20000, fit_intercept=False).fit(Xg, d.rep)
    p = m.predict_proba(Xg)[:, 1]
    cov = np.linalg.inv((Xg.values * (p * (1 - p))[:, None]).T @ Xg.values)
    b, se = m.coef_[0], np.sqrt(np.diag(cov)); i = list(Xg.columns).index("PI_GENDER_M")
    out["gender_or"][key(s)] = [float(np.exp(b[i])), float(np.exp(b[i] - 1.96 * se[i])), float(np.exp(b[i] + 1.96 * se[i])), float(2 * (1 - stats.norm.cdf(abs(b[i] / se[i]))))]
    # likelihood-ratio tests
    ll0 = loglik(X, d.rep)
    cache = {}
    for a in TEST_ATTRS:
        for m_ in MIN_NS:
            lab = lump(d[a], m_)
            sig = (a, tuple(sorted(lab.unique())))
            if sig not in cache:
                X1 = design(d.assign(_g=lab), list(s) + ["_g"])
                lr = 2 * (loglik(X1, d.rep) - ll0); df = X1.shape[1] - X.shape[1]
                cache[sig] = [lr, df, float(1 - stats.chi2.cdf(lr, df)) if df > 0 else None]
            out["lr"][f"{a}::{key(s)}::{m_}"] = cache[sig]
    print("done", key(s), flush=True)

occ = d[d.PI_OCCUPATION == "Not recorded"]
seg = d.assign(seg=d.EARLY_NON.str.title() + " / " + d.MEDICAL_NONMED.str.title()).groupby(["seg", "PI_GENDER"]).rep.agg(["size", "mean"]).reset_index()
out["meta"] = {
    "N": int(len(d)), "rep_rate": float(d.rep.mean()),
    "amt_share": float(d.loc[d.rep == 1, "SUM_ASSURED"].sum() / d.SUM_ASSURED.sum()),
    "f_rate": float(d[d.PI_GENDER == "F"].rep.mean()), "m_rate": float(d[d.PI_GENDER == "M"].rep.mean()),
    "n_women": int((d.PI_GENDER == "F").sum()),
    "occ_missing_n": int(len(occ)), "occ_missing_rate": float(occ.rep.mean()),
    "zero_income": float((d.PI_ANNUAL_INCOME == 0).mean()), "sa_zero": int((d.SUM_ASSURED == 0).sum()),
    "minors": int((d.PI_AGE < 18).sum()), "reason_missing": int((d.REASON_FOR_CLAIM == "Not recorded").sum()),
    "early_rate": float(d[d.EARLY_NON == "EARLY"].rep.mean()), "nonearly_rate": float(d[d.EARLY_NON == "NON EARLY"].rep.mean()),
}
out["gender_strata"] = [[r.seg, r.PI_GENDER, int(r["size"]), float(r["mean"])] for _, r in seg.iterrows()]
json.dump(out, open("aggregates.json", "w"), separators=(",", ":"))
print("written")

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Insurance Claim Bias Dashboard", layout="wide")

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#c3c2b7"
OTHER = "Other (small groups)"

ATTRS = {
    "Gender": "PI_GENDER", "Age band": "age_band", "State": "PI_STATE", "Sales zone": "ZONE",
    "Occupation": "PI_OCCUPATION", "Payment mode": "PAYMENT_MODE", "Early / non-early claim": "EARLY_NON",
    "Medical / non-medical": "MEDICAL_NONMED", "Sum assured band": "sa_band", "Cause of death": "REASON_FOR_CLAIM",
}
CONTROL_OPTS = {
    "Early / non-early claim": "EARLY_NON", "Medical / non-medical underwriting": "MEDICAL_NONMED",
    "Sum assured band": "sa_band", "Payment mode": "PAYMENT_MODE", "Age band": "age_band",
}


def style(fig, h=360, legend=False):
    fig.update_layout(template="simple_white", height=h, margin=dict(l=10, r=10, t=30, b=10),
                      showlegend=legend, font=dict(size=13), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


@st.cache_data
def load():
    with open("aggregates.json") as f:
        return json.load(f)


A = load()
M = A["meta"]


def ckey(controls):
    return "|".join(sorted(controls)) if controls else "none"


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    r = np.empty(n)
    r[o] = np.minimum(q, 1)
    return r


def fp(p):
    if p is None or np.isnan(p):
        return "n/a"
    return "below 0.0001" if p < 0.0001 else (f"{p:.4f}".rstrip("0").rstrip(".") if p < 0.01 else f"{p:.2f}")


@st.cache_data
def group_table(attr, controls, min_n):
    rows = pd.DataFrame(A["groups"][f"{attr}::{ckey(controls)}"], columns=["g", "n", "obs", "exp", "var"])
    small = (rows.n < min_n) | (rows.g == OTHER)
    big = rows[~small]
    oth = rows[small]
    if len(oth):
        big = pd.concat([big, pd.DataFrame([[OTHER, oth.n.sum(), oth.obs.sum(), oth.exp.sum(), oth["var"].sum()]], columns=rows.columns)])
    t = big.reset_index(drop=True)
    t["obs_rate"] = t.obs / t.n
    t["exp_rate"] = t.exp / t.n
    t["gap_pp"] = (t.obs_rate - t.exp_rate) * 100
    t["z"] = (t.obs - t.exp) / np.sqrt(t["var"])
    t["p"] = 2 * (1 - stats.norm.cdf(np.abs(t.z)))
    t["q"] = bh(t.p)
    lo_hi = [wilson(k, n) for k, n in zip(t.obs, t.n)]
    t["lo"] = [a for a, b in lo_hi]
    t["hi"] = [b for a, b in lo_hi]
    return t.sort_values("gap_pp", ascending=False).reset_index(drop=True)


def overall_test(attr, controls, min_n):
    lr, df, p = A["lr"][f"{attr}::{ckey(controls)}::{min_n}"]
    return lr, df, (float("nan") if p is None else p)


def gender_or(controls):
    return tuple(A["gender_or"][ckey(controls)])


N = M["N"]

# ---------------- sidebar ----------------
st.sidebar.header("Test settings")
ctrl_labels = st.sidebar.multiselect(
    "Factors treated as legitimate reasons for repudiation",
    list(CONTROL_OPTS), default=list(CONTROL_OPTS),
    help="The 'expected' repudiation rate for each group is calculated from these factors. Remove a factor if you do not accept it as a fair reason.")
controls = tuple(CONTROL_OPTS[c] for c in ctrl_labels)
min_n = st.sidebar.slider("Smallest group to test (claims)", 15, 80, 30, 5,
                          help="Smaller groups are combined into 'Other (small groups)' because their rates are too noisy to judge.")
st.sidebar.caption("A group is flagged when its gap from the expected rate stays significant after correcting for testing many groups at once (5% false-discovery rate).")

st.title("Insurance claim bias dashboard")
st.caption(f"{N:,} death claims. Repudiated = claim rejected. A gap from the expected rate is a signal to investigate, not proof of unfair treatment.")

t1, t2, t3, t4, t5 = st.tabs(["1. Overview", "2. Explore rates", "3. Bias test by group", "4. Gender check", "5. Data notes"])

# ---------------- tab 1 ----------------
with t1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Claims", f"{N:,}")
    c2.metric("Repudiated (count)", f"{M['rep_rate']:.1%}")
    c3.metric("Repudiated (sum assured)", f"{M['amt_share']:.1%}")
    c4.metric("Women repudiated", f"{M['f_rate']:.1%}")
    c5.metric("Men repudiated", f"{M['m_rate']:.1%}")

    rows = []
    for lab in ["State", "Sales zone", "Occupation", "Gender", "Age band"]:
        t = group_table(ATTRS[lab], controls, min_n)
        t = t[(t.g != OTHER) & (t.q < 0.05)]
        for _, r in t.iterrows():
            rows.append((f"{lab}: {r.g}", r.gap_pp, int(r.n), r.obs_rate, r.exp_rate))
    flag = pd.DataFrame(rows, columns=["Group", "Gap", "n", "obs", "exp"]).sort_values("Gap")
    st.subheader("Groups repudiated more or less often than their risk factors explain")
    if flag.empty:
        st.success("With the current settings, no group differs from its expected rate after correcting for multiple testing.")
    else:
        fig = go.Figure(go.Bar(
            x=flag.Gap, y=flag.Group, orientation="h",
            marker_color=[ORANGE if g > 0 else BLUE for g in flag.Gap],
            customdata=np.stack([flag.n, flag.obs * 100, flag.exp * 100], axis=1),
            hovertemplate="%{y}<br>Actual %{customdata[1]:.1f}% vs expected %{customdata[2]:.1f}%<br>%{customdata[0]} claims<extra></extra>"))
        fig.update_xaxes(title="Actual minus expected repudiation rate (percentage points)", zeroline=True, zerolinecolor="#888")
        st.plotly_chart(style(fig, h=max(320, 26 * len(flag) + 90)), width="stretch")
        st.caption("Orange = repudiated more often than expected. Blue = less often. Only groups that stay significant after multiple-testing correction are shown.")

    lr_s = overall_test("PI_STATE", controls, min_n)
    lr_z = overall_test("ZONE", controls, min_n)
    lr_o = overall_test("PI_OCCUPATION", controls, min_n)
    orr, olo, ohi, op = gender_or(controls)

    def verdict(p):
        return "clear unexplained differences" if p < 0.01 else ("some unexplained differences" if p < 0.05 else "no clear unexplained differences")

    def ptxt(p):
        return f"p {'' if p < 0.0001 else '= '}{fp(p)}"

    st.subheader("What the tests say")
    gtxt = "no significant difference" if op >= 0.05 else "a statistically significant difference"
    st.markdown(
        f"- **Gender:** {gtxt}. Men's odds of repudiation are {orr:.2f} times women's after adjustment (95% range {olo:.2f} to {ohi:.2f}, {ptxt(op)}).\n"
        f"- **State:** {verdict(lr_s[2])} ({ptxt(lr_s[2])}).\n"
        f"- **Sales zone:** {verdict(lr_z[2])} ({ptxt(lr_z[2])}).\n"
        f"- **Occupation:** {verdict(lr_o[2])} ({ptxt(lr_o[2])}).")

# ---------------- tab 2 ----------------
with t2:
    lab = st.selectbox("Break repudiation rate down by", list(ATTRS), index=2)
    attr = ATTRS[lab]
    t = group_table(attr, tuple(), min_n).sort_values("obs_rate", ascending=False)
    fig = go.Figure(go.Bar(
        x=t.g, y=t.obs_rate, marker_color=BLUE, name="Repudiation rate (95% range)",
        error_y=dict(type="data", symmetric=False, array=(t.hi - t.obs_rate), arrayminus=(t.obs_rate - t.lo), color="#666", thickness=1),
        customdata=np.stack([t.n, t.obs], axis=1),
        hovertemplate="%{x}<br>%{y:.1%} repudiated (%{customdata[1]} of %{customdata[0]} claims)<extra></extra>"))
    fig.add_trace(go.Scatter(x=t.g, y=[M["rep_rate"]] * len(t), mode="lines", line=dict(color=ORANGE, width=2, dash="dash"),
                             name="Overall rate", hoverinfo="skip"))
    fig.update_yaxes(tickformat=".0%", title="Repudiation rate")
    st.plotly_chart(style(fig, h=400, legend=True), width="stretch")
    st.caption("Bars show the raw rate with a 95% range. Short ranges mean more claims behind the number. Raw rates do not allow for risk factors; see the bias test for that.")
    o = t[["g", "n", "obs", "obs_rate"]].rename(columns={"g": lab, "n": "Claims", "obs": "Repudiated", "obs_rate": "Rate"})
    o["Rate"] = o.Rate.map("{:.1%}".format)
    st.dataframe(o, hide_index=True, width="stretch")

# ---------------- tab 3 ----------------
with t3:
    lab = st.selectbox("Test fairness across", ["Gender", "Age band", "State", "Sales zone", "Occupation"], index=2, key="bt")
    attr = ATTRS[lab]
    t = group_table(attr, controls, min_n)
    lr, df_, p = overall_test(attr, controls, min_n)
    c1, c2, c3 = st.columns(3)
    c1.metric("Groups tested", f"{len(t)}")
    c2.metric("Groups flagged", f"{int((t.q < 0.05).sum())}")
    c3.metric("Any real difference between groups?", "Yes" if p < 0.05 else "No clear sign",
              help=f"Likelihood-ratio test after allowing for the chosen factors: {ptxt(p)}")
    ts = t.sort_values("obs_rate", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ts.g, y=ts.obs_rate, name="Actual rate", marker_color=BLUE,
                         customdata=np.stack([ts.n, ts.exp_rate * 100, ts.q], axis=1),
                         hovertemplate="%{x}<br>Actual %{y:.1%}, expected %{customdata[1]:.1f}%<br>%{customdata[0]} claims, adjusted p = %{customdata[2]:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ts.g, y=ts.exp_rate, mode="markers", name="Expected rate",
                             marker=dict(color=ORANGE, size=10, symbol="diamond", line=dict(color="white", width=1)), hoverinfo="skip"))
    fig.update_yaxes(tickformat=".0%", title="Repudiation rate")
    st.plotly_chart(style(fig, h=420, legend=True), width="stretch")
    st.caption("Expected rate = what the chosen factors predict for the claims in each group. A bar well above its diamond means more repudiation than those factors explain.")

    ap = 1 - t[t.g != OTHER].obs_rate
    if len(ap):
        ratio = ap.min() / ap.max()
        lo_g, hi_g = t.loc[ap.idxmin(), "g"], t.loc[ap.idxmax(), "g"]
        st.markdown(f"**Approval-rate ratio (four-fifths rule, raw):** {ratio:.2f}. The lowest approval rate ({ap.min():.1%}, {lo_g}) is {ratio:.0%} of the highest ({ap.max():.1%}, {hi_g}). A ratio under 0.80 is the usual warning level, but it ignores risk factors, so read it with the adjusted result above.")

    o = t[["g", "n", "obs", "obs_rate", "exp_rate", "gap_pp", "q"]].copy()
    o.columns = [lab, "Claims", "Repudiated", "Actual rate", "Expected rate", "Gap (pp)", "Adjusted p"]
    o["Actual rate"] = o["Actual rate"].map("{:.1%}".format)
    o["Expected rate"] = o["Expected rate"].map("{:.1%}".format)
    o["Gap (pp)"] = o["Gap (pp)"].map("{:+.1f}".format)
    o["Adjusted p"] = o["Adjusted p"].map("{:.3f}".format)
    o["Flag"] = np.where(t.q < 0.05, np.where(t.gap_pp > 0, "More than expected", "Less than expected"), "")
    st.dataframe(o, hide_index=True, width="stretch")

# ---------------- tab 4 ----------------
with t4:
    orr, olo, ohi, op = gender_or(controls)
    c1, c2, c3 = st.columns(3)
    c1.metric("Approval rate: women", f"{1 - M['f_rate']:.1%}")
    c2.metric("Approval rate: men", f"{1 - M['m_rate']:.1%}")
    c3.metric("Adjusted odds ratio (men vs women)", f"{orr:.2f}", help=f"95% range {olo:.2f} to {ohi:.2f}; {ptxt(op)}")
    if op < 0.05:
        st.warning(f"After allowing for the chosen factors, men's odds of repudiation are {orr:.2f} times women's (95% range {olo:.2f} to {ohi:.2f}, {ptxt(op)}). This is statistically significant.")
    else:
        st.info(f"The gender gap is not statistically significant after adjustment: odds ratio {orr:.2f}, 95% range {olo:.2f} to {ohi:.2f}, {ptxt(op)}. Raw repudiation is slightly lower for women, so the data gives no sign of bias against them. The range is wide because there are only {M['n_women']} women, so a modest gap cannot be ruled out.")
    s = pd.DataFrame(A["gender_strata"], columns=["seg", "PI_GENDER", "size", "mean"])
    fig = go.Figure()
    for gnd, col, nm in [("F", ORANGE, "Women"), ("M", BLUE, "Men")]:
        x = s[s.PI_GENDER == gnd]
        fig.add_trace(go.Bar(x=x.seg, y=x["mean"], name=nm, marker_color=col, customdata=x["size"],
                             hovertemplate="%{x}<br>" + nm + ": %{y:.1%} repudiated (%{customdata} claims)<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%", title="Repudiation rate")
    st.subheader("Repudiation rate by gender within each claim type")
    st.plotly_chart(style(fig, legend=True), width="stretch")
    st.caption("Some of these groups are small (hover to see the number of claims), so single bars can move a lot.")

# ---------------- tab 5 ----------------
with t5:
    st.subheader("Data quality notes")
    st.markdown(
        f"- **Missing occupation:** {M['occ_missing_n']} claims have no occupation recorded, and only {M['occ_missing_rate']:.0%} of them were repudiated, against {M['rep_rate']:.0%} overall. Missing data like this may reflect how records were captured rather than real claim outcomes.\n"
        f"- **Income:** {M['zero_income']:.0%} of claims show an annual income of zero, so income is not used in any test.\n"
        f"- **Sum assured of zero:** {M['sa_zero']} claims. **Insured under 18:** {M['minors']} claims.\n"
        f"- **Cause of death:** missing for {M['reason_missing']} claims.\n"
        f"- **Early / non-early:** in this file, non-early claims are repudiated more often ({M['nonearly_rate']:.0%}) than early claims ({M['early_rate']:.0%}). That is the opposite of what is usually expected, so please confirm what the column means.")
    st.subheader("How the bias test works")
    st.markdown(
        "1. A logistic regression uses the factors you tick in the sidebar to predict how likely each claim is to be repudiated.\n"
        "2. For each group, the predicted probabilities are added up to give the expected number of repudiations, which is compared with the actual number.\n"
        "3. A group is flagged when the gap stays significant after correcting for testing many groups (Benjamini-Hochberg, 5%).\n"
        "4. The overall test asks whether adding the group variable improves the model beyond those factors.\n\n"
        "**Limits:** the test can only allow for factors in this file. A gap can come from things not recorded here, such as evidence of misrepresentation, and it does not show who made the decision or why. Treat a flag as a reason to review those files.")
    st.subheader("About the data in this app")
    st.markdown("This public app contains only summary tables, with groups of at least 16 claims. No individual claim records are published. Results were calculated in advance from the full claims file for every combination of the factors in the sidebar.")

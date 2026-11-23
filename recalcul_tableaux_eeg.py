"""Statistical comparison of the EEG summaries (Tables 6 and 7 of the article).

For each participant and condition, the script computes the duration-corrected
average maximum peak, the proportion of values above 100, and the session mean and
median of the four EEG indicators, after the IQR filter of the pipeline. It then
compares the gestural and mouse conditions with Wilcoxon signed-rank tests
(asymptotic, ties corrected, one-tailed in the hypothesized direction), with paired
effect sizes d_z and d_rm and their 95% CIs, for the full sample and for each
presentation order. It also compares the gesture-minus-mouse differences between the
two presentation orders with Welch's t-tests.

Usage:
    python recalcul_tableaux_eeg.py --raw data --order <questionnaire .xlsx> --out output

The questionnaire file provides the presentation order of each participant
(column 18, "1ère modalité"); it is not distributed with this repository.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from eeg_gesture_classifier.config import AnalysisConfig
from eeg_gesture_classifier.data_loading import load_all_participants, seed_global_rngs
from eeg_gesture_classifier.features import (
    compute_average_corrected_max_peak,
    compute_proportion_above_threshold,
)
from eeg_gesture_classifier.outliers import apply_iqr_filter

# Indicator order in the data: engagement, memorization, valence, workload.
# Row order and labels used in the article tables.
ROWS = [(3, "Cognitive Load"), (0, "Engagement"), (2, "Attraction"), (1, "Memorization")]
FILE_COLS = ["Enga", "Memo", "Vale", "Work"]


# ----------------------------------------------------------------------------- stats
def wilcoxon_one_tailed(gesture: np.ndarray, mouse: np.ndarray) -> tuple[float, float]:
    """Asymptotic Wilcoxon signed-rank test (ties corrected, no continuity correction),
    one-tailed in the direction gesture > mouse. Returns (|Z|, one-tailed p)."""
    d = np.round(gesture - mouse, 10)
    d = d[d != 0]
    n = len(d)
    ranks = stats.rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    _, counts = np.unique(np.abs(d), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - (counts**3 - counts).sum() / 48
    z = (w_plus - n * (n + 1) / 4) / np.sqrt(var)
    return abs(z), float(stats.norm.sf(z))


def dz_with_ci(diff: np.ndarray) -> tuple[float, float, float]:
    """d_z = mean(differences) / SD(differences), 95% CI from the noncentral t distribution."""
    n = len(diff)
    dz = diff.mean() / diff.std(ddof=1)
    t_obs = dz * np.sqrt(n)

    def bound(q: float) -> float:
        f = lambda ncp: stats.nct.cdf(t_obs, n - 1, ncp) - q
        width = 2.0
        while width <= 64:
            a, b = t_obs - width, t_obs + width
            fa, fb = f(a), f(b)
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                return optimize.brentq(f, a, b) / np.sqrt(n)
            width *= 2
        # Fallback for very large t: normal approximation of the standard error of d_z.
        se = np.sqrt(1 / n + dz**2 / (2 * n))
        return dz + (1.96 if q == 0.025 else -1.96) * se

    return dz, bound(0.975), bound(0.025)


def label(d: float) -> str:
    a = abs(d)
    return "VS" if a < 0.2 else "S" if a < 0.5 else "M" if a < 0.8 else "L"


def compare(gesture: np.ndarray, mouse: np.ndarray) -> dict:
    n = len(gesture)
    tcrit = stats.t.ppf(0.975, n - 1)
    diff = gesture - mouse
    z, p = wilcoxon_one_tailed(gesture, mouse)
    dz, lo, hi = dz_with_ci(diff)
    r = np.corrcoef(gesture, mouse)[0, 1]
    k = np.sqrt(2 * (1 - r))
    out = {"n": n, "Z": z, "p_unilateral": p, "dz": dz, "dz_lo": lo, "dz_hi": hi,
           "drm": dz * k, "drm_lo": lo * k, "drm_hi": hi * k}
    for name, x in (("C", mouse), ("G", gesture)):
        m, s = x.mean(), x.std(ddof=1)
        out.update({f"M_{name}": m, f"SD_{name}": s,
                    f"CI_lo_{name}": m - tcrit * s / np.sqrt(n), f"CI_hi_{name}": m + tcrit * s / np.sqrt(n)})
    return out


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    diff = a.mean() - b.mean()
    se = np.sqrt(va + vb)
    q = stats.t.ppf(0.975, df)
    t = diff / se
    return {"M_CG": a.mean(), "SD_CG": a.std(ddof=1), "M_GC": b.mean(), "SD_GC": b.std(ddof=1),
            "diff": diff, "lo": diff - q * se, "hi": diff + q * se, "t": t, "df": df,
            "p_bilateral": 2 * stats.t.sf(abs(t), df)}


# ----------------------------------------------------------------------------- tables
def table_rows(values: np.ndarray, mouse_first: np.ndarray, metric_name: str) -> list[dict]:
    rows = []
    for col, name in ROWS:
        g, m = values[:, col], values[:, col + 4]
        for group, mask in (("C→G", mouse_first), ("G→C", ~mouse_first), ("All", np.ones(len(g), bool))):
            res = compare(g[mask], m[mask])
            rows.append({"Measure": metric_name, "Indicator": name, "Group": group, **res})
    return rows


def order_rows(values: np.ndarray, mouse_first: np.ndarray, metric_name: str) -> list[dict]:
    rows = []
    for col, name in ROWS:
        d = values[:, col] - values[:, col + 4]
        rows.append({"Measure": metric_name, "Indicator": name, **welch(d[mouse_first], d[~mouse_first])})
    return rows


def fmt(x: float, dec: int) -> str:
    s = f"{x:.{dec}f}"
    if dec == 4:
        s = s.replace("0.", ".", 1)
    return s


def fmt_p(p: float) -> str:
    if p <= 0.001:
        return r"${\leq}.001^{***}$"
    stars = "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""
    txt = f"{p:.3f}"[1:]
    return f"${txt}{stars}$" if stars else f"${txt}$, \\textit{{n.s.}}"


def latex_table(rows: list[dict], dec: int, kind: str) -> str:
    """Rows in the format of Tables 6 and 7 of the article."""
    word = "peak" if kind == "peak" else "proportion"
    out = []
    for col, name in ROWS:
        sub = [r for r in rows if r["Indicator"] == name]
        sig = [r for r in sub if r["Group"] == "All"][0]["p_unilateral"] < 0.05
        interp = f"Exploratory: higher {word}" if sig else "Exploratory: no difference"
        out.append(f"% {name.upper()}")
        for i, r in enumerate(sub):
            grp = "(All)" if r["Group"] == "All" else ("(C$\\rightarrow$G)" if r["Group"] == "C→G" else "(G$\\rightarrow$C)")
            first = f"\\multirow{{6}}{{*}}{{\\textsc{{{name}}}}}" if i == 0 else ""
            last = f"\\multirow{{6}}{{*}}{{{interp}}}" if i == 0 else ""
            dz = f"{r['dz']:.2f} ([{r['dz_lo']:.2f};{r['dz_hi']:.2f}], {label(r['dz'])})"
            drm = f"{r['drm']:.2f} ([{r['drm_lo']:.2f};{r['drm_hi']:.2f}], {label(r['drm'])})"
            out.append(
                f"{first} & Classical {grp} & {fmt(r['M_C'], dec)} & {fmt(r['SD_C'], dec)} & "
                f"[{fmt(r['CI_lo_C'], dec)};{fmt(r['CI_hi_C'], dec)}] & \\multirow{{2}}{{*}}{{{r['Z']:.3f}}} & "
                f"\\multirow{{2}}{{*}}{{{fmt_p(r['p_unilateral'])}}} & {dz} & {last} \\\\"
            )
            out.append(
                f" & Gestural {grp} & {fmt(r['M_G'], dec)} & {fmt(r['SD_G'], dec)} & "
                f"[{fmt(r['CI_lo_G'], dec)};{fmt(r['CI_hi_G'], dec)}] & & & {drm} & \\\\"
            )
            if r["Group"] == "C→G":
                out.append("\\cdashlinelr{2-8}")
            elif r["Group"] == "G→C":
                out.append("\\cmidrule{2-8}")
        out.append("\\midrule")
    out[-1] = "\\bottomrule"
    return "\n".join(out)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data"))
    ap.add_argument("--order", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("output"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = replace(AnalysisConfig(), data_dir=args.raw)
    seed_global_rngs(config.random_seed)
    raw = load_all_participants(config)

    q = pd.read_excel(args.order, header=None)
    ids = q.iloc[2:41, 0].astype(int).tolist()
    assert ids == config.participant_ids, "participant order differs"
    mouse_first = q.iloc[2:41, 17].astype(str).str.strip().str.lower().str.startswith("souris").to_numpy()

    filtered = apply_iqr_filter(raw, config)
    prop = compute_proportion_above_threshold(filtered, config)
    means = np.array([[np.mean(x) for x in p] for p in filtered])
    medians = np.array([[np.median(x) for x in p] for p in filtered])
    seed_global_rngs(config.random_seed)
    peak = compute_average_corrected_max_peak(filtered, config)

    t6 = table_rows(peak, mouse_first, "Peak")
    t7 = table_rows(prop, mouse_first, "Proportion")
    mm = [r for r in table_rows(means, mouse_first, "Mean") + table_rows(medians, mouse_first, "Median")
          if r["Group"] == "All"]
    order = order_rows(peak, mouse_first, "Peak") + order_rows(prop, mouse_first, "Proportion")

    cols = [f"{n}_{c}" for c in ("gesture", "mouse") for n in FILE_COLS]
    pid = config.participant_ids
    with pd.ExcelWriter(args.out / "eeg_comparisons.xlsx") as w:
        pd.DataFrame(peak, index=pid, columns=cols).rename_axis("Participant").to_excel(w, sheet_name="Peaks")
        pd.DataFrame(prop, index=pid, columns=cols).rename_axis("Participant").to_excel(w, sheet_name="Proportions")
        pd.DataFrame(means, index=pid, columns=cols).rename_axis("Participant").to_excel(w, sheet_name="Means")
        pd.DataFrame(medians, index=pid, columns=cols).rename_axis("Participant").to_excel(w, sheet_name="Medians")
        pd.DataFrame(t6).to_excel(w, sheet_name="Table 6", index=False)
        pd.DataFrame(t7).to_excel(w, sheet_name="Table 7", index=False)
        pd.DataFrame(mm).to_excel(w, sheet_name="Means and medians", index=False)
        pd.DataFrame(order).to_excel(w, sheet_name="Order effect", index=False)

    (args.out / "tables_6_7.tex").write_text(
        "% Table 6 (peaks)\n" + latex_table(t6, 2, "peak")
        + "\n\n% Table 7 (proportions)\n" + latex_table(t7, 4, "proportion") + "\n",
        encoding="utf-8",
    )
    print("Done:", args.out)


if __name__ == "__main__":
    main()

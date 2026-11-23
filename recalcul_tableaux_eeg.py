"""Recalcul des tableaux 6 et 7 (EEG) après correction du traitement des valeurs extrêmes.

Correction du 23-11-2026. Ce script :
1. charge les exports BitBrain bruts (39 participants, phase catalogue uniquement) ;
2. applique le filtre IQR de la méthode publiée (remplacement par la borne dans les
   DEUX conditions, voir eeg_gesture_classifier/outliers.py) ;
3. calcule, par participant et par condition, le pic maximal moyen (correction de durée,
   5 000 simulations), la proportion de valeurs > 100, la moyenne et la médiane ;
4. reproduit les tests des tableaux 6 et 7 (Wilcoxon signé, asymptotique, correction des
   ex aequo, unilatéral dans le sens des hypothèses, d_z et d_rm avec IC à 95 %), pour
   l'échantillon complet et pour chaque ordre de présentation ;
5. compare l'effet d'ordre (test de Welch sur les écarts gestes moins souris) ;
6. vérifie que le même calcul appliqué aux fichiers publiés redonne les valeurs publiées ;
7. écrit un classeur Excel et les tableaux LaTeX corrigés.

Usage (depuis le dossier Code) :
    python recalcul_tableaux_eeg.py
Les chemins par défaut supposent l'arborescence du dossier « Soumission dernier article
de thèse » ; ils peuvent être changés avec --raw, --donnees et --sortie.
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
from eeg_gesture_classifier.outliers import apply_iqr_filter, apply_original_asymmetric_filter

# Ordre des indicateurs dans les données : engagement, memorization, valence, workload.
# Ordre et libellés des lignes dans les tableaux de l'article.
ROWS = [(3, "Cognitive Load"), (0, "Engagement"), (2, "Attraction"), (1, "Memorization")]
FILE_COLS = ["Enga", "Memo", "Vale", "Work"]


# ----------------------------------------------------------------------------- stats
def wilcoxon_one_tailed(gesture: np.ndarray, mouse: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signé asymptotique (ex aequo corrigés, sans correction de continuité),
    unilatéral dans le sens gestes > souris. Retourne (|Z|, p unilatéral)."""
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
    """d_z = moyenne(écarts) / ET(écarts), IC à 95 % par la loi de t non centrée."""
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
        # Repli (très grands t) : approximation normale de l'erreur-type de d_z.
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
            rows.append({"Mesure": metric_name, "Indicateur": name, "Groupe": group, **res})
    return rows


def order_rows(values: np.ndarray, mouse_first: np.ndarray, metric_name: str) -> list[dict]:
    rows = []
    for col, name in ROWS:
        d = values[:, col] - values[:, col + 4]
        rows.append({"Mesure": metric_name, "Indicateur": name, **welch(d[mouse_first], d[~mouse_first])})
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
    """Tableau au format des tableaux 6 et 7 de l'article."""
    word = "peak" if kind == "peak" else "proportion"
    out = []
    for col, name in ROWS:
        sub = [r for r in rows if r["Indicateur"] == name]
        sig = [r for r in sub if r["Groupe"] == "All"][0]["p_unilateral"] < 0.05
        interp = f"Exploratory: higher {word}" if sig else "Exploratory: no difference"
        out.append(f"% {name.upper()}")
        for i, r in enumerate(sub):
            grp = "(All)" if r["Groupe"] == "All" else ("(C$\\rightarrow$G)" if r["Groupe"] == "C→G" else "(G$\\rightarrow$C)")
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
            if r["Groupe"] == "C→G":
                out.append("\\cdashlinelr{2-8}")
            elif r["Groupe"] == "G→C":
                out.append("\\cmidrule{2-8}")
        out.append("\\midrule")
    out[-1] = "\\bottomrule"
    return "\n".join(out)


# ----------------------------------------------------------------------------- main
def main() -> None:
    here = Path(__file__).resolve().parent
    base = here.parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=base / "Données" / "Exported_rawData EEG (backup)")
    ap.add_argument("--donnees", type=Path, default=base / "Données")
    ap.add_argument("--sortie", type=Path, default=here.parent / "Résultats")
    args = ap.parse_args()
    args.sortie.mkdir(parents=True, exist_ok=True)

    config = replace(AnalysisConfig(), data_dir=args.raw)
    seed_global_rngs(config.random_seed)
    raw = load_all_participants(config)

    # Ordre de présentation, tiré du fichier des questionnaires (colonne « 1ère modalité »).
    q = pd.read_excel(args.donnees / "Données questionnaires.xlsx", header=None)
    ids = q.iloc[2:41, 0].astype(int).tolist()
    assert ids == config.participant_ids, "ordre des participants différent"
    mouse_first = q.iloc[2:41, 17].astype(str).str.strip().str.lower().str.startswith("souris").to_numpy()
    assert mouse_first.sum() == 21

    # Vérification : le filtre d'origine redonne les proportions publiées.
    published_prop = pd.read_excel(args.donnees / "Proportion of high brain activity.xlsx").iloc[:, 1:].to_numpy()
    published_peak = pd.read_excel(args.donnees / "Highest bain activity.xlsx").iloc[:, 1:].to_numpy()
    original_prop = compute_proportion_above_threshold(apply_original_asymmetric_filter(raw, config), config)
    exact_share = float((np.abs(original_prop - published_prop) < 1e-9).mean())

    # Calcul corrigé.
    filtered = apply_iqr_filter(raw, config)
    prop = compute_proportion_above_threshold(filtered, config)
    means = np.array([[np.mean(x) for x in p] for p in filtered])
    medians = np.array([[np.median(x) for x in p] for p in filtered])
    seed_global_rngs(config.random_seed)
    peak = compute_average_corrected_max_peak(filtered, config)

    # Tests.
    t6 = table_rows(peak, mouse_first, "Pic")
    t7 = table_rows(prop, mouse_first, "Proportion")
    t6_pub = table_rows(published_peak, mouse_first, "Pic (publié)")
    t7_pub = table_rows(published_prop, mouse_first, "Proportion (publiée)")
    mm = [r for r in table_rows(means, mouse_first, "Moyenne") + table_rows(medians, mouse_first, "Médiane")
          if r["Groupe"] == "All"]
    order = order_rows(peak, mouse_first, "Pic") + order_rows(prop, mouse_first, "Proportion")

    # Excel.
    cols = [f"{n}_{c}" for c in ("gestes", "souris") for n in FILE_COLS]
    pid = config.participant_ids
    xlsx = args.sortie / "Tableau - Recalcul EEG corrigé - 23-11-26.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame({
            "Élément": [
                "Traitement des valeurs extrêmes",
                "Simulations de correction de durée (pics)",
                "Graine aléatoire",
                "Seuil d'activité élevée",
                "Participants (souris d'abord / gestes d'abord)",
                "Proportions publiées reproduites exactement par le filtre d'origine",
            ],
            "Valeur": [
                "IQR × 3, bornes par participant sur les deux sessions, remplacement par la borne dans les deux conditions",
                config.simulations_peak, config.random_seed, config.high_activity_threshold,
                f"39 ({int(mouse_first.sum())} / {int((~mouse_first).sum())})",
                f"{exact_share:.1%}",
            ],
        }).to_excel(w, sheet_name="Paramètres", index=False)
        pd.DataFrame(peak, index=pid, columns=cols).rename_axis("Participant").to_excel(w, sheet_name="Pics corrigés")
        pd.DataFrame(prop, index=pid, columns=cols).rename_axis("Participant").to_excel(w, sheet_name="Proportions corrigées")
        pd.DataFrame(means, index=pid, columns=cols).rename_axis("Participant").to_excel(w, sheet_name="Moyennes corrigées")
        pd.DataFrame(medians, index=pid, columns=cols).rename_axis("Participant").to_excel(w, sheet_name="Médianes corrigées")
        pd.DataFrame(t6).to_excel(w, sheet_name="Tableau 6 corrigé", index=False)
        pd.DataFrame(t7).to_excel(w, sheet_name="Tableau 7 corrigé", index=False)
        comp = []
        for pub, cor in ((t6_pub, t6), (t7_pub, t7)):
            for a, b in zip(pub, cor):
                if a["Groupe"] == "All":
                    comp.append({"Mesure": b["Mesure"], "Indicateur": b["Indicateur"],
                                 "Z publié": a["Z"], "p publié": a["p_unilateral"], "d_z publié": a["dz"],
                                 "Z corrigé": b["Z"], "p corrigé": b["p_unilateral"], "d_z corrigé": b["dz"]})
        pd.DataFrame(comp).to_excel(w, sheet_name="Avant-après", index=False)
        pd.DataFrame(mm).to_excel(w, sheet_name="Moyennes et médianes", index=False)
        pd.DataFrame(order).to_excel(w, sheet_name="Effet d'ordre EEG", index=False)

    (args.sortie / "Tableaux 6 et 7 corrigés (lignes LaTeX).tex").write_text(
        "% Tableau 6 (pics), lignes à coller entre \\midrule et \\bottomrule\n"
        + latex_table(t6, 2, "peak")
        + "\n\n% Tableau 7 (proportions), lignes à coller entre \\midrule et \\bottomrule\n"
        + latex_table(t7, 4, "proportion") + "\n",
        encoding="utf-8",
    )
    print("Terminé :", xlsx)


if __name__ == "__main__":
    main()

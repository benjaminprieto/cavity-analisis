# 08_tests/07_integration_strict_test.py
"""
Test estricto del Módulo 07 (Integration)

Qué hace:
- Valida artefactos de módulos 01–05 (existencia + esquema mínimo).
- Verifica columnas y rangos de scores en critical_residues_final.csv.
- Comprueba cálculo de critical_score = sum_i w_i * score_i (tolerancia).
- Chequeos de consistencia: segid↔protein, duplicados, NaNs, coberturas.
- Sensibilidad: barrido de umbral; ablation (quita cada score y reordena).
- Correlaciones entre scores; solape Top-K entre módulos.
- Enriquecimiento de cavidad (si lista presente).
- Plots y reporte.

Uso:
  python 08_tests/07_integration_strict_test.py --config 03_configs/analyses/07_integration.yaml [--strict]
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

# --------------------- utilidades ---------------------

REQ_NET_COLS  = {"resid","resname","segid","protein","betweenness","eigenvector","closeness","degree"}
REQ_EN_COLS   = {"segid","resid","resname","binding_energy"}
REQ_DCCM_COLS = {"resid1","resid2","correlation"}
REQ_PCA_COLS  = {"resid","resname","segid"}  # + PCi
REQ_PATH_COLS = {"resid","frequency"}

def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def read_csv(p: Path, name: str) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"[FALTA] {name}: {p}")
    return pd.read_csv(p)

def maybe_read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def assert_range_01(series: pd.Series, name: str, strict: bool, issues: list):
    if series.isna().any():
        issues.append(f"{name} contiene NaN")
        if strict: raise AssertionError(f"{name} contiene NaN")
    mn, mx = series.min(), series.max()
    if (mn < -1e-6) or (mx > 1 + 1e-6):
        issues.append(f"{name} fuera de [0,1] (min={mn:.3f}, max={mx:.3f})")
        if strict: raise AssertionError(f"{name} fuera de [0,1]")

def safe_corr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return df[cols].corr()

def fisher_exact(a,b,c,d):
    # Tabla:
    # [a b]
    # [c d]
    # OR = (a*d) / (b*c)
    # p-value exacta: usaremos aproximación simple si no hay scipy.
    if b == 0 or c == 0:
        or_val = float('inf')
    else:
        or_val = (a*d) / (b*c)
    # p-value: no exacta sin scipy; reportamos OR y una pseudo-pvalue por chi2
    # Chi-square con corrección de Yates
    n = a+b+c+d
    num = n*(abs(a*d - b*c) - n/2)**2
    den = (a+b)*(c+d)*(a+c)*(b+d) if (a+b)*(c+d)*(a+c)*(b+d) != 0 else np.nan
    chi2 = num/den if den and den>0 else np.nan
    # aprox p-value 1 - CDF_chi2(df=1)
    p = float(np.exp(-0.5*chi2)) if (not math.isnan(chi2)) else float('nan')
    return or_val, p

# --------------------- plots ---------------------

def plot_hist(arr, title, outpng, xlabel="valor", bins=40):
    if arr is None or len(arr)==0: return
    fig = plt.figure(figsize=(6,4))
    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("conteo")
    fig.tight_layout()
    fig.savefig(outpng, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_bar(labels, values, title, outpng, rotate=True):
    if not values: return
    fig = plt.figure(figsize=(8,4))
    x = np.arange(len(values))
    plt.bar(x, values)
    if rotate:
        plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    else:
        plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel("valor")
    fig.tight_layout()
    fig.savefig(outpng, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_corr_matrix(corr_df: pd.DataFrame, title: str, outpng: Path):
    if corr_df is None or corr_df.empty: return
    fig = plt.figure(figsize=(5,4))
    im = plt.imshow(corr_df.values, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(corr_df.index)), corr_df.index, fontsize=8)
    plt.title(title)
    fig.tight_layout()
    fig.savefig(outpng, dpi=150, bbox_inches="tight")
    plt.close(fig)

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Test estricto Módulo 07 (Integration)")
    ap.add_argument("--config", required=True, help="03_configs/analyses/07_integration.yaml")
    ap.add_argument("--strict", action="store_true", help="Fallar ante cualquier inconsistencia")
    args = ap.parse_args()
    strict = args.strict

    PROJ = Path(__file__).resolve().parents[1]
    cfg_path = (PROJ / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_yaml(cfg_path)

    outdir = PROJ / cfg["output"]["base_dir"]
    ensure_dir(outdir)
    plots = outdir / "plots"
    ensure_dir(plots)

    weights = dict(cfg["integration"]["weights"])
    min_crit = float(cfg["integration"]["min_critical_score"])
    cav_cfg = cfg["integration"]["cavity_check"]
    cavity_list = list(cav_cfg.get("cavity_residues", [])) if cav_cfg.get("enabled", False) else []

    # --- entradas esperadas de módulos previos ---
    art = cfg["artifacts"]
    p_net  = PROJ / art["network_ranking"]
    p_eng  = PROJ / art["energy_binding_per_res"]
    p_dccm = PROJ / art["dccm_high_correlations"]
    p_pca  = PROJ / art["pca_participation"]
    p_path = PROJ / art["pathways_bottlenecks"]

    issues = []

    # Validar existencia y esquema mínimo (si existen)
    try:
        net = read_csv(p_net, "centrality_ranking.csv")
        if not REQ_NET_COLS.issubset(set(net.columns)):
            issues.append(f"network_ranking: columnas faltantes {REQ_NET_COLS - set(net.columns)}")
            if strict: raise AssertionError("network_ranking.csv esquema inválido")
    except FileNotFoundError as e:
        issues.append(str(e))
        if strict: raise

    eng  = maybe_read_csv(p_eng)
    if not eng.empty and not REQ_EN_COLS.issubset(set(eng.columns)):
        issues.append(f"binding_per_residue.csv: columnas faltantes {REQ_EN_COLS - set(eng.columns)}")
        if strict: raise AssertionError("binding_per_residue.csv esquema inválido")

    dccm = maybe_read_csv(p_dccm)
    if not dccm.empty and not REQ_DCCM_COLS.issubset(set(dccm.columns)):
        issues.append(f"high_correlations.csv: columnas faltantes {REQ_DCCM_COLS - set(dccm.columns)}")
        if strict: raise AssertionError("high_correlations.csv esquema inválido")

    pca  = maybe_read_csv(p_pca)
    if not pca.empty and not REQ_PCA_COLS.issubset(set(pca.columns)):
        issues.append(f"residue_participation.csv: columnas mínimas faltantes {REQ_PCA_COLS - set(pca.columns)}")
        if strict: raise AssertionError("residue_participation.csv esquema inválido")

    pathw = maybe_read_csv(p_path)
    if not pathw.empty and not REQ_PATH_COLS.issubset(set(pathw.columns)):
        issues.append(f"bottleneck_residues.csv: columnas faltantes {REQ_PATH_COLS - set(pathw.columns)}")
        if strict: raise AssertionError("bottleneck_residues.csv esquema inválido")

    # --- salidas del módulo 07 ---
    out_cfg = cfg["integration"]["output"]
    p_crit = outdir / out_cfg["critical_residues"]
    p_sum  = outdir / out_cfg["summary"]
    p_cav  = outdir / out_cfg["cavity_importance"]

    if not p_crit.exists():
        raise FileNotFoundError(f"No encuentro critical_residues_final.csv: {p_crit}")
    df = pd.read_csv(p_crit)

    # columnas esperadas
    exp_cols = {"resid","resname","segid","protein","critical_score",
                "network_score","energy_score","dccm_score","pca_score","pathway_score"}
    missing = exp_cols - set(df.columns)
    if missing:
        issues.append(f"critical_residues_final.csv: faltan columnas {missing}")
        if strict: raise AssertionError("critical_residues_final.csv columnas faltantes")

    # tipos y NaNs
    for c in ["critical_score","network_score","energy_score","dccm_score","pca_score","pathway_score"]:
        assert_range_01(df[c].astype(float), c, strict, issues)

    # critical_score ≈ suma ponderada
    wsum = sum(weights.values())
    if abs(wsum - 1.0) > 1e-6:
        weights = {k: v/wsum for k,v in weights.items()}
    cs_recalc = (
        df["network_score"]*weights.get("network",0) +
        df["energy_score"] *weights.get("energy",0)  +
        df["dccm_score"]   *weights.get("dccm",0)   +
        df["pca_score"]    *weights.get("pca",0)    +
        df["pathway_score"]*weights.get("pathway",0)
    )
    diff = (df["critical_score"] - cs_recalc).abs()
    max_diff = float(diff.max())
    if max_diff > 1e-6:
        issues.append(f"critical_score no coincide con suma ponderada (max diff={max_diff:.3e})")
        if strict: raise AssertionError("critical_score mal calculado")

    # consistencia segid↔protein
    if "protein" in df.columns:
        map_ok = (
            ((df["segid"]=="A") & (df["protein"]=="NRP1")) |
            ((df["segid"]=="B") & (df["protein"]=="XYLT1")) |
            (df["segid"].isin(["A","B"]) == False)  # por si aparecen otros (ej U/ligando) que no tienen protein
        )
        if (~map_ok).any():
            issues.append("Inconsistencia segid↔protein (A≠NRP1 o B≠XYLT1 en alguna fila)")

    # duplicados resid
    dup = df["resid"].duplicated().sum()
    if dup > 0:
        issues.append(f"Hay {dup} resid duplicados en integración")

    # cobertura vs network ranking
    net_resids = set(net["resid"].astype(int)) if "resid" in net.columns else set()
    int_resids = set(df["resid"].astype(int))
    if net_resids and len(int_resids) < 0.9*len(net_resids):
        issues.append("Cobertura de residuos en integración <90% respecto a network ranking")

    # -------- Sensibilidad: barrido de umbral --------
    thresholds = np.linspace(0.3, 0.9, 13)
    counts = []
    for t in thresholds:
        counts.append(int((df["critical_score"] >= t).sum()))
    plot_bar([f"{t:.2f}" for t in thresholds], counts,
             "Sensibilidad: #críticos vs umbral", plots/"sensitivity_threshold_counts.png")

    # -------- Ablation: quitar uno por vez --------
    # Recalcula ranking sin cada score (dejándolo 0), y mide Spearman-like overlap top-k
    k = 50
    base_top = set(df.sort_values("critical_score", ascending=False).head(k)["resid"].astype(int))
    ablation_results = []
    for key in ["network","energy","dccm","pca","pathway"]:
        df_ab = df.copy()
        col = f"{key}_score"
        if col not in df_ab.columns:
            continue
        df_ab[col] = 0.0
        cs = (
            df_ab["network_score"]*weights.get("network",0) +
            df_ab["energy_score"] *weights.get("energy",0)  +
            df_ab["dccm_score"]   *weights.get("dccm",0)   +
            df_ab["pca_score"]    *weights.get("pca",0)    +
            df_ab["pathway_score"]*weights.get("pathway",0)
        )
        top_ab = set(df_ab.assign(cs=cs).sort_values("cs", ascending=False).head(k)["resid"].astype(int))
        overlap = len(base_top & top_ab) / max(1,len(base_top))
        ablation_results.append((key, overlap))
    ablation_results.sort(key=lambda x: x[1])
    if ablation_results:
        plot_bar([k for k,_ in ablation_results], [v for _,v in ablation_results],
                 f"Ablation Top-{k} overlap con ranking base", plots/"ablation_topk_overlap.png")

    # -------- Correlaciones entre scores --------
    corr = safe_corr(df, ["network_score","energy_score","dccm_score","pca_score","pathway_score","critical_score"])
    plot_corr_matrix(corr, "Matriz de correlación entre scores", plots/"scores_corr_matrix.png")

    # -------- Solape Top-K entre módulos (si tenemos sus “picks”) --------
    topk = 50
    top_sets = {}
    # network: por betweenness (si existe)
    if "betweenness" in net.columns:
        top_sets["network"] = set(net.sort_values("betweenness", ascending=False).head(topk)["resid"].astype(int))
    # energy: más favorables (binding_energy más negativo)
    if not eng.empty and "binding_energy" in eng.columns:
        top_sets["energy"] = set(eng.sort_values("binding_energy").head(topk)["resid"].astype(int))
    # dccm: nodos con mayor frecuencia en high_corr (aproximado)
    if not dccm.empty:
        cnt = Counter()
        for _, r in dccm.iterrows():
            cnt[int(r["resid1"])] += abs(float(r["correlation"]))
            cnt[int(r["resid2"])] += abs(float(r["correlation"]))
        dccm_top = sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:topk]
        top_sets["dccm"] = set([res for res,_ in dccm_top])
    # pca: PCs altos -> sumar PCs si existen
    if not pca.empty:
        pc_cols = [c for c in pca.columns if c.startswith("PC")]
        if pc_cols:
            tmp = pca.copy()
            tmp["pca_participation"] = tmp[pc_cols].mean(axis=1)
            top_sets["pca"] = set(tmp.sort_values("pca_participation", ascending=False).head(topk)["resid"].astype(int))
    # pathways: por frecuencia
    if not pathw.empty and "frequency" in pathw.columns:
        top_sets["pathway"] = set(pathw.sort_values("frequency", ascending=False).head(topk)["resid"].astype(int))

    # Overlaps pairwise
    overlap_rows = []
    keys = list(top_sets.keys())
    for a,b in combinations(keys,2):
        inter = len(top_sets[a] & top_sets[b])
        overlap_rows.append({"A":a, "B":b, "overlap": inter, "jaccard": inter / max(1,len(top_sets[a] | top_sets[b]))})
    if overlap_rows:
        ov_df = pd.DataFrame(overlap_rows).sort_values("overlap", ascending=False)
        ov_df.to_csv(outdir/"topk_pairwise_overlap.csv", index=False)

    # -------- Cavidad: enriquecimiento (si hay lista) --------
    cav_report = "No cavity list provided."
    if cavity_list:
        crit_mask = df["critical_score"] >= min_crit
        is_cavity = df["resid"].astype(int).isin(set(cavity_list))
        a = int(((crit_mask) & (is_cavity)).sum())                # críticos ∩ cavidad
        b = int(((crit_mask) & (~is_cavity)).sum())               # críticos fuera
        c = int((~crit_mask & is_cavity).sum())                   # no críticos ∩ cavidad
        d = int((~crit_mask & ~is_cavity).sum())                  # no críticos fuera
        OR, p = fisher_exact(a,b,c,d)
        cav_report = f"cavity OR={OR:.2f}, approx p={p:.3g}  (a={a}, b={b}, c={c}, d={d})"
        # guardar histogramas
        plot_hist(df.loc[is_cavity,"critical_score"].values, "Distribución score (cavidad)", plots/"cavity_score_dist.png")
        plot_hist(df.loc[~is_cavity,"critical_score"].values, "Distribución score (no cavidad)", plots/"non_cavity_score_dist.png")

    # -------- Guardar reporte --------
    report_lines = []
    report_lines.append("TEST ESTRICTO — PASO 07 (INTEGRATION)")
    report_lines.append("="*72)
    report_lines.append(f"YAML: {cfg_path}")
    report_lines.append(f"OUTPUT DIR: {outdir}")
    report_lines.append("")

    if issues:
        report_lines.append("HALLAZGOS / ISSUES:")
        for m in issues:
            report_lines.append(f"  - {m}")
        report_lines.append("")
    else:
        report_lines.append("OK: estructura y contenidos coherentes.\n")

    report_lines.append(f"Pesos (normalizados): {json.dumps(weights)}")
    report_lines.append(f"Max |critical_score - suma_pesos| = {max_diff:.3e}")
    report_lines.append("")
    report_lines.append(f"Sensibilidad (umbral -> #críticos): {dict(zip([f'{t:.2f}' for t in thresholds], counts))}")
    report_lines.append("")
    if ablation_results:
        report_lines.append("Ablation Top-50 overlap (menor => score más influyente):")
        for k,v in ablation_results:
            report_lines.append(f"  {k:8s}  overlap={v:.3f}")
        report_lines.append("")
    if not corr.empty:
        report_lines.append("Correlación entre scores:")
        report_lines.append(corr.round(3).to_string())
        report_lines.append("")

    if overlap_rows:
        report_lines.append("Top-K overlaps por pares (ver CSV topk_pairwise_overlap.csv):")
        report_lines.append(pd.DataFrame(overlap_rows).sort_values("overlap", ascending=False).head(10).to_string(index=False))
        report_lines.append("")

    report_lines.append("Cavity enrichment:")
    report_lines.append(f"  {cav_report}")
    report_lines.append("")

    # Dónde quedaron los plots
    pl = sorted([p.name for p in plots.glob("*.png")])
    if pl:
        report_lines.append("PLOTS generados:")
        for n in pl:
            report_lines.append(f"  - {n}")

    rep = outdir/"strict_test_report.txt"
    rep.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\n✅ Reporte escrito en: {rep}")

    # Si modo estricto + hubo issues, sal con error
    if strict and issues:
        raise SystemExit(2)

if __name__ == "__main__":
    main()

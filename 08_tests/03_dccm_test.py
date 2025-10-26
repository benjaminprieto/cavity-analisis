# 08_tests/03_dccm_test.py
"""
Test integral del PASO 03 (DCCM):
- Lee YAML (--config)
- Verifica y carga:
    * dccm_matrix.npy
    * dccm_cross_AB.csv
    * dccm_high_pairs.csv
- Sanity checks (tamaños, NaNs, rangos [-1,1])
- Estadísticas de correlaciones (globales, A<->B, top |r|)
- Plots (matplotlib puro) en .../03_dccm/plots/
- Reporte en .../03_dccm/test_report.txt

Uso:
    python 08_tests/03_dccm_test.py --config 03_configs/analyses/03_dccm.yaml
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


# ---------------------------- utilidades base --------------------------------

def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_read_csv(p: Path, name: str) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"No encuentro {name}: {p}")
    return pd.read_csv(p)

def plot_hist(values: np.ndarray, title: str, outpath: Path, bins: int = 50):
    if values.size == 0:
        return
    fig = plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel("correlation r")
    plt.ylabel("conteo")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_heatmap_square(mat: np.ndarray, title: str, outpath: Path, max_dim: int = 200):
    """Heatmap compacto para una submatriz NxN (recorta a max_dim para no explotar)."""
    if mat.size == 0:
        return
    n = mat.shape[0]
    sub = mat[:min(n, max_dim), :min(n, max_dim)]
    fig = plt.figure(figsize=(6, 5))
    im = plt.imshow(sub, vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="r")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_top_pairs(df: pd.DataFrame, title: str, outpath: Path, n: int = 20):
    """Barras con los top-N por |r| del DataFrame (debe tener columnas resid/segid y 'correlation')."""
    if df.empty:
        return
    dft = df.copy()
    dft["abs_r"] = dft["correlation"].abs()
    dft = dft.sort_values("abs_r", ascending=False).head(n)
    # etiquetas cortas
    if {"segid1","resid1","resname1","segid2","resid2","resname2"}.issubset(dft.columns):
        labels = [f"{r.segid1}{int(r.resid1)}-{r.resname1}__{r.segid2}{int(r.resid2)}-{r.resname2}" for _, r in dft.iterrows()]
    else:
        labels = [str(i) for i in range(len(dft))]
    fig = plt.figure(figsize=(10, 4))
    plt.bar(range(len(dft)), dft["correlation"].values)
    plt.xticks(range(len(dft)), labels, rotation=45, ha="right", fontsize=8)
    plt.title(title)
    plt.xlabel("par de residuos")
    plt.ylabel("r")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------- main -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Full test del PASO 03 (DCCM) con plots y reporte.")
    ap.add_argument("--config", required=True, help="Ruta al YAML (03_configs/analyses/03_dccm.yaml)")
    args = ap.parse_args()

    # localizar raíz del repo y YAML
    PROJ_ROOT = Path(__file__).resolve().parents[1]
    cfg_path = (PROJ_ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_yaml(cfg_path)

    # rutas base
    outdir = (PROJ_ROOT / cfg["output"]["base_dir"]).resolve()
    ensure_dir(outdir)
    plots_dir = outdir / "plots"
    ensure_dir(plots_dir)

    # rutas de outputs según YAML
    dconf = cfg["dccm"]["output"]
    p_dccm  = outdir / dconf["dccm_full"]
    if p_dccm.suffix.lower() != ".npy":
        p_dccm = p_dccm.with_suffix(".npy")   # el core fuerza .npy si falta
    p_cross = outdir / dconf["dccm_cross"]
    p_high  = outdir / dconf["high_correlations"]

    # cargar
    if not p_dccm.exists():
        raise FileNotFoundError(f"No encuentro DCCM .npy: {p_dccm}")
    dccm = np.load(p_dccm)
    df_cross = safe_read_csv(p_cross, "dccm_cross_AB")
    df_high  = safe_read_csv(p_high,  "dccm_high_pairs")

    # sanity checks
    issues = []
    if dccm.ndim != 2 or dccm.shape[0] != dccm.shape[1]:
        issues.append(f"DCCM no es cuadrada: shape={dccm.shape}")
    if np.isnan(dccm).any():
        issues.append("DCCM contiene NaN")
    if (dccm < -1.001).any() or (dccm > 1.001).any():
        issues.append("DCCM tiene valores fuera de [-1,1] (tolerancia 1e-3)")

    # estadísticas globales de DCCM (excluyendo diagonal)
    if dccm.size > 0:
        mask_offdiag = ~np.eye(dccm.shape[0], dtype=bool)
        vals = dccm[mask_offdiag].astype(float)
        stats_global = dict(
            n=int(vals.size),
            mean=float(np.mean(vals)),
            median=float(np.median(vals)),
            min=float(np.min(vals)),
            q25=float(np.percentile(vals, 25)),
            q75=float(np.percentile(vals, 75)),
            max=float(np.max(vals)),
        )
    else:
        vals = np.array([])
        stats_global = dict(n=0, mean=np.nan, median=np.nan, min=np.nan, q25=np.nan, q75=np.nan, max=np.nan)

    # estadísticas de cross AB
    stats_cross = {}
    if not df_cross.empty and "correlation" in df_cross.columns:
        v = df_cross["correlation"].astype(float).values
        stats_cross = dict(
            n=int(v.size),
            mean=float(np.mean(v)),
            median=float(np.median(v)),
            min=float(np.min(v)),
            q25=float(np.percentile(v, 25)),
            q75=float(np.percentile(v, 75)),
            max=float(np.max(v)),
        )

    # top pares por |r| del high_pairs
    df_high_top = pd.DataFrame()
    if not df_high.empty and "correlation" in df_high.columns:
        df_high_top = df_high.copy()
        df_high_top["abs_r"] = df_high_top["correlation"].abs()
        df_high_top = df_high_top.sort_values("abs_r", ascending=False).head(20)

    # PLOTS
    plot_heatmap_square(dccm, "DCCM (submatriz)", plots_dir / "dccm_heatmap.png", max_dim=200)
    plot_hist(vals, "Histograma correlaciones (off-diagonal)", plots_dir / "dccm_hist_offdiag.png", bins=60)

    if not df_cross.empty and "correlation" in df_cross.columns:
        plot_hist(df_cross["correlation"].values, "Histograma correlaciones A↔B", plots_dir / "dccm_hist_cross_AB.png", bins=60)

    if not df_high_top.empty:
        plot_top_pairs(df_high_top, "Top 20 |r| (alta correlación)", plots_dir / "dccm_top20_high_pairs.png", n=20)

    # export auxiliar: top20_cross_AB.csv (por |r|)
    if not df_cross.empty:
        df_cross_top = df_cross.copy()
        df_cross_top["abs_r"] = df_cross_top["correlation"].abs()
        df_cross_top = df_cross_top.sort_values("abs_r", ascending=False).head(20)
        df_cross_top.drop(columns=["abs_r"], inplace=True)
        df_cross_top.to_csv(outdir / "top20_cross_AB.csv", index=False)

    # REPORTE
    lines = []
    lines.append("TEST REPORTE — PASO 03 (DCCM)")
    lines.append("=" * 72)
    lines.append(f"YAML: {cfg_path}")
    lines.append(f"OUTPUT DIR: {outdir}\n")

    lines.append("ARCHIVOS")
    lines.append(f"  - DCCM:         {p_dccm}")
    lines.append(f"  - Cross A↔B:    {p_cross}")
    lines.append(f"  - High pairs:   {p_high}")

    lines.append("\nCHEQUEOS")
    if issues:
        for msg in issues:
            lines.append(f"  ⚠ {msg}")
    else:
        lines.append("  OK: DCCM cuadrada, sin NaNs, dentro de [-1,1]")

    lines.append("\nESTADÍSTICAS DCCM (off-diagonal)")
    lines.append(f"  n={stats_global['n']}, mean={stats_global['mean']:.3f}, median={stats_global['median']:.3f}, "
                 f"min={stats_global['min']:.3f}, q25={stats_global['q25']:.3f}, q75={stats_global['q75']:.3f}, "
                 f"max={stats_global['max']:.3f}")

    lines.append("\nESTADÍSTICAS CROSS A↔B")
    if stats_cross:
        lines.append(f"  n={stats_cross['n']}, mean={stats_cross['mean']:.3f}, median={stats_cross['median']:.3f}, "
                     f"min={stats_cross['min']:.3f}, q25={stats_cross['q25']:.3f}, q75={stats_cross['q75']:.3f}, "
                     f"max={stats_cross['max']:.3f}")
    else:
        lines.append("  (sin datos o sin columna 'correlation')")

    if not df_high_top.empty:
        lines.append("\nTOP 10 high_pairs por |r|")
        for _, r in df_high_top.head(10).iterrows():
            lbl1 = f"{r['segid1']}{int(r['resid1'])}-{r['resname1']}"
            lbl2 = f"{r['segid2']}{int(r['resid2'])}-{r['resname2']}"
            lines.append(f"  {lbl1:16s} ↔ {lbl2:16s}  r={r['correlation']:.3f} ({r['type']})")

    # listar plots
    lines.append("\nPLOTS")
    for png in sorted(plots_dir.glob("*.png")):
        lines.append(f"  - {png.name}")

    report = outdir / "test_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\n✅ Reporte escrito en: {report}")


if __name__ == "__main__":
    main()

# 08_tests/04_pca_test.py
"""
Test integral del PASO 04 (PCA):
- Lee YAML (--config)
- Verifica y carga:
    * components.npy
    * variance_explained.csv
    * residue_participation.csv
    * pc_projection.csv
- Chequeos de consistencia (dimensiones, NaNs)
- Estadísticas útiles
- Plots (matplotlib) en .../04_pca/plots/
- Reporte en .../04_pca/test_report.txt

Uso:
    python 08_tests/04_pca_test.py --config 03_configs/analyses/04_pca.yaml
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

def safe_load_npy(p: Path, name: str) -> np.ndarray:
    if not p.exists():
        raise FileNotFoundError(f"No encuentro {name}: {p}")
    return np.load(p)

def safe_read_csv(p: Path, name: str) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"No encuentro {name}: {p}")
    return pd.read_csv(p)

def plot_scree(variance: np.ndarray, cumulative: np.ndarray, outpath: Path):
    # Scree (barras) + acumulada (línea)
    x = np.arange(1, len(variance)+1)
    fig = plt.figure(figsize=(7,4))
    plt.bar(x, variance)
    plt.plot(x, cumulative, marker="o")
    plt.xlabel("PC")
    plt.ylabel("Varianza explicada (fracción)")
    plt.title("Scree plot (varianza explicada y acumulada)")
    plt.xticks(x)
    plt.ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_projection_scatter(df_proj: pd.DataFrame, outpath_scatter: Path, outpath_ts: Path):
    if {"PC1","PC2"}.issubset(df_proj.columns):
        fig = plt.figure(figsize=(5,4))
        plt.scatter(df_proj["PC1"], df_proj["PC2"], s=8)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Proyección: PC1 vs PC2")
        fig.tight_layout()
        fig.savefig(outpath_scatter, dpi=150, bbox_inches="tight")
        plt.close(fig)

    if "frame" in df_proj.columns and "PC1" in df_proj.columns:
        fig = plt.figure(figsize=(7,3.5))
        plt.plot(df_proj["frame"].values, df_proj["PC1"].values)
        plt.xlabel("frame")
        plt.ylabel("PC1")
        plt.title("Evolución temporal de PC1")
        fig.tight_layout()
        fig.savefig(outpath_ts, dpi=150, bbox_inches="tight")
        plt.close(fig)

def plot_participation_top(part_df: pd.DataFrame, outpath: Path, pc_col: str = "PC1", n: int = 20):
    if part_df.empty or pc_col not in part_df.columns:
        return
    dft = part_df.sort_values(pc_col, ascending=False).head(n).copy()
    labels = [f"{r.segid}{int(r.resid)}-{r.resname}" for _, r in dft.iterrows()]
    fig = plt.figure(figsize=(10,4))
    plt.bar(range(len(dft)), dft[pc_col].values)
    plt.xticks(range(len(dft)), labels, rotation=45, ha="right", fontsize=8)
    plt.title(f"Top {n} participación por residuo — {pc_col}")
    plt.xlabel("Residuo")
    plt.ylabel("participación (normalizada)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------- main -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Full test del PASO 04 (PCA) con plots y reporte.")
    ap.add_argument("--config", required=True, help="Ruta al YAML (03_configs/analyses/04_pca.yaml)")
    args = ap.parse_args()

    # raíz del proyecto y YAML
    PROJ_ROOT = Path(__file__).resolve().parents[1]
    cfg_path = (PROJ_ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_yaml(cfg_path)

    # dir de salida y plots
    outdir = (PROJ_ROOT / cfg["output"]["base_dir"]).resolve()
    ensure_dir(outdir)
    plots_dir = outdir / "plots"
    ensure_dir(plots_dir)

    # rutas según YAML
    pca_out = cfg["pca"]["output"]
    p_components = outdir / pca_out["components"]
    if p_components.suffix.lower() != ".npy":
        p_components = p_components.with_suffix(".npy")
    p_variance   = outdir / pca_out["variance_explained"]
    p_particip   = outdir / pca_out["participation"]
    p_projection = outdir / pca_out["projection"]

    # cargar outputs
    components = safe_load_npy(p_components, "components.npy")          # (nPC, 3N)
    df_var     = safe_read_csv(p_variance, "variance_explained.csv")    # PC, variance_explained, cumulative_variance
    df_part    = safe_read_csv(p_particip, "residue_participation.csv") # resid, resname, segid, PC1..PCk
    df_proj    = safe_read_csv(p_projection, "pc_projection.csv")       # PC1..PCk, frame

    # chequeos
    issues = []

    # 1) componentes vs varianza/proyección
    if "variance_explained" not in df_var.columns or "cumulative_variance" not in df_var.columns:
        issues.append("variance_explained.csv no tiene columnas esperadas (variance_explained, cumulative_variance)")
    nPC_var = len(df_var)
    nPC_comp = components.shape[0] if components.ndim == 2 else 0
    nPC_proj = df_proj.drop(columns=[c for c in ["frame"] if c in df_proj.columns]).shape[1]
    if nPC_comp != nPC_var or nPC_comp != nPC_proj:
        issues.append(f"Inconsistencia #PC: components={nPC_comp}, variance={nPC_var}, projection={nPC_proj}")

    if components.ndim != 2:
        issues.append(f"components.npy no es 2D: shape={components.shape}")

    # 2) participación: columnas PCi
    pc_cols_part = [c for c in df_part.columns if c.startswith("PC")]
    if len(pc_cols_part) != nPC_var:
        issues.append(f"participation PC-cols={len(pc_cols_part)} difiere de nPC={nPC_var}")
    if df_part.isna().sum().sum() > 0:
        issues.append("residue_participation.csv contiene NaN")
    if df_proj.isna().sum().sum() > 0:
        issues.append("pc_projection.csv contiene NaN")
    if df_var.isna().sum().sum() > 0:
        issues.append("variance_explained.csv contiene NaN")

    # 3) valores de varianza
    if not df_var.empty:
        ve = df_var["variance_explained"].values
        if (ve < -1e-9).any() or (ve > 1+1e-9).any():
            issues.append("variance_explained fuera de rango [0,1]")
        if df_var["cumulative_variance"].iloc[-1] > 1.0001:
            issues.append("cumulative_variance > 1")

    # estadísticas rápidas
    stats = {}
    if not df_var.empty:
        stats["cum_var_last"] = float(df_var["cumulative_variance"].iloc[-1])
        stats["pc1_var"] = float(df_var["variance_explained"].iloc[0])

    # Plots
    if not df_var.empty:
        plot_scree(
            df_var["variance_explained"].values,
            df_var["cumulative_variance"].values,
            plots_dir / "pca_scree.png"
        )
    if not df_proj.empty:
        plot_projection_scatter(
            df_proj,
            plots_dir / "projection_pc1_pc2.png",
            plots_dir / "projection_pc1_timeseries.png"
        )
    if not df_part.empty:
        plot_participation_top(df_part, plots_dir / "participation_top20_PC1.png", pc_col="PC1", n=20)

    # Reporte
    lines = []
    lines.append("TEST REPORTE — PASO 04 (PCA)")
    lines.append("=" * 72)
    lines.append(f"YAML: {cfg_path}")
    lines.append(f"OUTPUT DIR: {outdir}\n")

    lines.append("ARCHIVOS")
    lines.append(f"  - components:           {p_components}")
    lines.append(f"  - variance_explained:   {p_variance}")
    lines.append(f"  - residue_participation:{p_particip}")
    lines.append(f"  - pc_projection:        {p_projection}")

    lines.append("\nCHEQUEOS")
    if issues:
        for m in issues:
            lines.append(f"  ⚠ {m}")
    else:
        lines.append("  OK: dimensiones y contenidos coherentes.")

    if stats:
        lines.append("\nESTADÍSTICAS")
        lines.append(f"  Varianza acumulada (PC1..PCk): {stats['cum_var_last']:.3f}")
        lines.append(f"  Varianza PC1: {stats['pc1_var']:.3f}")

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

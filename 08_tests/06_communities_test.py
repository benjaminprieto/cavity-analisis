# 08_tests/06_communities_test.py
"""
Test integral del PASO 06 (Communities)
- Verifica artefactos esperados:
    * communities.json
    * community_assignment.csv
    * community_stats.csv
    * inter_community_residues.csv
- Chequeos de consistencia (columnas, NaNs, tamaños)
- Plots:
    * tamaños de comunidad (top 20)
    * distribución de tamaños
    * top inter-comunidad por n_inter_connections
- Reporte en .../06_communities/test_report.txt

Uso:
    python 08_tests/06_communities_test.py --config 03_configs/analyses/06_communities.yaml
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt


# -------------------------- utilidades base ----------------------------------

def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_read_csv(p: Path, name: str) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"No encuentro {name}: {p}")
    return pd.read_csv(p)

def safe_read_json(p: Path, name: str) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"No encuentro {name}: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def plot_hist(arr: np.ndarray, title: str, outpath: Path, bins: int = 30):
    if arr.size == 0:
        return
    fig = plt.figure(figsize=(6,4))
    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.xlabel("valor")
    plt.ylabel("conteo")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_bars(labels: list[str], values: list[float], title: str, outpath: Path):
    if not labels or not values:
        return
    fig = plt.figure(figsize=(10,4))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.title(title)
    plt.ylabel("valor")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------- main -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Test PASO 06 (Communities) — validación + plots + reporte")
    ap.add_argument("--config", required=True, help="03_configs/analyses/06_communities.yaml")
    args = ap.parse_args()

    PROJ_ROOT = Path(__file__).resolve().parents[1]
    cfg_path = (PROJ_ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_yaml(cfg_path)

    outdir = (PROJ_ROOT / cfg["output"]["base_dir"]).resolve()
    ensure_dir(outdir)
    plots_dir = outdir / "plots"
    ensure_dir(plots_dir)

    out_cfg = cfg["community_detection"]["output"]
    p_json   = outdir / out_cfg["communities_json"]
    p_assign = outdir / out_cfg["community_assignment"]
    p_stats  = outdir / out_cfg["community_stats"]
    p_inter  = outdir / out_cfg["inter_community"]

    # cargar
    j = safe_read_json(p_json,   "communities.json")
    df_assign = safe_read_csv(p_assign, "community_assignment.csv")
    df_stats  = safe_read_csv(p_stats,  "community_stats.csv")
    df_inter  = safe_read_csv(p_inter,  "inter_community_residues.csv")

    issues = []

    # chequeos claves
    if not {"node","community","resid","resname","segid"}.issubset(df_assign.columns):
        issues.append("community_assignment.csv: faltan columnas esperadas (node, community, resid, resname, segid).")
    if not {"community","n_nodes","residues"}.issubset(df_stats.columns):
        issues.append("community_stats.csv: faltan columnas esperadas (community, n_nodes, residues).")
    if not df_inter.empty:
        if not {"node","resid","resname","segid","community","n_inter_connections","degree"}.issubset(df_inter.columns):
            issues.append("inter_community_residues.csv: faltan columnas esperadas.")

    # NaNs
    if df_assign.isna().sum().sum() > 0:
        issues.append("community_assignment.csv contiene NaN")
    if df_stats.isna().sum().sum() > 0:
        issues.append("community_stats.csv contiene NaN")
    if df_inter.isna().sum().sum() > 0:
        issues.append("inter_community_residues.csv contiene NaN")

    # consistencia: n_nodes coincide con conteo real por comunidad
    # (no estricto por segid, pero chequeo básico)
    sizes_from_assign = df_assign.groupby("community")["node"].count().rename("n_nodes_calc")
    df_stats = df_stats.merge(sizes_from_assign, left_on="community", right_index=True, how="left")
    if (df_stats["n_nodes"] != df_stats["n_nodes_calc"]).any():
        issues.append("n_nodes en community_stats.csv no coincide con conteo en community_assignment.csv")

    # métricas rápidas
    n_nodes_total = int(df_assign.shape[0])
    n_comms = int(df_assign["community"].nunique())
    size_arr = df_stats["n_nodes"].to_numpy(dtype=float) if "n_nodes" in df_stats.columns else np.array([])
    max_comm = int(df_stats.loc[df_stats["n_nodes"].idxmax(), "community"]) if not df_stats.empty else -1
    max_size = int(df_stats["n_nodes"].max()) if not df_stats.empty else 0

    # plots
    if not df_stats.empty:
        # top 20 comunidades por tamaño
        df_top = df_stats.sort_values("n_nodes", ascending=False).head(20)
        labels = [f"C{int(c)}" for c in df_top["community"].values]
        vals   = df_top["n_nodes"].astype(int).tolist()
        plot_bars(labels, vals, "Top 20 — Tamaño por comunidad", plots_dir / "communities_top20_sizes.png")
        plot_hist(size_arr, "Distribución de tamaños de comunidad", plots_dir / "communities_sizes_hist.png", bins=30)

    if not df_inter.empty:
        df_ib = df_inter.sort_values(["n_inter_connections","degree"], ascending=False).head(20)
        labels = [f"{r.segid}{int(r.resid)}" for _, r in df_ib.iterrows()]
        vals   = df_ib["n_inter_connections"].astype(int).tolist()
        plot_bars(labels, vals, "Top 20 puentes inter-comunidad", plots_dir / "inter_community_top20.png")

    # reporte
    lines = []
    lines.append("TEST REPORTE — PASO 06 (COMMUNITIES)")
    lines.append("=" * 72)
    lines.append(f"YAML: {cfg_path}")
    lines.append(f"OUTPUT DIR: {outdir}\n")

    lines.append("ARCHIVOS")
    lines.append(f"  - communities.json:             {p_json}")
    lines.append(f"  - community_assignment.csv:     {p_assign}")
    lines.append(f"  - community_stats.csv:          {p_stats}")
    lines.append(f"  - inter_community_residues.csv: {p_inter}")

    lines.append("\nCHEQUEOS")
    if issues:
        for m in issues:
            lines.append(f"  ⚠ {m}")
    else:
        lines.append("  OK: estructura y contenidos coherentes.")

    lines.append("\nMÉTRICAS")
    lines.append(f"  N nodos total: {n_nodes_total}")
    lines.append(f"  N comunidades: {n_comms}")
    if n_comms > 0:
        lines.append(f"  Tamaño máx comunidad: C{max_comm} con {max_size} nodos")

    lines.append("\nPLOTS")
    for png in sorted(plots_dir.glob("*.png")):
        lines.append(f"  - {png.name}")

    report = outdir / "test_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n✅ Reporte escrito en: {report}")


if __name__ == "__main__":
    main()

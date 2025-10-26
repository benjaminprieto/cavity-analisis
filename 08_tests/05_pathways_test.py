# 08_tests/05_pathways_test.py
"""
Test integral del PASO 05 (Pathways)
- Si AÚN NO corres el módulo (p.ej. falta energía), te valida el YAML y te lista
  los prerequisitos faltantes + sugerencias (cambiar a 'persistence', etc.)
- Si YA tienes resultados, valida y resume:
    * weighted_paths.json
    * bottleneck_residues.csv
    * pathway_importance.csv
  + genera gráficos en .../05_pathways/plots/
  + escribe un reporte en .../05_pathways/test_report.txt

Uso:
  python 08_tests/05_pathways_test.py --config 03_configs/analyses/05_pathways.yaml
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- utilidades base -------------------------------

def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_read_csv(p: Path, name: str) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"No encuentro {name}: {p}")
    return pd.read_csv(p)

def safe_read_json(p: Path, name: str):
    if not p.exists():
        raise FileNotFoundError(f"No encuentro {name}: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def plot_hist(arr: np.ndarray, title: str, outpath: Path, bins: int = 40):
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
    ap = argparse.ArgumentParser(description="Test PASO 05 (Pathways) — prereqs o resultados + plots + reporte.")
    ap.add_argument("--config", required=True, help="Ruta al YAML (03_configs/analyses/05_pathways.yaml)")
    args = ap.parse_args()

    PROJ_ROOT = Path(__file__).resolve().parents[1]
    cfg_path = (PROJ_ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_yaml(cfg_path)

    # --- Rutas principales desde YAML ---
    outdir = (PROJ_ROOT / cfg["output"]["base_dir"]).resolve()
    ensure_dir(outdir)
    plots_dir = outdir / "plots"
    ensure_dir(plots_dir)

    p_graph   = (PROJ_ROOT / cfg["artifacts"]["network_graph"]).resolve()
    p_ranking = p_graph.with_name("centrality_ranking.csv")
    edge_mode = str(cfg["pathways"]["edge_weight"]).lower()

    p_inter = None
    if edge_mode == "interaction_energy":
        p_inter = (PROJ_ROOT / cfg["artifacts"]["inter_energy"]).resolve()

    # Outputs esperados
    p_paths = outdir / cfg["pathways"]["output"]["pathways"]
    p_bottl = outdir / cfg["pathways"]["output"]["bottlenecks"]
    p_impc  = outdir / cfg["pathways"]["output"]["importance"]

    # --- Revisión de prerequisitos ---
    prereq_missing = []
    if not p_graph.exists():
        prereq_missing.append(f"No existe grafo de red (Módulo 1): {p_graph}")
    if not p_ranking.exists():
        prereq_missing.append(f"No existe centrality_ranking.csv junto al grafo: {p_ranking}")
    if edge_mode == "interaction_energy" and (p_inter is None or not p_inter.exists()):
        prereq_missing.append(f"No existe inter_energy CSV (Módulo 2): {p_inter}")

    # Si faltan prerequisitos, generamos reporte de prereqs y salimos
    if prereq_missing:
        lines = []
        lines.append("TEST PASO 05 — PREREQUISITOS INCOMPLETOS")
        lines.append("=" * 72)
        lines.append(f"YAML: {cfg_path}")
        lines.append("\nFALTAN:")
        for m in prereq_missing:
            lines.append(f"  - {m}")

        # Sugerencias útiles
        lines.append("\nSUGERENCIAS:")
        if edge_mode == "interaction_energy":
            lines.append("  * Aún no tienes energías. Dos opciones:")
            lines.append("    1) Ejecuta el Módulo 2 (02_energy.py) para generar inter_nrp1_xylt1.csv.")
            lines.append("    2) O usa el grafo de persistencia cambiando en YAML:")
            lines.append("       pathways.edge_weight: persistence")
        else:
            lines.append("  * Con 'persistence' solo necesitas los artefactos del Módulo 1 (grafo + ranking).")

        # PDB/DCD para interfaz
        p_pdb = (PROJ_ROOT / cfg["inputs"]["pdb"]).resolve()
        p_dcd = (PROJ_ROOT / cfg["inputs"]["dcd"]).resolve()
        if not p_pdb.exists():
            lines.append(f"  * Revisa PDB (para detectar interfaz): {p_pdb} (no encontrado)")
        if not p_dcd.exists():
            lines.append(f"  * Revisa DCD (para detectar interfaz): {p_dcd} (no encontrado)")

        report = outdir / "test_prereqs_report.txt"
        report.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        print(f"\n⚠️ Reporte escrito en: {report}")
        return

    # --- Si existen resultados, leer y validar; si no, solo avisar que puedes correr el módulo ---
    have_outputs = p_paths.exists() and p_bottl.exists() and p_impc.exists()
    if not have_outputs:
        lines = []
        lines.append("TEST PASO 05 — LISTO PARA EJECUTAR")
        lines.append("=" * 72)
        lines.append(f"YAML: {cfg_path}")
        lines.append("\nPrerequisitos OK. Aún no hay resultados de pathways en:")
        lines.append(f"  - {p_paths}")
        lines.append(f"  - {p_bottl}")
        lines.append(f"  - {p_impc}")
        lines.append("\nEjecuta:")
        lines.append("  python 02_scripts/05_pathways.py --config 03_configs/analyses/05_pathways.yaml")
        report = outdir / "test_ready_report.txt"
        report.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        print(f"\nℹ️ Reporte escrito en: {report}")
        return

    # ---------------------- VALIDACIÓN Y RESÚMENES ----------------------------
    paths_list = safe_read_json(p_paths, "weighted_paths.json")
    df_bott = safe_read_csv(p_bottl, "bottleneck_residues.csv")
    df_imp  = safe_read_csv(p_impc,  "pathway_importance.csv")

    issues = []

    # Chequeos básicos
    if not isinstance(paths_list, list):
        issues.append("weighted_paths.json no es una lista.")
    else:
        if len(paths_list) > 0:
            sample = paths_list[0]
            for k in ["source", "target", "path", "length", "residues", "score"]:
                if k not in sample:
                    issues.append(f"weighted_paths.json: falta clave '{k}' en items.")

    if not {"resid","appearances","frequency"}.issubset(df_bott.columns):
        issues.append("bottleneck_residues.csv no tiene columnas esperadas (resid, appearances, frequency).")

    if not {"source","target","path","length","residues","score"}.issubset(df_imp.columns):
        issues.append("pathway_importance.csv no tiene columnas esperadas.")

    # Métricas rápidas de rutas
    n_paths = len(paths_list) if isinstance(paths_list, list) else 0
    lengths = np.array([p.get("length", 0) for p in paths_list], dtype=float) if n_paths else np.array([])
    scores  = np.array([p.get("score", 0.0)  for p in paths_list], dtype=float) if n_paths else np.array([])

    # Plots
    if n_paths:
        plot_hist(lengths, "Distribución de longitudes de ruta", plots_dir / "path_lengths_hist.png", bins=30)
        plot_hist(scores,  "Distribución de score (coste)",     plots_dir / "path_scores_hist.png", bins=30)

    if not df_bott.empty:
        # Top 20 bottlenecks
        df_topb = df_bott.sort_values("frequency", ascending=False).head(20)
        labels = [f"{int(r.resid)}" for _, r in df_topb.iterrows()]
        vals   = df_topb["frequency"].astype(float).tolist()
        plot_bars(labels, vals, "Top 20 bottlenecks (frecuencia)", plots_dir / "bottlenecks_top20.png")

    # Resumen textual
    lines = []
    lines.append("TEST REPORTE — PASO 05 (Pathways)")
    lines.append("=" * 72)
    lines.append(f"YAML: {cfg_path}")
    lines.append(f"OUTPUT DIR: {outdir}")
    lines.append(f"edge_weight: {edge_mode}\n")

    lines.append("ARCHIVOS")
    lines.append(f"  - weighted_paths.json:   {p_paths}")
    lines.append(f"  - bottleneck_residues:   {p_bottl}")
    lines.append(f"  - pathway_importance:    {p_impc}")

    lines.append("\nCHEQUEOS")
    if issues:
        for m in issues:
            lines.append(f"  ⚠ {m}")
    else:
        lines.append("  OK: Estructura de archivos consistente.")

    lines.append("\nRESUMEN DE RUTAS")
    lines.append(f"  total paths: {n_paths}")
    if n_paths:
        lines.append(f"  length: min={lengths.min():.0f}, p50={np.median(lengths):.0f}, max={lengths.max():.0f}")
        lines.append(f"  score : min={scores.min():.3f}, p50={np.median(scores):.3f}, max={scores.max():.3f}")

        # Top 10 por score (mejores)
        df_best = pd.DataFrame(paths_list).sort_values("score", ascending=True).head(10)
        lines.append("\n  TOP 10 por score (mejores):")
        for _, r in df_best.iterrows():
            lines.append(f"    src={int(r['source'])} → tgt={int(r['target'])}  len={int(r['length'])}  score={float(r['score']):.3f}")

    if not df_bott.empty:
        lines.append("\nBOTTLENECKS (Top 10)")
        for _, r in df_bott.sort_values("frequency", ascending=False).head(10).iterrows():
            lines.append(f"  resid={int(r['resid'])}  freq={float(r['frequency']):.3f}  appearances={int(r['appearances'])}")

    lines.append("\nPLOTS")
    for png in sorted(plots_dir.glob("*.png")):
        lines.append(f"  - {png.name}")

    report = outdir / "test_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n✅ Reporte escrito en: {report}")


if __name__ == "__main__":
    main()

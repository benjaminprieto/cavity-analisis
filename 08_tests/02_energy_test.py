# 08_tests/02_energy_test.py
"""
Test integral del PASO 02 (energy):
- Lee YAML (ruta por --config)
- Verifica y carga los 4 outputs:
    * intra_nrp1_matrix.csv
    * intra_xylt1_matrix.csv
    * inter_nrp1_xylt1.csv
    * binding_per_residue.csv
- Chequeos de consistencia (columnas, NaNs, tamaños)
- Métricas/resúmenes útiles
- Plots (matplotlib puro) en .../02_energy/plots/
- Reporte de texto en .../02_energy/test_report.txt

Uso:
    python 08_tests/02_energy_test.py --config 03_configs/analyses/02_energy.yaml
"""

from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------- utilidades base --------------------------------

def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def check_required_cols(df: pd.DataFrame, required: list[str], name: str) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    issues = []
    if missing:
        issues.append(f"{name}: faltan columnas {missing}")
    if df.empty:
        issues.append(f"{name}: dataframe vacío")
    if df.isna().sum().sum() > 0:
        issues.append(f"{name}: contiene NaN")
    return issues

def quick_stats_energy(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return dict(label=label, n_pairs=0)
    v = df["e_total"].values
    return dict(
        label=label,
        n_pairs=int(len(v)),
        mean=float(np.mean(v)),
        median=float(np.median(v)),
        min=float(np.min(v)),
        max=float(np.max(v)),
        q25=float(np.percentile(v, 25)),
        q75=float(np.percentile(v, 75)),
    )

def safe_read_csv(p: Path, name: str) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"No encuentro {name}: {p}")
    return pd.read_csv(p)

def plot_hist(values: np.ndarray, title: str, outpath: Path, bins: int = 50):
    fig = plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel("E_total (kcal/mol)")
    plt.ylabel("conteo")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_top_bars(df: pd.DataFrame, title: str, outpath: Path, n:int=20):
    # df debe tener columnas: segid,resid,resname,binding_energy
    if df.empty:
        return
    top = df.sort_values("binding_energy").head(n).copy()  # más favorables (más negativos)
    labels = [f"{r.segid}{int(r.resid)}-{r.resname}" for _, r in top.iterrows()]
    fig = plt.figure(figsize=(10, 4))
    plt.bar(range(len(top)), top["binding_energy"].values)
    plt.xticks(range(len(top)), labels, rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("Residuo")
    plt.ylabel("Binding energy (kcal/mol)")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_inter_heatmap_small(df_inter: pd.DataFrame, outpath: Path, max_pairs: int = 400):
    """
    Heatmap compacta de los pares inter con E_total más favorables (negativos).
    Limita a max_pairs para que no explote la figura.
    """
    if df_inter.empty:
        return
    # quedarnos con los más negativos
    df_sorted = df_inter.sort_values("e_total").head(max_pairs).copy()
    # etiqueta tipo "A123-GLU" y "B45-LYS"
    df_sorted["r1"] = df_sorted.apply(lambda r: f"{r['segid1']}{int(r['resid1'])}-{r['resname1']}", axis=1)
    df_sorted["r2"] = df_sorted.apply(lambda r: f"{r['segid2']}{int(r['resid2'])}-{r['resname2']}", axis=1)
    # pivote — puede ser rectangular; reemplazamos NaN por 0 para dibujar
    piv = df_sorted.pivot_table(index="r1", columns="r2", values="e_total", aggfunc="mean").fillna(0.0)
    if piv.shape[0] == 0 or piv.shape[1] == 0:
        return
    # por límites de tamaño, recortar si es gigantesco
    max_dim = 40
    piv = piv.iloc[:max_dim, :max_dim]

    fig = plt.figure(figsize=(max(6, piv.shape[1]*0.3), max(4, piv.shape[0]*0.3)))
    im = plt.imshow(piv.values, interpolation="nearest", aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="E_total (kcal/mol)")
    plt.yticks(range(piv.shape[0]), piv.index, fontsize=8)
    plt.xticks(range(piv.shape[1]), piv.columns, rotation=90, fontsize=8)
    plt.title("Interacción NRP1↔XYLT1 (pares más favorables)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------- main -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Full test del PASO 02 (energy) con plots y reporte.")
    ap.add_argument("--config", required=True, help="Ruta al YAML (03_configs/analyses/02_energy.yaml)")
    args = ap.parse_args()

    # localizar raíz del repo y YAML
    PROJ_ROOT = Path(__file__).resolve().parents[1]
    cfg_path = (PROJ_ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_yaml(cfg_path)

    base_dir = (PROJ_ROOT / cfg["output"]["base_dir"]).resolve()
    ensure_dir(base_dir)
    plots_dir = base_dir / "plots"
    ensure_dir(plots_dir)

    # rutas de salida (según YAML)
    out_cfg = cfg["energy_decomposition"]["output"]
    p_intra_A = base_dir / out_cfg["intra_nrp1_matrix"]
    p_intra_B = base_dir / out_cfg["intra_xylt1_matrix"]
    p_inter   = base_dir / out_cfg["inter_energy"]
    p_bind    = base_dir / out_cfg["binding_energy_per_residue"]

    # cargar
    df_intra_A = safe_read_csv(p_intra_A, "intra_nrp1_matrix")
    df_intra_B = safe_read_csv(p_intra_B, "intra_xylt1_matrix")
    df_inter   = safe_read_csv(p_inter,   "inter_nrp1_xylt1")
    df_bind    = safe_read_csv(p_bind,    "binding_per_residue")

    # chequeos de columnas
    issues = []
    issues += check_required_cols(df_intra_A, ["resid1","resname1","segid1","resid2","resname2","segid2","e_total"], "intra_nrp1")
    issues += check_required_cols(df_intra_B, ["resid1","resname1","segid1","resid2","resname2","segid2","e_total"], "intra_xylt1")
    issues += check_required_cols(df_inter,   ["resid1","resname1","segid1","resid2","resname2","segid2","e_total"], "inter")
    issues += check_required_cols(df_bind,    ["segid","resid","resname","binding_energy"], "binding_per_residue")

    # estadísticas rápidas
    s_intra_A = quick_stats_energy(df_intra_A, "intra_NRP1(A)")
    s_intra_B = quick_stats_energy(df_intra_B, "intra_XYLT1(B)")
    s_inter   = quick_stats_energy(df_inter,   "inter_A<->B")

    # resumen por cadena (sanity de segid en intra/inter)
    chains_intra_A = df_intra_A[["segid1","segid2"]].stack().value_counts().to_dict() if not df_intra_A.empty else {}
    chains_intra_B = df_intra_B[["segid1","segid2"]].stack().value_counts().to_dict() if not df_intra_B.empty else {}
    chains_inter   = df_inter[["segid1","segid2"]].stack().value_counts().to_dict() if not df_inter.empty else {}

    # top pares inter más favorables
    top_pairs = pd.DataFrame()
    if not df_inter.empty:
        top_pairs = df_inter.sort_values("e_total").head(20).copy()
        # etiqueta corta
        top_pairs["pair"] = top_pairs.apply(
            lambda r: f"{r['segid1']}{int(r['resid1'])}-{r['resname1']}__{r['segid2']}{int(r['resid2'])}-{r['resname2']}",
            axis=1
        )

    # Plots
    if not df_intra_A.empty:
        plot_hist(df_intra_A["e_total"].values, "Distribución E_total intra NRP1 (A)", plots_dir / "hist_intra_A.png")
    if not df_intra_B.empty:
        plot_hist(df_intra_B["e_total"].values, "Distribución E_total intra XYLT1 (B)", plots_dir / "hist_intra_B.png")
    if not df_inter.empty:
        plot_hist(df_inter["e_total"].values, "Distribución E_total inter (A<->B)", plots_dir / "hist_inter.png")
        plot_inter_heatmap_small(df_inter, plots_dir / "heatmap_inter_small.png", max_pairs=400)

    if not df_bind.empty:
        plot_top_bars(df_bind, "Top 20 residuos (A) más favorables al binding", plots_dir / "top20_binding_A.png", n=20)

    # export auxiliar: top_pairs_inter.csv
    if not top_pairs.empty:
        top_pairs[["pair","e_total","e_coulomb","e_lj"]].to_csv(base_dir / "top_pairs_inter.csv", index=False)

    # reporte
    lines = []
    lines.append("TEST REPORTE — PASO 02 (energy)")
    lines.append("=" * 72)
    lines.append(f"YAML: {cfg_path}")
    lines.append(f"OUTPUT: {base_dir}\n")

    lines.append("ARCHIVOS")
    lines.append(f"  - intra_nrp1_matrix: {p_intra_A}")
    lines.append(f"  - intra_xylt1_matrix: {p_intra_B}")
    lines.append(f"  - inter_nrp1_xylt1: {p_inter}")
    lines.append(f"  - binding_per_residue: {p_bind}")

    lines.append("\nCHEQUEOS")
    if issues:
        for msg in issues:
            lines.append(f"  ⚠ {msg}")
    else:
        lines.append("  OK: columnas y datos válidos en los 4 outputs")

    lines.append("\nESTADÍSTICAS (E_total)")
    for st in [s_intra_A, s_intra_B, s_inter]:
        lines.append(f"  - {st['label']}: n={st['n_pairs']}, mean={st.get('mean',None):.3f}, median={st.get('median',None):.3f}, "
                     f"min={st.get('min',None):.3f}, q25={st.get('q25',None):.3f}, q75={st.get('q75',None):.3f}, max={st.get('max',None):.3f}")

    lines.append("\nSEGID (conteos) — sanity")
    lines.append(f"  - intra_NRP1(A): {chains_intra_A}")
    lines.append(f"  - intra_XYLT1(B): {chains_intra_B}")
    lines.append(f"  - inter_A<->B   : {chains_inter}")

    if not df_bind.empty:
        lines.append("\nTOP 10 contribuyentes (A) más favorables (binding_energy más negativo)")
        cols = ["segid","resid","resname","binding_energy","e_coulomb","e_lj"]
        lines += ["    " + " | ".join(map(str, row)) for row in df_bind.sort_values("binding_energy").head(10)[cols].values]

    if not top_pairs.empty:
        lines.append("\nTOP 10 pares inter más favorables (E_total más negativo)")
        for _, r in top_pairs.head(10).iterrows():
            lines.append(f"    {r['pair']:35s}  E_total={r['e_total']:.3f}  (Coul={r['e_coulomb']:.3f}, LJ={r['e_lj']:.3f})")

    # listar plots generados
    lines.append("\nPLOTS")
    for png in sorted(plots_dir.glob("*.png")):
        lines.append(f"  - {png.name}")

    report = base_dir / "test_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\n✅ Reporte escrito en: {report}")


if __name__ == "__main__":
    main()

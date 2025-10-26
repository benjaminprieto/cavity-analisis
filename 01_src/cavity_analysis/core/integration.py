# 01_src/cavity_analysis/core/integration.py
"""
Módulo 7: Integration
Integra resultados de los módulos 01–05 para priorizar residuos críticos.
Lee artefactos desde disco (CSV/JSON) y produce:
  - critical_residues_final.csv
  - cavity_is_critical.json (opcional)
  - summary_report.txt
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple
import os
import json
import numpy as np
import pandas as pd


# ------------------------------ utilidades IO --------------------------------

def _read_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _try_read_csv(p: Path) -> pd.DataFrame:
    if not p or not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def _normalize_0_1(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    smin = series.min()
    smax = series.max()
    if pd.isna(smin) or pd.isna(smax) or smax - smin == 0:
        return pd.Series(0.0, index=series.index)
    return (series - smin) / (smax - smin)


# ----------------------------- integradores parciales ------------------------

def integrate_network_from_csv(ranking_csv: Path, verbose=True) -> pd.DataFrame:
    """
    Espera un CSV con columnas al menos:
      resid, resname, segid, protein, betweenness, eigenvector, closeness, degree
    Devuelve: resid, resname, segid, protein, network_score
    """
    if verbose:
        print("  Procesando network centrality…")
    df = _try_read_csv(ranking_csv)
    if df.empty:
        if verbose: print("    ⚠️ network: no se pudo leer ranking CSV.")
        return pd.DataFrame(columns=["resid","resname","segid","protein","network_score"])

    for col in ["betweenness","eigenvector","closeness","degree"]:
        if col in df.columns:
            df[col + "_norm"] = _normalize_0_1(df[col])
    norm_cols = [c for c in df.columns if c.endswith("_norm")]
    df["network_score"] = df[norm_cols].mean(axis=1) if norm_cols else 0.0
    cols = ["resid","resname","segid","protein","network_score"]
    return df[cols].copy()


def integrate_energy_from_csv(binding_csv: Path, verbose=True) -> pd.DataFrame:
    """
    Espera binding_per_residue.csv con columnas:
      segid, resid, resname, binding_energy (negativo = favorable)
    Devuelve: resid, resname, segid, energy_score
    """
    if verbose:
        print("  Procesando energy contribution…")
    df = _try_read_csv(binding_csv)
    if df.empty:
        if verbose: print("    ⚠️ energy: no hay binding_per_residue.csv")
        return pd.DataFrame(columns=["resid","resname","segid","energy_score"])
    if "binding_energy" not in df.columns:
        if verbose: print("    ⚠️ energy: falta columna 'binding_energy'.")
        return pd.DataFrame(columns=["resid","resname","segid","energy_score"])

    df["energy_score"] = _normalize_0_1(-df["binding_energy"])
    return df[["resid","resname","segid","energy_score"]].copy()


def integrate_dccm_from_csv(high_corr_csv: Path, verbose=True) -> pd.DataFrame:
    """
    Espera high_correlations.csv con columnas:
      resid1, resid2, correlation
    Devuelve: resid, dccm_score
    """
    if verbose:
        print("  Procesando DCCM participation…")
    df = _try_read_csv(high_corr_csv)
    if df.empty:
        if verbose: print("    ⚠️ dccm: no hay high_correlations.csv")
        return pd.DataFrame(columns=["resid","dccm_score"])
    if not {"resid1","resid2","correlation"}.issubset(df.columns):
        if verbose: print("    ⚠️ dccm: faltan columnas esperadas.")
        return pd.DataFrame(columns=["resid","dccm_score"])

    from collections import Counter
    acc = Counter()
    for _, r in df.iterrows():
        try:
            c = abs(float(r["correlation"]))
        except Exception:
            c = 0.0
        acc[int(r["resid1"])] += c
        acc[int(r["resid2"])] += c

    out = pd.DataFrame([{"resid": k, "dccm_participation": v} for k, v in acc.items()])
    out["dccm_score"] = _normalize_0_1(out["dccm_participation"])
    return out[["resid","dccm_score"]].copy()


def integrate_pca_from_csv(part_csv: Path, n_components=3, verbose=True) -> pd.DataFrame:
    """
    Espera residue_participation.csv con columnas:
      resid, resname, segid, PC1, PC2, ...
    Devuelve: resid, resname, segid, pca_score
    """
    if verbose:
        print("  Procesando PCA participation…")
    df = _try_read_csv(part_csv)
    if df.empty:
        if verbose: print("    ⚠️ pca: no hay residue_participation.csv")
        return pd.DataFrame(columns=["resid","resname","segid","pca_score"])

    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    pc_cols = [c for c in pc_cols if c in df.columns]
    if pc_cols:
        df["pca_participation"] = df[pc_cols].mean(axis=1)
    else:
        df["pca_participation"] = 0.0
    df["pca_score"] = _normalize_0_1(df["pca_participation"])
    return df[["resid","resname","segid","pca_score"]].copy()


def integrate_pathways_from_csv(bottlenecks_csv: Path, verbose=True) -> pd.DataFrame:
    """
    Espera bottleneck_residues.csv con columnas:
      resid, frequency
    Devuelve: resid, pathway_score
    """
    if verbose:
        print("  Procesando pathway bottlenecks…")
    df = _try_read_csv(bottlenecks_csv)
    if df.empty:
        if verbose: print("    ⚠️ pathways: no hay bottleneck_residues.csv")
        return pd.DataFrame(columns=["resid","pathway_score"])
    if "frequency" not in df.columns:
        if verbose: print("    ⚠️ pathways: falta 'frequency'.")
        return pd.DataFrame(columns=["resid","pathway_score"])

    df["pathway_score"] = _normalize_0_1(df["frequency"])
    return df[["resid","pathway_score"]].copy()


# ----------------------------- fusión y reporte ------------------------------

def _merge_scores(
    network_df: pd.DataFrame,
    energy_df: pd.DataFrame,
    dccm_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    path_df: pd.DataFrame,
    weights: Dict[str, float],
    verbose=True
) -> pd.DataFrame:

    if network_df.empty:
        raise ValueError("Network ranking vacío: no se puede integrar sin base de residuos.")

    df = network_df.copy()

    def _left_merge(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
        return base.merge(other, on="resid", how="left")

    if not energy_df.empty:
        df = _left_merge(df, energy_df)
    if not dccm_df.empty:
        df = _left_merge(df, dccm_df)
    if not pca_df.empty:
        df = _left_merge(df, pca_df)
    if not path_df.empty:
        df = _left_merge(df, path_df)

    # rellenar NaN
    for col in ["network_score","energy_score","dccm_score","pca_score","pathway_score"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    if verbose:
        print("\n  Aplicando pesos:")
    # normaliza si no suman 1
    wsum = sum(weights.values())
    if abs(wsum - 1.0) > 1e-6:
        weights = {k: (v / wsum) for k, v in weights.items()}
        if verbose:
            print(f"    (normalizados) suma=1.0")

    df["critical_score"] = 0.0
    for k, w in weights.items():
        col = f"{k}_score"
        if col in df.columns:
            df["critical_score"] += df[col] * float(w)
            if verbose:
                print(f"    {k}: {w:.2%}")
        else:
            if verbose:
                print(f"    {k}: 0.00% (no disponible)")

    df = df.sort_values("critical_score", ascending=False)
    return df


def _cavity_check(
    integrated_df: pd.DataFrame,
    cavity_residues: list[int],
    threshold: float,
    verbose=True
) -> Dict[str, Any]:

    if verbose:
        print("\n  Analizando importancia de cavidad…")
    cav = integrated_df[integrated_df["resid"].isin(cavity_residues)].copy()
    n_total = len(cav)
    if n_total == 0:
        return {
            "n_total": 0,
            "n_critical": 0,
            "pct_critical": 0.0,
            "avg_score": 0.0,
            "is_critical": False,
            "critical_residues": []
        }
    crit = cav[cav["critical_score"] >= threshold]
    n_crit = len(crit)
    pct = 100.0 * n_crit / n_total
    avg = float(cav["critical_score"].mean())
    is_crit = (pct >= 50.0)
    if verbose:
        print(f"    total={n_total} | críticos={n_crit} ({pct:.1f}%) | avg={avg:.3f} | ¿crítica? {'SÍ' if is_crit else 'NO'}")
    return {
        "n_total": int(n_total),
        "n_critical": int(n_crit),
        "pct_critical": float(pct),
        "avg_score": avg,
        "is_critical": is_crit,
        "critical_residues": [int(x) for x in crit["resid"].tolist()]
    }


def _summary_text(
    project_name: str,
    integrated_df: pd.DataFrame,
    min_score: float,
    cavity_info: Dict[str, Any] | None
) -> str:
    crit = integrated_df[integrated_df["critical_score"] >= min_score]
    lines = []
    lines.append("="*80)
    lines.append("RESUMEN DE ANÁLISIS ALOSTÉRICO")
    lines.append("="*80 + "\n")
    lines.append(f"Proyecto: {project_name}\n")
    lines.append("RESULTADOS GENERALES")
    lines.append("-"*80)
    lines.append(f"Total residuos analizados: {len(integrated_df)}")
    lines.append(f"Residuos críticos identificados (≥ {min_score}): {len(crit)}")
    lines.append(f"Porcentaje crítico: {100.0*len(crit)/len(integrated_df):.1f}%\n")

    lines.append("TOP 20 RESIDUOS MÁS CRÍTICOS")
    lines.append("-"*80)
    cols = ["resid","resname","segid","protein","critical_score"]
    show = crit[cols].head(20) if not crit.empty else integrated_df[cols].head(20)
    lines.append(show.to_string(index=False))

    if cavity_info is not None:
        lines.append("\n")
        lines.append("IMPORTANCIA DE CAVIDAD ALOSTÉRICA")
        lines.append("-"*80)
        lines.append(f"Total residuos en cavidad: {cavity_info['n_total']}")
        lines.append(f"Residuos críticos en cavidad: {cavity_info['n_critical']} ({cavity_info['pct_critical']:.1f}%)")
        lines.append(f"Score promedio cavidad: {cavity_info['avg_score']:.3f}")
        lines.append(f"¿La cavidad es crítica?: {'✅ SÍ' if cavity_info['is_critical'] else '❌ NO'}\n")

        # distribución por proteína (si existe 'protein')
        if "protein" in integrated_df.columns:
            nrp1 = crit[crit["protein"]=="NRP1"]
            xylt1 = crit[crit["protein"]=="XYLT1"]
            lines.append("DISTRIBUCIÓN POR PROTEÍNA")
            lines.append("-"*80)
            lines.append(f"NRP1: {len(nrp1)} residuos críticos")
            lines.append(f"XYLT1: {len(xylt1)} residuos críticos")
            lines.append("")

    return "\n".join(lines)


# ------------------------------- runner YAML ---------------------------------

def run_integration_from_yaml(yaml_path: Path):
    cfg = _read_yaml(yaml_path)
    verbose = bool(cfg.get("run", {}).get("verbose", True))
    project = cfg.get("project_name","project")
    outroot = Path(cfg["run"]["outroot_results"])
    outdir = Path(cfg["output"]["base_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("="*80)
        print("MÓDULO 7: INTEGRATION")
        print("="*80)

    # Cargar artefactos
    art = cfg["artifacts"]
    p_net  = Path(art["network_ranking"])
    p_eng  = Path(art["energy_binding_per_res"])
    p_dccm = Path(art["dccm_high_correlations"])
    p_pca  = Path(art["pca_participation"])
    p_path = Path(art["pathways_bottlenecks"])

    net_df  = integrate_network_from_csv(p_net,  verbose=verbose)
    eng_df  = integrate_energy_from_csv(p_eng,   verbose=verbose)
    dcc_df  = integrate_dccm_from_csv(p_dccm,    verbose=verbose)
    pca_df  = integrate_pca_from_csv(p_pca,      n_components=3, verbose=verbose)
    path_df = integrate_pathways_from_csv(p_path, verbose=verbose)

    weights = dict(cfg["integration"]["weights"])
    min_crit = float(cfg["integration"]["min_critical_score"])

    integrated = _merge_scores(net_df, eng_df, dcc_df, pca_df, path_df, weights, verbose=verbose)
    crit_df = integrated[integrated["critical_score"] >= min_crit].copy()
    if verbose:
        print(f"\n  Residuos críticos (score ≥ {min_crit}): {len(crit_df)}")

    cav_info = None
    cav_cfg = cfg["integration"]["cavity_check"]
    if bool(cav_cfg.get("enabled", False)):
        cav_res = list(cav_cfg.get("cavity_residues", []))
        cav_thr = float(cav_cfg.get("threshold", 0.6))
        cav_info = _cavity_check(integrated, cav_res, cav_thr, verbose=verbose)

    # Guardar salidas
    out_cfg = cfg["integration"]["output"]

    p_crit = outdir / out_cfg["critical_residues"]
    _ensure_parent(p_crit)
    integrated.to_csv(p_crit, index=False)

    if cav_info is not None:
        p_cav = outdir / out_cfg["cavity_importance"]
        _ensure_parent(p_cav)
        with p_cav.open("w", encoding="utf-8") as f:
            json.dump(cav_info, f, indent=2)

    summary = _summary_text(project, integrated, min_crit, cav_info)
    p_sum = outdir / out_cfg["summary"]
    _ensure_parent(p_sum)
    p_sum.write_text(summary, encoding="utf-8")

    if verbose:
        print(f"\n✅ Guardado:")
        print(f"  - {p_crit}")
        if cav_info is not None:
            print(f"  - {p_cav}")
        print(f"  - {p_sum}")

    return {
        "integrated": integrated,
        "critical": crit_df,
        "cavity": cav_info,
        "summary_path": str(p_sum),
    }

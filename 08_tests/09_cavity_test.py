# 08_tests/09_cavity_test.py
from __future__ import annotations
import os, sys, json, pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml
import MDAnalysis as mda

# Ajusta el PYTHONPATH para importar el core
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "01_src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cavity_analysis.core.cavity import run_cavity_allostery_analysis


def _csv(p: Path):
    return pd.read_csv(p) if p.exists() else None

def _json(p: Path):
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _npy(p: Path):
    return np.load(p) if p.exists() else None

def _pkl(p: Path):
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def load_previous_results_from_yaml(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruye el diccionario all_results leyendo de disco según cfg['previous_results']."""
    prev = cfg.get("previous_results", {})
    out: Dict[str, Any] = {}

    # --- NETWORK (módulo 1)
    net = prev.get("network", {})
    ndir = Path(net.get("dir", ""))
    G    = _pkl(ndir / net.get("graph_pkl", ""))
    rank = _csv(ndir / net.get("ranking_csv", ""))
    if G is not None and rank is not None:
        out["network"] = {"network": G, "ranking": rank}

    # --- ENERGY (módulo 2)
    en   = prev.get("energy", {})
    edir = Path(en.get("dir", ""))
    inter = _csv(edir / en.get("inter_energy_csv", ""))
    bind  = _csv(edir / en.get("binding_per_residue_csv", ""))
    if inter is not None or bind is not None:
        out["energy"] = {}
        if inter is not None: out["energy"]["inter"] = inter
        if bind  is not None: out["energy"]["binding_per_residue"] = bind

    # --- DCCM (módulo 3)
    dccm = prev.get("dccm", {})
    ddir = Path(dccm.get("dir", ""))
    M    = _npy(ddir / dccm.get("dccm_npy", ""))
    high = _csv(ddir / dccm.get("high_corr_csv", ""))
    if M is not None or high is not None:
        out["dccm"] = {}
        if M is not None:    out["dccm"]["dccm"] = M
        if high is not None: out["dccm"]["high_correlations"] = high

    # --- PCA (módulo 4)
    pca  = prev.get("pca", {})
    pdir = Path(pca.get("dir", ""))
    part = _csv(pdir / pca.get("participation_csv", ""))
    if part is not None:
        out["pca"] = {"participation": part}

    # --- PATHWAYS (módulo 5)
    pw   = prev.get("pathways", {})
    wdir = Path(pw.get("dir", ""))
    paths = _json(wdir / pw.get("pathways_json", ""))
    bott  = _csv(wdir / pw.get("bottlenecks_csv", ""))
    if paths is not None or bott is not None:
        out["pathways"] = {}
        if paths is not None: out["pathways"]["pathways"] = paths
        if bott  is not None: out["pathways"]["bottlenecks"] = bott

    # --- COMMUNITIES (módulo 6)
    com   = prev.get("communities", {})
    cdir  = Path(com.get("dir", ""))
    cjson = _json(cdir / com.get("communities_json", ""))
    if cjson is not None:
        # el core espera {"communities": assignments}
        assign = cjson.get("assignments", cjson)
        out["communities"] = {"communities": assign}

    # --- INTEGRATION (módulo 7)
    integ = prev.get("integration", {})
    idir  = Path(integ.get("dir", ""))
    crit  = _csv(idir / integ.get("critical_csv", ""))
    if crit is not None:
        out["integration"] = {"integrated": crit}

    return out


def main():
    # YAML por defecto
    yaml_path = ROOT / "03_configs" / "analyses" / "09_cavity.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No encuentro el YAML de módulo 9: {yaml_path}")

    cfg = yaml.safe_load(open(yaml_path, "r", encoding="utf-8"))
    verbose = bool(cfg.get("run", {}).get("verbose", True))

    # PDB/DCD de inputs
    pdb = cfg["inputs"]["pdb"]
    dcd = cfg["inputs"].get("dcd", None)

    print(f"[INFO] PDB: {pdb}")
    if dcd:
        print(f"[INFO] DCD: {dcd}")

    # Universe
    if dcd and Path(dcd).exists():
        u = mda.Universe(pdb, dcd)
    else:
        u = mda.Universe(pdb)

    # Cargar resultados previos
    print("[INFO] Cargando resultados previos (módulos 1–7)…")
    prev = load_previous_results_from_yaml(cfg)

    # Validaciones mínimas
    required_keys = ["network", "integration"]  # lo mínimo para un veredicto coherente
    for k in required_keys:
        if k not in prev:
            print(f"[WARN] Falta '{k}' en previous_results; el análisis seguirá pero con menos señales.")

    # Ejecutar core
    print("=" * 80)
    print("TEST MÓDULO 9 — CAVITY ALLOSTERY")
    print("=" * 80)
    results = run_cavity_allostery_analysis(u, cfg, prev, verbose=True)

    # Resumen rápido
    print("\n" + "-" * 80)
    print("RESUMEN DEL TEST")
    print("-" * 80)
    if results and "verdict" in results:
        print(f"Veredicto: {results['verdict']['verdict']} | Score: {results['verdict']['score']:.1f}/100")
    else:
        print("No se obtuvo veredicto (results vacío).")

    # Chequeo de outputs esperados en disco
    out_base = Path(cfg["output"]["base_dir"])
    expected = [
        out_base / "tables" / "9_cavity_residues_data.csv",
        out_base / "tables" / "9_cavity_comparison.json",
        out_base / "tables" / "9_cavity_verdict.json",
        out_base / "tables" / "9_cavity_allostery_report.txt",
    ]
    print("\nArchivos esperados:")
    for p in expected:
        print(f" - {p}  [{'OK' if p.exists() else 'FALTA'}]")

    # Top 10 de cavidad (si disponible)
    if results and "cavity_data" in results:
        cd = results["cavity_data"]
        cols = [c for c in ["resid", "resname", "critical_score", "betweenness"] if c in cd.columns]
        if not cd.empty and "critical_score" in cd.columns:
            top10 = cd.sort_values("critical_score", ascending=False).head(10)[cols]
            print("\nTop 10 residuos de cavidad por critical_score:")
            print(top10.to_string(index=False))
        else:
            print("\nNo hay columna 'critical_score' en cavity_data para mostrar top10.")
    else:
        print("\nNo se pudo cargar 'cavity_data' del resultado.")


if __name__ == "__main__":
    main()

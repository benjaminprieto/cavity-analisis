# 01_src/cavity_analysis/core/energy.py
"""
Módulo 2: Descomposición de energía
- Calcula energías de interacción intra e inter proteína (modelo simplificado)
- Guarda: intra_nrp1_matrix.csv, intra_xylt1_matrix.csv, inter_nrp1_xylt1.csv, binding_per_residue.csv

Requiere:
  - MDAnalysis, numpy, pandas, pyyaml, tqdm (opcional)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import os
import json

import yaml
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import distances

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Constantes (modelo simple)
COULOMB_CONSTANT = 332.0636  # kcal·Å/(mol·e²)
EPSILON_0 = 1.0              # no usado; epsilon efectivo se pasa como arg


# ----------------------------- utilidades IO/CFG -----------------------------

def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------- selección de frames ---------------------------

def _select_frames(u, frames_cfg: Dict[str, Any], verbose: bool = True) -> List[int]:
    """
    frames_cfg:
      mode: "stride" | "range" | "list"
      stride: int
      start: int
      stop: int|null
      list: [ints]
    """
    n = len(u.trajectory)
    mode = (frames_cfg or {}).get("mode", "stride")

    if mode == "list":
        lst = frames_cfg.get("list") or []
        frames = [int(f) for f in lst if 0 <= int(f) < n]

    elif mode == "range":
        start = int(frames_cfg.get("start", 0) or 0)
        stop  = frames_cfg.get("stop", None)
        stop  = int(stop) if stop is not None else n
        stride = int(frames_cfg.get("stride", 1) or 1)
        start = max(0, start); stop = min(n, stop)
        frames = list(range(start, stop, max(1, stride)))

    else:  # stride
        stride = int(frames_cfg.get("stride", 1) or 1)
        start = int(frames_cfg.get("start", 0) or 0)
        stop  = frames_cfg.get("stop", None)
        stop  = int(stop) if stop is not None else n
        start = max(0, start); stop = min(n, stop)
        frames = list(range(start, stop, max(1, stride)))

    if verbose:
        print(f"[INFO] Frames seleccionados: {len(frames)} (mode={mode})")
    return frames


# ----------------------------- física simplificada --------------------------

def estimate_charge(atom) -> float:
    """
    Estimación burda de carga parcial por tipo de átomo.
    """
    name = atom.name.upper()
    if name.startswith("N"):
        return -0.5
    if name.startswith("O"):
        return -0.5
    if name.startswith("C"):
        return 0.1
    if name.startswith("H"):
        return 0.3
    return 0.0


def calculate_coulomb_energy(pos1: np.ndarray, charges1: np.ndarray,
                             pos2: np.ndarray, charges2: np.ndarray,
                             epsilon: float = 4.0) -> float:
    """
    Energía Coulomb (simplificada): E = k * q1*q2 / (epsilon * r)
    Devuelve la suma de todas las parejas átomo-átomo.
    """
    dist = distances.distance_array(pos1, pos2)
    dist = np.maximum(dist, 0.1)  # evita singularidad
    charge_matrix = np.outer(charges1, charges2)
    energy_matrix = COULOMB_CONSTANT * charge_matrix / (epsilon * dist)
    return float(energy_matrix.sum())


def calculate_lj_energy(pos1: np.ndarray, pos2: np.ndarray,
                        sigma: float = 3.5, epsilon_lj: float = 0.1) -> float:
    """
    Lennard-Jones (12-6) simplificado:
      E = 4 * epsilon * [ (sigma/r)^12 - (sigma/r)^6 ]
    """
    dist = distances.distance_array(pos1, pos2)
    dist = np.maximum(dist, 0.5)  # evita explosiones
    sigma_r = sigma / dist
    sigma_r6 = sigma_r ** 6
    sigma_r12 = sigma_r6 ** 2
    energy_matrix = 4.0 * epsilon_lj * (sigma_r12 - sigma_r6)

    # capping de rango
    energy_matrix[dist > 12.0] = 0.0
    return float(energy_matrix.sum())


def _residue_pair_energy(res1, res2,
                         eps_coulomb: float = 4.0,
                         sigma_lj: float = 3.5,
                         eps_lj: float = 0.1) -> Dict[str, float]:
    """
    Energía de par de residuos (res1-res2).
    """
    # posiciones
    p1 = res1.atoms.positions
    p2 = res2.atoms.positions
    # cargas aproximadas
    q1 = np.array([estimate_charge(a) for a in res1.atoms], dtype=float)
    q2 = np.array([estimate_charge(a) for a in res2.atoms], dtype=float)

    e_coul = calculate_coulomb_energy(p1, q1, p2, q2, epsilon=eps_coulomb)
    e_lj   = calculate_lj_energy(p1, p2, sigma=sigma_lj, epsilon_lj=eps_lj)
    return dict(e_coulomb=e_coul, e_lj=e_lj, e_total=e_coul + e_lj)


def calculate_residue_residue_energy(u, frames: List[int],
                                     selection1: str, selection2: str,
                                     cutoff: float = 12.0,
                                     eps_coulomb: float = 4.0,
                                     sigma_lj: float = 3.5,
                                     eps_lj: float = 0.1,
                                     verbose: bool = True) -> pd.DataFrame:
    """
    Calcula energía res–res promedio en frames, entre selection1 y selection2.
    - Primero filtra por distancia de COM de residuos (cutoff) para acelerar.
    """
    atoms1 = u.select_atoms(selection1)
    atoms2 = u.select_atoms(selection2)
    if atoms1.n_atoms == 0 or atoms2.n_atoms == 0:
        raise ValueError(f"Selecciones vacías: '{selection1}' vs '{selection2}'")

    residues1 = atoms1.residues
    residues2 = atoms2.residues

    acc: Dict[tuple, Dict[str, float]] = {}
    it = tqdm(frames, desc="Calculando energías") if (verbose and tqdm) else frames

    for fi in it:
        u.trajectory[fi]

        # precomputar COMs (vectorizados) para cada frame
        coms1 = np.array([r.atoms.center_of_mass() for r in residues1])
        coms2 = np.array([r.atoms.center_of_mass() for r in residues2])

        # matriz de distancias COM
        dCOM = distances.distance_array(coms1, coms2)
        # pares candidatos por cutoff
        idx1, idx2 = np.where(dCOM <= float(cutoff))

        for i1, i2 in zip(idx1, idx2):
            r1 = residues1[i1]
            r2 = residues2[i2]

            pair_key = (int(r1.resid), int(r2.resid))
            if pair_key not in acc:
                acc[pair_key] = {
                    "resid1": int(r1.resid), "resname1": str(r1.resname), "segid1": str(r1.segid),
                    "resid2": int(r2.resid), "resname2": str(r2.resname), "segid2": str(r2.segid),
                    "e_coulomb": 0.0, "e_lj": 0.0, "count": 0
                }

            e = _residue_pair_energy(r1, r2, eps_coulomb=eps_coulomb,
                                     sigma_lj=sigma_lj, eps_lj=eps_lj)
            acc[pair_key]["e_coulomb"] += e["e_coulomb"]
            acc[pair_key]["e_lj"]      += e["e_lj"]
            acc[pair_key]["count"]     += 1

    # promediar
    rows = []
    for data in acc.values():
        c = max(1, int(data["count"]))
        rows.append({
            **{k: data[k] for k in ["resid1","resname1","segid1","resid2","resname2","segid2"]},
            "e_coulomb": data["e_coulomb"] / c,
            "e_lj": data["e_lj"] / c,
            "e_total": (data["e_coulomb"] + data["e_lj"]) / c
        })

    return pd.DataFrame(rows)


def calculate_per_residue_binding_energy(energy_df: pd.DataFrame,
                                         chain1: str, chain2: str) -> pd.DataFrame:
    """
    Contribución al binding por residuo de chain1 (suma de e_total contra chain2).
    """
    if energy_df.empty:
        return pd.DataFrame(columns=["segid","resid","resname","binding_energy","e_coulomb","e_lj"])

    g = energy_df.groupby(["segid1","resid1","resname1"], as_index=False).agg(
        binding_energy=("e_total","sum"),
        e_coulomb=("e_coulomb","sum"),
        e_lj=("e_lj","sum")
    )
    g = g.rename(columns={"segid1":"segid","resid1":"resid","resname1":"resname"})
    g = g.sort_values("binding_energy")
    return g


# --------------------------------- runner ------------------------------------

def run_energy_decomposition(u, cfg: Dict[str,Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Orquesta el análisis según YAML y guarda los resultados.
    """
    if verbose:
        print("=" * 80)
        print("MÓDULO 2: ENERGY DECOMPOSITION")
        print("=" * 80)

    ed = cfg["energy_decomposition"]
    frames = _select_frames(u, cfg.get("frames", {}), verbose=verbose)
    if verbose:
        print(f"\nFrames a analizar: {len(frames)}")

    outdir = Path(cfg["output"]["base_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}

    # (1) intra NRP1
    if ed["intra_nrp1"]["enabled"]:
        if verbose: print("\n1) Energías intra-NRP1...")
        sel = ed["intra_nrp1"]["selection"]
        cutoff = float(ed["intra_nrp1"]["cutoff"])
        df = calculate_residue_residue_energy(u, frames, sel, sel, cutoff, verbose=verbose)
        results["intra_nrp1"] = df
        p = outdir / ed["output"]["intra_nrp1_matrix"]
        _ensure_parent(p); df.to_csv(p, index=False)
        if verbose: print(f"   ✅ {p}")

    # (2) intra XYLT1
    if ed["intra_xylt1"]["enabled"]:
        if verbose: print("\n2) Energías intra-XYLT1...")
        sel = ed["intra_xylt1"]["selection"]
        cutoff = float(ed["intra_xylt1"]["cutoff"])
        df = calculate_residue_residue_energy(u, frames, sel, sel, cutoff, verbose=verbose)
        results["intra_xylt1"] = df
        p = outdir / ed["output"]["intra_xylt1_matrix"]
        _ensure_parent(p); df.to_csv(p, index=False)
        if verbose: print(f"   ✅ {p}")

    # (3) inter NRP1 <-> XYLT1
    if ed["inter_nrp1_xylt1"]["enabled"]:
        if verbose: print("\n3) Energías inter NRP1-XYLT1...")
        sel1 = ed["inter_nrp1_xylt1"]["selection1"]
        sel2 = ed["inter_nrp1_xylt1"]["selection2"]
        cutoff = float(ed["inter_nrp1_xylt1"]["cutoff"])
        df_inter = calculate_residue_residue_energy(u, frames, sel1, sel2, cutoff, verbose=verbose)
        results["inter"] = df_inter
        p = outdir / ed["output"]["inter_energy"]
        _ensure_parent(p); df_inter.to_csv(p, index=False)
        if verbose: print(f"   ✅ {p}")

        if verbose: print("\n4) Contribución por residuo al binding (chain1 vs chain2)...")
        chain_nrp1 = cfg["inputs"]["chains"]["nrp1"]
        chain_xylt1 = cfg["inputs"]["chains"]["xylt1"]
        df_bind = calculate_per_residue_binding_energy(df_inter, chain_nrp1, chain_xylt1)
        results["binding_per_residue"] = df_bind
        p = outdir / ed["output"]["binding_energy_per_residue"]
        _ensure_parent(p); df_bind.to_csv(p, index=False)
        if verbose:
            print(f"   ✅ {p}")
            print("\n   Top 10 contribuyentes (más favorables):")
            cols = ["segid","resid","resname","binding_energy"]
            print(df_bind.head(10)[cols].to_string(index=False))

    return results


def run_energy_from_yaml(yaml_path: Path):
    """
    Entrada cómoda: lee YAML, carga Universe y ejecuta.
    """
    cfg = _read_yaml(yaml_path)
    verbose = bool(cfg.get("run",{}).get("verbose", True))

    pdb = Path(cfg["inputs"]["pdb"])
    dcd = Path(cfg["inputs"]["dcd"])
    if verbose:
        print(f"[INFO] Cargando PDB: {pdb}")
        print(f"[INFO] Cargando DCD: {dcd}")
    if not pdb.exists():
        raise FileNotFoundError(f"No existe PDB: {pdb}")
    if not dcd.exists():
        raise FileNotFoundError(f"No existe DCD: {dcd}")

    u = mda.Universe(str(pdb), str(dcd))
    Path(cfg["output"]["base_dir"]).mkdir(parents=True, exist_ok=True)
    return run_energy_decomposition(u, cfg, verbose=verbose)

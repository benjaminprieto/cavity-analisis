# 01_src/cavity_analysis/core/dccm.py
"""
Módulo 3: DCCM (Dynamic Cross-Correlation Matrix)
- Carga PDB+DCD (MDAnalysis)
- Calcula DCCM de una selección (por defecto: Cα proteína)
- Opcional: extrae correlaciones cruzadas A↔B
- Identifica pares con |r| >= umbral
- Guarda:
    * dccm_matrix.npy
    * dccm_cross_AB.csv
    * dccm_high_pairs.csv
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml
import MDAnalysis as mda

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ------------------------------ utilidades IO --------------------------------

def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------- selección de frames ----------------------------

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


# -------------------------------- DCCM core ----------------------------------

def calculate_dccm(u, frames: List[int], selection: str, verbose: bool = True):
    """
    Calcula DCCM vectorizada para una selección (ej. 'protein and name CA').

    Devuelve:
      - dccm: np.ndarray (N, N) con correlaciones en [-1, 1]
      - atoms: MDAnalysis.AtomGroup (la selección usada)
    """
    atoms = u.select_atoms(selection)
    N = atoms.n_atoms
    if N == 0:
        raise ValueError(f"La selección '{selection}' no devolvió átomos.")
    if verbose:
        print(f"  Calculando DCCM para {N} átomos...")

    # Recolectar posiciones (F, N, 3)
    pos = []
    it = tqdm(frames, desc="Recopilando posiciones") if (verbose and tqdm) else frames
    for fi in it:
        u.trajectory[fi]
        pos.append(atoms.positions.copy())
    X = np.asarray(pos, dtype=np.float64)  # (F, N, 3)

    # Fluctuaciones: Δr = r - <r>
    mean_pos = X.mean(axis=0, keepdims=True)  # (1, N, 3)
    dX = X - mean_pos                          # (F, N, 3)

    # Numerador: <Δri · Δrj> = sum_t sum_comp dX[t,i,c]*dX[t,j,c] / F
    # tensordot sobre tiempo y componente (ejes [0,2])
    # Resultado (N,N)
    num = np.tensordot(dX, dX, axes=([0, 2], [0, 2])) / float(dX.shape[0])

    # Denominador: sqrt(<Δri^2><Δrj^2>), donde <Δri^2> = sum_t sum_c dX[t,i,c]^2 / F
    var = (dX ** 2).sum(axis=(0, 2)) / float(dX.shape[0])  # (N,)
    denom = np.sqrt(np.outer(var, var))                    # (N,N)
    # Evita /0
    denom = np.where(denom > 0, denom, 1.0)
    C = num / denom

    # clip numérico
    C = np.clip(C, -1.0, 1.0)
    return C, atoms


def extract_cross_protein_correlations(dccm: np.ndarray, atoms, chain1: str, chain2: str) -> pd.DataFrame:
    """
    Extrae todos los pares (i in chain1, j in chain2) con su r.
    """
    idx1 = [i for i, a in enumerate(atoms) if str(a.segid) == str(chain1)]
    idx2 = [i for i, a in enumerate(atoms) if str(a.segid) == str(chain2)]

    sub = dccm[np.ix_(idx1, idx2)]

    rows = []
    for i_local, i_glob in enumerate(idx1):
        a1 = atoms[i_glob]
        for j_local, j_glob in enumerate(idx2):
            a2 = atoms[j_glob]
            rows.append(dict(
                resid1=int(a1.resid), resname1=str(a1.resname), segid1=str(a1.segid),
                resid2=int(a2.resid), resname2=str(a2.resname), segid2=str(a2.segid),
                correlation=float(sub[i_local, j_local])
            ))
    return pd.DataFrame(rows)


def identify_high_correlations(dccm: np.ndarray, atoms, threshold: float = 0.5) -> pd.DataFrame:
    """
    Pairs con |r| >= threshold. Solo i<j para evitar duplicados.
    """
    N = dccm.shape[0]
    rows = []
    for i in range(N):
        for j in range(i + 1, N):
            r = float(dccm[i, j])
            if abs(r) >= float(threshold):
                a = atoms[i]; b = atoms[j]
                rows.append(dict(
                    resid1=int(a.resid), resname1=str(a.resname), segid1=str(a.segid),
                    resid2=int(b.resid), resname2=str(b.resname), segid2=str(b.segid),
                    correlation=r,
                    type=("positive" if r > 0 else "negative")
                ))
    if rows:
        df = pd.DataFrame(rows)
        # ordenar por |r| descendente
        df = df.reindex(df["correlation"].abs().sort_values(ascending=False).index)
        return df
    return pd.DataFrame(columns=["resid1","resname1","segid1","resid2","resname2","segid2","correlation","type"])


# -------------------------------- orquestador --------------------------------

def run_dccm(u, cfg: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print("=" * 80)
        print("MÓDULO 3: DYNAMIC CROSS-CORRELATION MATRIX (DCCM)")
        print("=" * 80)

    dconf = cfg["dccm"]
    selection = dconf["selection"]

    frames = _select_frames(u, cfg.get("frames", {}), verbose=verbose)
    if verbose:
        print(f"\nFrames a analizar: {len(frames)}")

    # 1) DCCM
    dccm, atoms = calculate_dccm(u, frames, selection, verbose=verbose)
    if verbose:
        finite = dccm[np.isfinite(dccm)]
        vmax = finite[finite < 1].max() if finite.size else np.nan
        vmin = finite.min() if finite.size else np.nan
        print(f"\nDCCM calculada: {dccm.shape}")
        print(f"  Correlación máxima (excl. diag): {vmax:.3f}")
        print(f"  Correlación mínima: {vmin:.3f}")

    outdir = Path(cfg["output"]["base_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    # guardar matriz completa
    dccm_path = outdir / dconf["output"]["dccm_full"]
    _ensure_parent(dccm_path)
    # fuerza extensión .npy si falta
    if dccm_path.suffix.lower() != ".npy":
        dccm_path = dccm_path.with_suffix(".npy")
    np.save(dccm_path, dccm)
    if verbose:
        print(f"\n✅ DCCM guardada: {dccm_path}")

    results = {"dccm_path": str(dccm_path)}

    # 2) Matriz cruzada A↔B (si habilitado)
    if dconf.get("cross_protein", {}).get("enabled", False):
        if verbose:
            print("\nExtrayendo correlaciones cruzadas NRP1 (A) ↔ XYLT1 (B)...")
        chainA = cfg["inputs"]["chains"]["nrp1"]  # "A"
        chainB = cfg["inputs"]["chains"]["xylt1"] # "B"
        df_cross = extract_cross_protein_correlations(dccm, atoms, chainA, chainB)
        cross_path = outdir / dconf["output"]["dccm_cross"]
        _ensure_parent(cross_path)
        df_cross.to_csv(cross_path, index=False)
        results["dccm_cross"] = str(cross_path)
        if verbose:
            print(f"  ✅ Guardado: {cross_path} (pares: {len(df_cross)})")

    # 3) Pares con |r| >= umbral
    thr = float(dconf["thresholds"]["significant_correlation"])
    if verbose:
        print(f"\nIdentificando correlaciones |r| >= {thr}...")
    df_high = identify_high_correlations(dccm, atoms, thr)
    high_path = outdir / dconf["output"]["high_correlations"]
    _ensure_parent(high_path)
    df_high.to_csv(high_path, index=False)
    results["dccm_high_pairs"] = str(high_path)
    if verbose:
        print(f"  ✅ Guardado: {high_path} (pares: {len(df_high)})")

    return results


def run_dccm_from_yaml(yaml_path: Path):
    cfg = _read_yaml(yaml_path)
    verbose = bool(cfg.get("run", {}).get("verbose", True))

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
    return run_dccm(u, cfg, verbose=verbose)

# 01_src/cavity_analysis/core/pca.py
"""
Módulo 4: Principal Component Analysis (PCA)
- Carga PDB+DCD y selecciona frames
- Calcula PCA (sklearn) sobre posiciones (aplanadas)
- Guarda:
    * components.npy
    * variance_explained.csv
    * residue_participation.csv
    * pc_projection.csv
- (Opcional) resumen por regiones si están definidas en el YAML
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml
import MDAnalysis as mda
from sklearn.decomposition import PCA

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


# --------------------------------- PCA core ----------------------------------

def calculate_pca(u, frames: List[int], selection: str,
                  n_components: int = 10, verbose: bool = True) -> Dict[str, Any]:
    """
    Devuelve dict con:
      pca (objeto sklearn), projection (F x nPC), variance_explained, cumulative_variance,
      atoms (selección), n_components
    """
    atoms = u.select_atoms(selection)
    N = atoms.n_atoms
    if N == 0:
        raise ValueError(f"La selección '{selection}' no devolvió átomos.")
    if verbose:
        print(f"  Calculando PCA para {N} átomos...")

    # Recolectar posiciones (F, 3N)
    rows = []
    it = tqdm(frames, desc="Recopilando posiciones") if (verbose and tqdm) else frames
    for fi in it:
        u.trajectory[fi]
        rows.append(atoms.positions.astype(np.float64).reshape(-1))  # (3N,)
    X = np.asarray(rows, dtype=np.float64)  # (F, 3N)
    if verbose:
        print(f"  Matriz de datos: {X.shape}")

    # Centrado automático lo hace PCA de sklearn
    if verbose:
        print(f"  Calculando {n_components} componentes principales...")
    pca = PCA(n_components=n_components)
    projection = pca.fit_transform(X)  # (F, nPC)

    var = pca.explained_variance_ratio_
    cum = np.cumsum(var)

    if verbose:
        print("\n  Varianza explicada por PC:")
        top = min(5, n_components)
        for i in range(top):
            print(f"    PC{i+1}: {var[i]*100:.2f}%")
        print(f"    PC1-{n_components}: {cum[-1]*100:.2f}%")

    return dict(
        pca=pca,
        projection=projection,
        variance_explained=var,
        cumulative_variance=cum,
        atoms=atoms,
        n_components=n_components
    )


def _selection_index_map(atoms) -> Dict[int, int]:
    """
    Mapea atom.index (global en Universe) -> posición dentro de 'atoms' (0..Nsel-1).
    Robusto y O(N).
    """
    return {a.index: i for i, a in enumerate(atoms)}


def calculate_residue_participation(pca_results: Dict[str, Any], verbose: bool = True) -> pd.DataFrame:
    """
    Participación por residuo: norma L2 de los loadings (componentes) de sus átomos.
    Normaliza cada PC a [0,1] para facilitar ranking comparativo por residuo.
    """
    pca = pca_results["pca"]
    atoms = pca_results["atoms"]
    nPC = int(pca_results["n_components"])
    comps = pca.components_  # (nPC, 3Nsel)

    # mapa rápido atom.index -> idx en selección
    sel_map = _selection_index_map(atoms)

    rows = []
    for res in atoms.residues:
        # índices (en la selección) de los átomos del residuo
        idxs = []
        for a in res.atoms:
            j = sel_map.get(a.index, None)
            if j is not None:
                idxs.append(j)
        if not idxs:
            continue

        row = dict(resid=int(res.resid), resname=str(res.resname),
                   segid=str(res.segid),
                   protein=("NRP1" if str(res.segid) == "A" else ("XYLT1" if str(res.segid) == "B" else "OTHER")))

        # magnitud por PC = norma L2 de los 3*#átomos loadings
        for k in range(nPC):
            # recoger los 3 componentes por átomo
            vals = []
            for j in idxs:
                j3 = 3*j
                vals.extend([comps[k, j3], comps[k, j3+1], comps[k, j3+2]])
            row[f"PC{k+1}"] = float(np.linalg.norm(vals))
        rows.append(row)

    df = pd.DataFrame(rows)
    # normalizar cada PC a [0,1]
    for k in range(nPC):
        col = f"PC{k+1}"
        if col in df.columns and df[col].max() > 0:
            df[col] = df[col] / df[col].max()
    return df


def analyze_region_participation(participation_df: pd.DataFrame,
                                 region_cfg: Dict[str, Any],
                                 full_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resume participación promedio por región definida en full_cfg['regions'].
    """
    out = {}
    region_names = (region_cfg or {}).get("regions", []) or []
    for rname in region_names:
        rdef = (full_cfg.get("regions") or {}).get(rname, None)
        if not rdef:
            continue
        residues = rdef.get("residues", [])
        sub = participation_df[participation_df["resid"].isin(residues)]
        pc_cols = [c for c in participation_df.columns if c.startswith("PC")]
        avg = sub[pc_cols].mean().to_dict() if not sub.empty else {c: 0.0 for c in pc_cols}
        out[rname] = dict(n_residues=int(len(sub)), avg_participation={k: float(v) for k, v in avg.items()})
    return out


# -------------------------------- orquestador --------------------------------

def run_pca(u, cfg: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print("=" * 80)
        print("MÓDULO 4: PRINCIPAL COMPONENT ANALYSIS (PCA)")
        print("=" * 80)

    pcfg = cfg["pca"]
    frames = _select_frames(u, cfg.get("frames", {}), verbose=verbose)
    if verbose:
        print(f"\nFrames a analizar: {len(frames)}")

    # 1) PCA
    results = calculate_pca(
        u, frames,
        pcfg["selection"],
        int(pcfg["n_components"]),
        verbose=verbose
    )

    outdir = Path(cfg["output"]["base_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    # 2) Guardar componentes (eigenvectores)
    comp_path = outdir / pcfg["output"]["components"]
    if comp_path.suffix.lower() != ".npy":
        comp_path = comp_path.with_suffix(".npy")
    _ensure_parent(comp_path)
    np.save(comp_path, results["pca"].components_)

    # 3) Varianza explicada
    var_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(results["n_components"])],
        "variance_explained": results["variance_explained"],
        "cumulative_variance": results["cumulative_variance"]
    })
    var_path = outdir / pcfg["output"]["variance_explained"]
    _ensure_parent(var_path)
    var_df.to_csv(var_path, index=False)
    if verbose:
        print(f"\n✅ Varianza guardada: {var_path}")

    # 4) Participación por residuo
    if verbose:
        print("\nCalculando participación de residuos...")
    part_df = calculate_residue_participation(results, verbose=verbose)
    part_path = outdir / pcfg["output"]["participation"]
    _ensure_parent(part_path)
    part_df.to_csv(part_path, index=False)
    if verbose:
        print(f"  ✅ Guardado: {part_path}")

    # 5) Regiones (opcional)
    region_summary = None
    if pcfg.get("region_participation", {}).get("enabled", False):
        if verbose:
            print("\nAnalizando participación de regiones...")
        region_summary = analyze_region_participation(
            part_df, pcfg.get("region_participation", {}), cfg
        )
        if verbose:
            for rname, data in region_summary.items():
                print(f"\n  {rname}:")
                print(f"    Residuos: {data['n_residues']}")
                # muestra los 3 primeros PCs
                for pc, val in list(data["avg_participation"].items())[:3]:
                    print(f"    {pc}: {val:.3f}")

    # 6) Proyección de la trayectoria
    proj_df = pd.DataFrame(
        results["projection"],
        columns=[f"PC{i+1}" for i in range(results["n_components"])]
    )
    proj_df["frame"] = frames
    proj_path = outdir / pcfg["output"]["projection"]
    _ensure_parent(proj_path)
    proj_df.to_csv(proj_path, index=False)

    return dict(
        components_path=str(comp_path),
        variance_path=str(var_path),
        participation_path=str(part_path),
        projection_path=str(proj_path),
        region_summary=region_summary
    )


def run_pca_from_yaml(yaml_path: Path):
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
    return run_pca(u, cfg, verbose=verbose)

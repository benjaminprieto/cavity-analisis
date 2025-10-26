# 01_src/cavity_analysis/core/ser612.py
from __future__ import annotations
import os, json, tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from tqdm import tqdm


# -----------------------------
# select_frames (autocontenido)
# -----------------------------
def select_frames(u: mda.Universe, config: Dict) -> List[int]:
    fr = config.get("frames", {}) or {}
    mode = fr.get("mode", "stride")
    n_tot = len(u.trajectory)

    if mode == "stride":
        stride = int(fr.get("stride", 1))
        start  = int(fr.get("start", 0))
        stop   = fr.get("stop", None)
        stop   = int(stop) if stop is not None else n_tot
        start  = max(0, start); stop = min(n_tot, stop)
        return list(range(start, stop, max(1, stride)))

    if mode == "range":
        start = int(fr.get("start", 0))
        stop  = fr.get("stop", None)
        stop  = int(stop) if stop is not None else n_tot
        start = max(0, start); stop = min(n_tot, stop)
        return list(range(start, stop))

    if mode == "list":
        lst = fr.get("list", [])
        if not isinstance(lst, list):
            raise ValueError("frames.list debe ser lista de índices")
        return [int(i) for i in lst if 0 <= int(i) < n_tot]

    raise ValueError(f"frames.mode no soportado: {mode}")


# ---------------------------------------------------------
# 1) Distancia Ser612 ↔ Glu529 (centro de geometría)
# ---------------------------------------------------------
def calculate_ser612_distance_timeseries(
    u: mda.Universe,
    frames: List[int],
    ser612_selection: str,
    glu529_selection: str,
    verbose: bool = True
) -> pd.DataFrame:

    ser612 = u.select_atoms(ser612_selection)
    glu529 = u.select_atoms(glu529_selection)

    if ser612.n_atoms == 0:
        raise ValueError(f"No se encontró Ser612: '{ser612_selection}'")
    if glu529.n_atoms == 0:
        raise ValueError(f"No se encontró Glu529: '{glu529_selection}'")

    if verbose:
        print(f"  Ser612 resid: {int(ser612.resids[0])}")
        print(f"  Glu529 resid: {int(glu529.resids[0])}")

    dists, times = [], []
    iterator = tqdm(frames, desc="Distancias Ser612↔Glu529") if verbose else frames
    for fi in iterator:
        u.trajectory[fi]
        d = np.linalg.norm(ser612.center_of_geometry() - glu529.center_of_geometry())
        dists.append(float(d))
        times.append(float(u.trajectory.time))

    return pd.DataFrame({"frame": frames, "time_ps": times, "distance_A": dists})


# ---------------------------------------------------------
# 2) SASA con prioridad: FreeSASA → MDTraj → Fallback
# ---------------------------------------------------------
def calculate_sasa_freesasa(
    u: mda.Universe,
    frames: List[int],
    ser612_selection: str,
    verbose: bool = True
) -> pd.DataFrame | None:
    """Calcula SASA por-residuo con FreeSASA leyendo un PDB temporal por frame."""
    try:
        import freesasa  # conda-forge: freesasa
    except Exception:
        if verbose:
            print("  ⚠️  FreeSASA no instalado/disponible")
        return None

    ser = u.select_atoms(ser612_selection)
    if ser.n_atoms == 0:
        raise ValueError(f"No se encontró Ser612: '{ser612_selection}'")
    ser_resid = int(ser.resids[0])

    prot = u.select_atoms("protein")
    if prot.n_atoms == 0:
        raise ValueError("La selección 'protein' está vacía.")

    if verbose:
        print("\n  Calculando SASA con FreeSASA…")

    # PDB temporal reutilizable (Windows-friendly)
    tmp_path = None
    values, times = [], []

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="wb") as tmp:
            tmp_path = tmp.name

        iterator = tqdm(frames, desc="SASA FreeSASA") if verbose else frames
        for fi in iterator:
            u.trajectory[fi]
            # Escribir PDB de la proteína del frame actual
            prot.write(tmp_path)  # MDAnalysis escribe coords del frame actual

            # Calcular SASA con FreeSASA
            structure = freesasa.Structure(tmp_path)
            result = freesasa.calc(structure)

            # Buscar SASA del residuo 612 (ignorar la cadena si es necesario)
            ser_area = 0.0
            try:
                res_areas = result.residueAreas()  # dict "Chain:Resnum" -> ResidueAreas
                # intentar matcheo por número de residuo
                wanted = str(ser_resid)
                for key, area in res_areas.items():
                    # key típico: "A:612" (chain:resnum)
                    parts = key.split(":")
                    if len(parts) == 2 and parts[1].strip() == wanted:
                        ser_area += float(area.total)
                # si por alguna razón no sumó nada, fallback por índice
                if ser_area == 0.0:
                    # ordenar por inserción y usar el índice del residuo en MDAnalysis
                    prot_resids = [int(r.resid) for r in prot.residues]
                    if ser_resid in prot_resids:
                        idx = prot_resids.index(ser_resid)
                        ser_area = float(list(res_areas.values())[idx].total)
            except Exception:
                ser_area = 0.0

            values.append(ser_area)
            times.append(float(u.trajectory.time))

        return pd.DataFrame({"frame": frames, "time_ps": times, "sasa_A2": values})

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def calculate_sasa_mdtraj(
    u: mda.Universe,
    frames: List[int],
    ser612_selection: str,
    verbose: bool = True
) -> pd.DataFrame | None:
    """SASA por-residuo con MDTraj (Shrake-Rupley)."""
    try:
        import mdtraj as md
    except Exception:
        if verbose:
            print("  ⚠️  MDTraj no instalado/disponible")
        return None

    ser = u.select_atoms(ser612_selection)
    if ser.n_atoms == 0:
        raise ValueError(f"No se encontró Ser612: '{ser612_selection}'")
    ser_resid = int(ser.resids[0])

    prot = u.select_atoms("protein")
    if prot.n_atoms == 0:
        raise ValueError("La selección 'protein' está vacía.")

    if verbose:
        print("\n  Calculando SASA con MDTraj…")

    # Guardar topología PDB temporal (una vez)
    tmp_top = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="wb") as f:
            tmp_top = f.name
        prot.write(tmp_top)

        # Recolectar posiciones en nm
        xyz_nm = []
        times = []
        for fi in (tqdm(frames, desc="MDTraj XYZ") if verbose else frames):
            u.trajectory[fi]
            xyz_nm.append(prot.positions / 10.0)  # Å → nm
            times.append(float(u.trajectory.time))
        xyz_nm = np.asarray(xyz_nm, dtype=np.float32)  # (n_frames, n_atoms, 3)

        # Cargar topología y setear coords
        traj = md.load(tmp_top)
        if traj.n_atoms != xyz_nm.shape[1]:
            raise ValueError("Desacople n_atoms entre MDTraj y MDAnalysis.")
        traj.xyz = xyz_nm
        # SASA por residuo en nm^2
        sasa_nm2 = md.shrake_rupley(traj, mode="residue")  # shape (nF, nResidues)

        # Buscar índice de Ser612 por número de residuo
        ser_idx = None
        for i, residue in enumerate(traj.topology.residues):
            if residue.resSeq == ser_resid:
                ser_idx = i
                break
        if ser_idx is None:
            raise ValueError(f"No se encontró Ser612 (resid {ser_resid}) en topología MDTraj")

        ser_sasa_A2 = sasa_nm2[:, ser_idx] * 100.0  # nm^2 → Å^2

        return pd.DataFrame({"frame": frames, "time_ps": times, "sasa_A2": ser_sasa_A2})

    except Exception as e:
        if verbose:
            print(f"  ⚠️  Error MDTraj: {e}")
        return None

    finally:
        if tmp_top and os.path.exists(tmp_top):
            try:
                os.unlink(tmp_top)
            except Exception:
                pass


def calculate_accessibility_fallback(
    u: mda.Universe,
    frames: List[int],
    ser612_selection: str,
    cutoff: float = 5.0,
    verbose: bool = True
) -> pd.DataFrame:
    """Proxy de accesibilidad por recuento de vecinos cercanos (< cutoff Å)."""
    if verbose:
        print("\n  Calculando accesibilidad (fallback por vecinos)…")

    ser = u.select_atoms(ser612_selection)
    if ser.n_atoms == 0:
        raise ValueError(f"No se encontró Ser612: '{ser612_selection}'")
    prot = u.select_atoms(f"protein and not ({ser612_selection})")

    counts, times = [], []
    iterator = tqdm(frames, desc="Fallback vecinos") if verbose else frames
    for fi in iterator:
        u.trajectory[fi]
        D = distances.distance_array(ser.positions, prot.positions, box=u.dimensions)
        counts.append(int(np.sum(D < cutoff)))
        times.append(float(u.trajectory.time))

    df = pd.DataFrame({"frame": frames, "time_ps": times, "n_neighbors": counts})
    # Mapear a pseudo-SASA (0–100 Å^2 escala relativa)
    maxn, minn = df["n_neighbors"].max(), df["n_neighbors"].min()
    if maxn > minn:
        df["sasa_A2"] = 100.0 * (1.0 - (df["n_neighbors"] - minn) / (maxn - minn))
    else:
        df["sasa_A2"] = 50.0
    return df[["frame", "time_ps", "sasa_A2"]]


# ---------------------------------------------------------
# 3) Estado favorable (distancia + accesibilidad)
# ---------------------------------------------------------
def analyze_ser612_favorable_state(
    distance_df: pd.DataFrame,
    sasa_df: pd.DataFrame,
    optimal_range: Tuple[float, float],
    min_accessible: float,
    verbose: bool = True
) -> Dict:

    if distance_df is None or distance_df.empty or sasa_df is None or sasa_df.empty:
        raise ValueError("distance_df o sasa_df vacíos; no se puede analizar accesibilidad.")

    merged = distance_df.merge(sasa_df, on="frame", how="inner")
    dmin, dmax = float(optimal_range[0]), float(optimal_range[1])

    ok_dist = (merged["distance_A"] >= dmin) & (merged["distance_A"] <= dmax)
    ok_sasa = (merged["sasa_A2"] >= float(min_accessible))
    ok_both = ok_dist & ok_sasa

    n = len(merged)
    pct_dist  = 100.0 * ok_dist.sum()  / n if n else 0.0
    pct_sasa  = 100.0 * ok_sasa.sum()  / n if n else 0.0
    pct_both  = 100.0 * ok_both.sum()  / n if n else 0.0
    avg_d     = float(merged["distance_A"].mean())
    avg_s     = float(merged["sasa_A2"].mean())
    score     = pct_both / 100.0

    if pct_both >= 70.0:
        state, interp = "HIGHLY FAVORABLE", "Configuración óptima para glicosilación"
    elif pct_both >= 40.0:
        state, interp = "MODERATELY FAVORABLE", "Períodos favorables para glicosilación"
    else:
        state, interp = "UNFAVORABLE", "Raramente favorable para glicosilación"

    if verbose:
        print("\n  Estado Ser612:")
        print(f"    Distancia media: {avg_d:.2f} Å")
        print(f"    SASA media: {avg_s:.2f} Å²")
        print(f"    % distancia favorable: {pct_dist:.1f}%")
        print(f"    % SASA favorable:     {pct_sasa:.1f}%")
        print(f"    % favorables (ambos): {pct_both:.1f}%")
        print(f"    Glycosylation score:  {score:.3f}  →  {state}")

    return {
        "avg_distance_A": avg_d,
        "avg_sasa_A2": avg_s,
        "pct_favorable_distance": pct_dist,
        "pct_favorable_sasa": pct_sasa,
        "pct_favorable_both": pct_both,
        "glycosylation_score": score,
        "state": state,
        "interpretation": interp,
        "optimal_range_A": [dmin, dmax],
        "min_accessible_A2": float(min_accessible),
        "n_frames_analyzed": n
    }


# ---------------------------------------------------------
# 4) Orquestador
# ---------------------------------------------------------
def run_ser612_analysis(u: mda.Universe, config: Dict, verbose: bool = True):
    if verbose:
        print("=" * 80)
        print("MÓDULO 8: SER612 GLYCOSYLATION SITE ANALYSIS")
        print("=" * 80)

    s8 = config["ser612_analysis"]
    if not s8.get("enabled", True):
        if verbose:
            print("  Módulo 8 deshabilitado.")
        return None

    frames = select_frames(u, config)
    if verbose:
        print(f"\nFrames a analizar: {len(frames)}")

    out_base = config["output"]["base_dir"]
    os.makedirs(out_base, exist_ok=True)

    # 1) Distancia
    if verbose:
        print("\n1) Distancia Ser612 ↔ Glu529 …")
    dist_df = calculate_ser612_distance_timeseries(
        u, frames,
        s8["distance_to_catalytic"]["ser612_selection"],
        s8["distance_to_catalytic"]["glu529_selection"],
        verbose
    )
    dist_csv = os.path.join(out_base, s8["output"]["distance_timeseries"])
    os.makedirs(os.path.dirname(dist_csv), exist_ok=True)
    dist_df.to_csv(dist_csv, index=False)
    if verbose:
        print(f"   ✅ Guardado: {dist_csv}")

    # 2) SASA: FreeSASA → MDTraj → Fallback
    sasa_df = None
    if s8["sasa"].get("enabled", True):
        if verbose:
            print("\n2) SASA/Accesibilidad …")
        sasa_df = calculate_sasa_freesasa(u, frames, s8["distance_to_catalytic"]["ser612_selection"], verbose)
        if sasa_df is None:
            sasa_df = calculate_sasa_mdtraj(u, frames, s8["distance_to_catalytic"]["ser612_selection"], verbose)
        if sasa_df is None:
            sasa_df = calculate_accessibility_fallback(u, frames, s8["distance_to_catalytic"]["ser612_selection"], verbose=verbose)

        sasa_csv = os.path.join(out_base, s8["output"]["sasa_timeseries"])
        os.makedirs(os.path.dirname(sasa_csv), exist_ok=True)
        sasa_df.to_csv(sasa_csv, index=False)
        if verbose:
            print(f"   ✅ Guardado: {sasa_csv}")

    # 3) Estado favorable
    analysis = None
    if sasa_df is not None:
        if verbose:
            print("\n3) Análisis de estado favorable …")
        rng = tuple(s8["distance_to_catalytic"]["optimal_range"])
        min_acc = float(s8["sasa"]["min_accessible"])
        analysis = analyze_ser612_favorable_state(dist_df, sasa_df, rng, min_acc, verbose)
        acc_json = os.path.join(out_base, s8["output"]["accessibility_score"])
        with open(acc_json, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        if verbose:
            print(f"   ✅ Guardado: {acc_json}")

    return {"distance": dist_df, "sasa": sasa_df, "analysis": analysis}

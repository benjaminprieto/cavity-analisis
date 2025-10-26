# 01_src/cavity_analysis/core/network.py
"""
Módulo 1: Análisis de red de contactos global
- Calcula matriz de contactos persistentes a partir de PDB + DCD (MDAnalysis)
- Construye red (NetworkX) y calcula centralidades
- Identifica hubs (Top-N por métrica)
- Guarda: network_graph.pkl, centrality_ranking.csv, hub_residues.json

Requiere:
  - MDAnalysis, networkx, numpy, pandas, pyyaml
  - tqdm (opcional; barra de progreso)

YAML esperado (03_configs/analyses/01_network.yaml):
  run / inputs / frames / global_network / output
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import pickle
import yaml
import numpy as np
import pandas as pd
import networkx as nx
import MDAnalysis as mda
from MDAnalysis.analysis import distances

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # opcional


# =============================== UTILIDADES IO ===============================

def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


# =============================== FRAMES SELECCIÓN ============================

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

    else:  # "stride" (default)
        stride = int(frames_cfg.get("stride", 1) or 1)
        start = int(frames_cfg.get("start", 0) or 0)
        stop  = frames_cfg.get("stop", None)
        stop  = int(stop) if stop is not None else n
        start = max(0, start); stop = min(n, stop)
        frames = list(range(start, stop, max(1, stride)))

    if verbose:
        print(f"[INFO] Frames seleccionados: {len(frames)} (mode={mode})")
    return frames


# =============================== CÁLCULOS PRINCIPALES ========================

def calculate_contact_matrix(u, frames, selection, cutoff, verbose=True):
    """
    Calcula matriz de contactos persistentes (frecuencia 0–1) y devuelve el AtomGroup usado.
    """
    atoms = u.select_atoms(selection)
    n_atoms = atoms.n_atoms
    if n_atoms == 0:
        raise ValueError(f"La selección '{selection}' no devolvió átomos.")

    contact_freq = np.zeros((n_atoms, n_atoms), dtype=float)
    iterator = tqdm(frames, desc="Calculando contactos") if (verbose and tqdm) else frames

    for frame_idx in iterator:
        u.trajectory[frame_idx]
        dist_matrix = distances.distance_array(
            atoms.positions,
            atoms.positions,
            box=u.dimensions
        )
        # contacto si dist < cutoff (evitar diagonal 0)
        contacts = (dist_matrix < float(cutoff)) & (dist_matrix > 0.0)
        contact_freq += contacts.astype(float)

    contact_matrix = contact_freq / max(1, len(frames))
    return contact_matrix, atoms


def build_contact_network(contact_matrix, atoms, min_persistence):
    """
    Construye grafo de contactos: nodos=átomos, aristas si persistencia >= umbral.
    """
    n_atoms = len(atoms)
    G = nx.Graph()

    # Nodos con atributos
    for i, atom in enumerate(atoms):
        G.add_node(
            i,
            resid=int(atom.resid),
            resname=str(atom.resname),
            segid=str(atom.segid),   # cadenas: A (NRP1), B (XYLT1), U (UDP)
            index=i
        )

    thr = float(min_persistence)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            p = float(contact_matrix[i, j])
            if p >= thr:
                G.add_edge(i, j, weight=p)

    return G


def calculate_centrality_metrics(G, metrics, verbose=True):
    """
    Calcula métricas de centralidad (lista configurable en YAML).
    """
    results = {}
    if verbose:
        print("\nCalculando métricas de centralidad...")

    if "betweenness_centrality" in metrics:
        if verbose: print("  - Betweenness centrality...")
        results["betweenness"] = nx.betweenness_centrality(G)

    if "eigenvector_centrality" in metrics:
        if verbose: print("  - Eigenvector centrality...")
        try:
            results["eigenvector"] = nx.eigenvector_centrality(G, max_iter=1000)
        except Exception:
            if verbose: print("    ⚠️  No convergió, usando PageRank")
            results["eigenvector"] = nx.pagerank(G)

    if "closeness_centrality" in metrics:
        if verbose: print("  - Closeness centrality...")
        results["closeness"] = nx.closeness_centrality(G)

    if "degree" in metrics:
        if verbose: print("  - Degree...")
        results["degree"] = {n: d for n, d in G.degree()}

    return results


def identify_hubs(centrality_results, G, top_n):
    """
    Top-N hubs por métrica, con info de resid/segid.
    """
    hubs = {}
    for metric_name, values in centrality_results.items():
        sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)
        top_nodes = sorted_nodes[:int(top_n)]
        hubs[metric_name] = []
        for node_idx, value in top_nodes:
            nd = G.nodes[node_idx]
            hubs[metric_name].append({
                "node_idx": int(node_idx),
                "resid": int(nd["resid"]),
                "resname": str(nd["resname"]),
                "segid": str(nd["segid"]),  # A, B, U (según PDB)
                "value": float(value)
            })
    return hubs


# =============================== ORQUESTADOR ================================

def run_global_network(u, config: Dict[str, Any], verbose: bool = True):
    """
    Ejecuta el análisis global con Universe ya cargado.
    """
    net_cfg = config["global_network"]
    selection = net_cfg["selection"]
    cutoff = float(net_cfg["contact_cutoff"])
    min_persistence = float(net_cfg["min_persistence"])
    top_n = int(net_cfg["top_n_hubs"])
    metrics = list(net_cfg["metrics"])

    if verbose:
        print("=" * 80)
        print("MÓDULO 1: GLOBAL NETWORK ANALYSIS")
        print("=" * 80)

    frames = _select_frames(u, config.get("frames", {}), verbose=verbose)

    if verbose:
        print(f"\nParámetros:")
        print(f"  Selección: {selection}")
        print(f"  Cutoff: {cutoff} Å")
        print(f"  Persistencia mínima: {min_persistence}")
        print(f"  Frames a analizar: {len(frames)}")

    # 1) Contactos
    C, atoms = calculate_contact_matrix(u, frames, selection, cutoff, verbose=verbose)
    if verbose:
        persist_edges = int(((C >= min_persistence).sum() - C.shape[0]) // 2)
        print(f"\nMatriz de contactos: {C.shape}")
        print(f"  Contactos persistentes: {persist_edges}")

    # 2) Red
    if verbose: print("\nConstruyendo red...")
    G = build_contact_network(C, atoms, min_persistence)
    if verbose:
        print(f"  Nodos: {G.number_of_nodes()}")
        print(f"  Aristas: {G.number_of_edges()}")

    # 3) Centralidades
    centrality = calculate_centrality_metrics(G, metrics, verbose=verbose)

    # 4) Hubs
    if verbose: print(f"\nIdentificando top {top_n} hubs...")
    hubs = identify_hubs(centrality, G, top_n)

    # 5) Ranking por nodo (mapea cadenas a nombres de proteína)
    chains_map = config.get("inputs", {}).get("chains", {})  # {'nrp1':'A','xylt1':'B','udp':'U'}
    chainA = str(chains_map.get("nrp1", "A"))  # NRP1
    chainB = str(chains_map.get("xylt1", "B"))  # XYLT1

    rows = []
    for i, atom in enumerate(atoms):
        seg = str(atom.segid)
        protein = "NRP1" if seg == chainA else ("XYLT1" if seg == chainB else "OTHER")
        row = {
            "node_idx": i,
            "resid": int(atom.resid),
            "resname": str(atom.resname),
            "segid": seg,           # verás A/B/U según PDB
            "protein": protein
        }
        for m, vals in centrality.items():
            row[m] = float(vals.get(i, 0.0))
        rows.append(row)

    ranking_df = pd.DataFrame(rows)
    sort_by = "betweenness" if "betweenness" in ranking_df.columns else (list(centrality.keys())[0] if centrality else None)
    if sort_by:
        ranking_df = ranking_df.sort_values(sort_by, ascending=False)

    # 6) Guardar (sin networkx.gpickle; solo pickle estándar)
    outdir = Path(config["output"]["base_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    # Nota: cambiamos a .pkl para dejar claro que es pickle puro
    net_path = outdir / net_cfg["output"]["network_graph"]
    if net_path.suffix.lower() != ".pkl":
        net_path = net_path.with_suffix(".pkl")  # fuerza .pkl por claridad
    rank_path = outdir / net_cfg["output"]["centrality_ranking"]
    hubs_path = outdir / net_cfg["output"]["hub_residues"]

    _ensure_parent(net_path); _ensure_parent(rank_path); _ensure_parent(hubs_path)

    with open(net_path, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)

    ranking_df.to_csv(rank_path, index=False)
    with hubs_path.open("w", encoding="utf-8") as f:
        json.dump(hubs, f, indent=2)

    if verbose:
        print(f"\n✅ Resultados guardados:")
        print(f"  - {net_path}")
        print(f"  - {rank_path}")
        print(f"  - {hubs_path}")

    return {
        "graph_path": str(net_path),
        "ranking_path": str(rank_path),
        'hubs_path': str(hubs_path),
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
    }


# =============================== ENTRADA DESDE YAML ==========================

def run_global_network_from_yaml(yaml_path: Path):
    """
    Entrada conveniente:
      - Lee YAML (03_configs/analyses/01_network.yaml)
      - Carga PDB + DCD
      - Ejecuta análisis y guarda outputs
    """
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
    if verbose:
        print(f"[INFO] Frames en trayectoria: {len(u.trajectory)}")

    Path(cfg["output"]["base_dir"]).mkdir(parents=True, exist_ok=True)
    return run_global_network(u, cfg, verbose=verbose)

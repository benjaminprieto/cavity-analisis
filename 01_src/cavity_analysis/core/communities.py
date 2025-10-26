# 01_src/cavity_analysis/core/communities.py
"""
Módulo 6: Community Detection
- Carga el grafo del Módulo 1 (network_graph.pkl)
- Detecta comunidades (Louvain si está python-louvain; si no, greedy_modularity)
- Calcula:
    * asignación nodo→comunidad
    * estadísticas por comunidad (tamaño y desglose por segid)
    * nodos inter-comunidad (puentes)
- Escribe:
    * communities.json
    * community_assignment.csv
    * community_stats.csv
    * inter_community_residues.csv
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json
import pickle
import pandas as pd
import networkx as nx


# ----------------------------- utilidades IO ---------------------------------

def _read_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _load_graph(pkl_path: Path) -> nx.Graph:
    with pkl_path.open("rb") as f:
        G = pickle.load(f)
    if not isinstance(G, nx.Graph):
        raise TypeError("El objeto cargado no es networkx.Graph")
    return G


# -------------------------- detección de comunidades -------------------------

def detect_communities_louvain(G: nx.Graph, resolution: float = 1.0) -> Dict[int, int]:
    """
    Devuelve dict {node_idx: community_id}
    """
    try:
        import community as community_louvain  # paquete: python-louvain
        return community_louvain.best_partition(G, resolution=resolution)
    except ImportError:
        print("  ⚠️  'python-louvain' no disponible. Usando greedy modularity (fallback).")
        communities_gen = nx.community.greedy_modularity_communities(G)
        mapping: Dict[int, int] = {}
        for i, comm in enumerate(communities_gen):
            for node in comm:
                mapping[int(node)] = int(i)
        return mapping


def analyze_communities(G: nx.Graph, assignments: Dict[int, int]) -> pd.DataFrame:
    """
    Estadísticas por comunidad:
      - n_nodes
      - lista de resids
      - conteo por segid
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for node, cid in assignments.items():
        groups[int(cid)].append(int(node))

    rows = []
    for cid, nodes in groups.items():
        segids = [G.nodes[n].get("segid", "") for n in nodes]
        resids = [int(G.nodes[n].get("resid", -1)) for n in nodes]
        seg_counts = pd.Series(segids, dtype="object").value_counts().to_dict()
        row = {"community": int(cid), "n_nodes": int(len(nodes)), "residues": resids}
        # añade columnas n_<segid>
        for seg, cnt in seg_counts.items():
            row[f"n_{seg}"] = int(cnt)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("n_nodes", ascending=False)
    return df


def identify_inter_community_residues(G: nx.Graph, assignments: Dict[int, int]) -> pd.DataFrame:
    """
    Nodos que conectan comunidades diferentes (puentes).
    """
    rows = []
    for n in G.nodes():
        cid = int(assignments[n])
        neighbors_other = [m for m in G.neighbors(n) if int(assignments[m]) != cid]
        if neighbors_other:
            d = G.nodes[n]
            rows.append({
                "node": int(n),
                "resid": int(d.get("resid", -1)),
                "resname": str(d.get("resname", "")),
                "segid": str(d.get("segid", "")),
                "community": cid,
                "n_inter_connections": int(len(neighbors_other)),
                "degree": int(G.degree(n)),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["n_inter_connections", "degree"], ascending=False)
    return df


# -------------------------------- orquestador --------------------------------

def run_communities_from_yaml(yaml_path: Path):
    cfg = _read_yaml(yaml_path)
    verbose = bool(cfg.get("run", {}).get("verbose", True))

    outdir = Path(cfg["output"]["base_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    net_pkl = Path(cfg["artifacts"]["network_graph"])
    if verbose:
        print("=" * 80)
        print("MÓDULO 6: COMMUNITY DETECTION")
        print("=" * 80)
        print(f"[INFO] Cargando grafo: {net_pkl}")
    if not net_pkl.exists():
        raise FileNotFoundError(f"No existe network_graph.pkl: {net_pkl}")

    G = _load_graph(net_pkl)
    if verbose:
        print(f"[INFO] Grafo: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

    method = str(cfg["community_detection"]["method"]).lower()
    resolution = float(cfg["community_detection"].get("resolution", 1.0))
    if verbose:
        print(f"[INFO] Método: {method} | resolución={resolution}")

    # Detectar
    assignments = detect_communities_louvain(G, resolution=resolution)
    n_communities = len(set(assignments.values()))
    if verbose:
        print(f"[INFO] Comunidades detectadas: {n_communities}")

    # Stats por comunidad
    stats_df = analyze_communities(G, assignments)
    if verbose and not stats_df.empty:
        print("\nTop comunidades por tamaño:")
        print(stats_df[["community", "n_nodes"]].head(10))

    # Inter-comunidad
    inter_df = identify_inter_community_residues(G, assignments)
    if verbose:
        print(f"\n[INFO] Nodos inter-comunidad: {len(inter_df)}")

    # ------------------------------- Guardar ---------------------------------
    out_cfg = cfg["community_detection"]["output"]

    # 1) JSON resumen
    json_path = outdir / out_cfg["communities_json"]
    _ensure_parent(json_path)
    # para JSON, haz dict serializable
    assignments_json = {int(k): int(v) for k, v in assignments.items()}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "n_communities": int(n_communities),
            "assignments": assignments_json,
            "stats": stats_df.to_dict("records")
        }, f, indent=2)
    if verbose:
        print(f"\n✅ Comunidades (JSON): {json_path}")

    # 2) CSV asignación detallada por nodo/resid/segid
    assign_rows = []
    for n, cid in assignments.items():
        d = G.nodes[n]
        assign_rows.append({
            "node": int(n),
            "community": int(cid),
            "resid": int(d.get("resid", -1)),
            "resname": str(d.get("resname", "")),
            "segid": str(d.get("segid", "")),
        })
    assign_df = pd.DataFrame(assign_rows).sort_values(["community", "segid", "resid"])
    assign_csv = outdir / out_cfg["community_assignment"]
    _ensure_parent(assign_csv)
    assign_df.to_csv(assign_csv, index=False)

    # 3) CSV stats
    stats_csv = outdir / out_cfg["community_stats"]
    _ensure_parent(stats_csv)
    stats_df.to_csv(stats_csv, index=False)

    # 4) CSV inter-comunidad
    inter_csv = outdir / out_cfg["inter_community"]
    _ensure_parent(inter_csv)
    inter_df.to_csv(inter_csv, index=False)
    if verbose:
        print(f"✅ Asignación:          {assign_csv}")
        print(f"✅ Estadísticas:        {stats_csv}")
        print(f"✅ Inter-comunidad:     {inter_csv}")

    return {
        "assignments": assignments,
        "n_communities": n_communities,
        "community_assignment_csv": str(assign_csv),
        "community_stats_csv": str(stats_csv),
        "inter_community_csv": str(inter_csv),
        "json": str(json_path),
    }

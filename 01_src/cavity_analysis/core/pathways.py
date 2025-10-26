# 01_src/cavity_analysis/core/pathways.py
"""
Módulo 5: Pathway Analysis
- Carga grafo del Módulo 1 (pickle) y energías inter del Módulo 2 (CSV)
- Opcionalmente inspecciona PDB+DCD para identificar interfaz A↔B
- Pondera aristas por energía (|E|) o por persistencia de contacto
- Busca rutas (shortest + alternativas) desde hubs → interfaz
- Identifica bottlenecks (residuos que aparecen en muchos paths)
- Exporta:
    * weighted_paths.json
    * bottleneck_residues.csv
    * pathway_importance.csv
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import MDAnalysis as mda

# ---------------------------- utilidades generales ---------------------------

def _read_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _select_frames(u, frames_cfg: Dict[str, Any], verbose: bool = True) -> List[int]:
    n = len(u.trajectory)
    mode = (frames_cfg or {}).get("mode", "stride")
    if mode == "list":
        lst = frames_cfg.get("list") or []
        frames = [int(f) for f in lst if 0 <= int(f) < n]
    else:
        stride = int(frames_cfg.get("stride", 1) or 1)
        start = int(frames_cfg.get("start", 0) or 0)
        stop  = frames_cfg.get("stop", None)
        stop  = int(stop) if stop is not None else n
        frames = list(range(max(0,start), min(n,stop), max(1,stride)))
    if verbose:
        print(f"[INFO] Frames seleccionados (para interfaz si se usa): {len(frames)}")
    return frames

# -------------------------- funciones de este módulo -------------------------

def identify_interface_residues(u, chain1: str, chain2: str, cutoff: float = 5.0) -> List[int]:
    """Residuos en contacto (A o B) si algún átomo A-B está < cutoff (Å) en el frame actual."""
    from MDAnalysis.analysis import distances
    atoms1 = u.select_atoms(f"segid {chain1}")
    atoms2 = u.select_atoms(f"segid {chain2}")
    if len(atoms1) == 0 or len(atoms2) == 0:
        return []
    d = distances.distance_array(atoms1.positions, atoms2.positions, box=u.dimensions)
    where = np.where(d < float(cutoff))
    if where[0].size == 0:
        return []
    res1 = set(atoms1[where[0]].resids)
    res2 = set(atoms2[where[1]].resids)
    return sorted(list(res1 | res2))

def _load_graph(pkl_path: Path) -> nx.Graph:
    with pkl_path.open("rb") as f:
        G = pickle.load(f)
    if not isinstance(G, nx.Graph):
        raise TypeError("El objeto cargado no es networkx.Graph")
    # Sanity: nodos deben tener al menos resid, resname, segid
    return G

def _load_inter_energy(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # columnas esperadas: resid1,resname1,segid1,resid2,resname2,segid2,e_coulomb,e_lj,e_total
    if "e_total" not in df.columns:
        raise ValueError("inter_energy CSV no tiene columna 'e_total'")
    return df

def _apply_energy_weights(G_in: nx.Graph, energy_df: pd.DataFrame) -> nx.Graph:
    """Añade 'energy_weight' a cada arista: cost = 1/(|E_total|+0.1)."""
    G = G_in.copy()
    # dict rápido por tupla (resid_u, resid_v) indistinta de orden
    energy_map = {}
    for _, r in energy_df.iterrows():
        key = tuple(sorted((int(r["resid1"]), int(r["resid2"]))))
        energy_map[key] = float(abs(r["e_total"]))
    for u, v, data in G.edges(data=True):
        ru = int(G.nodes[u].get("resid", -999999))
        rv = int(G.nodes[v].get("resid", -999999))
        e = energy_map.get(tuple(sorted((ru, rv))), None)
        if e is None:
            e = 0.0  # sin dato => neutro
        data["energy_weight"] = 1.0 / (abs(e) + 0.1)
    return G

def _apply_persistence_weights(G_in: nx.Graph) -> nx.Graph:
    """Usa 'weight' (persistencia de contacto) del grafo base, conviértelo en coste: 1/(persist+1e-6)."""
    G = G_in.copy()
    for _, _, data in G.edges(data=True):
        persist = float(data.get("weight", 0.0))  # en módulo 1, 'weight' = persistencia ∈ [0,1]
        data["persistence_weight"] = 1.0 / (persist + 1e-6)
    return G

def _shortest_and_alternatives(G: nx.Graph, source: int, target: int,
                               n_paths: int, max_len: int, weight_key: str) -> List[Dict[str, Any]]:
    """Shortest path (ponderada) + algunas alternativas simples."""
    out = []
    try:
        sp = nx.shortest_path(G, source, target, weight=weight_key)
        if len(sp) <= max_len:
            out.append(sp)
    except nx.NetworkXNoPath:
        return []

    # alternativas: all_simple_paths (no ponderadas), recorta a n_paths-1
    count = 1
    try:
        for path in nx.all_simple_paths(G, source, target, cutoff=max_len):
            if path == out[0]:
                continue
            out.append(path)
            count += 1
            if count >= n_paths:
                break
    except Exception:
        pass
    return out

def _score_path(G: nx.Graph, path: List[int], weight_key: str) -> float:
    """Suma de pesos de coste (menor es mejor)."""
    s = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        s += float(G[u][v].get(weight_key, 1.0))
    return s

def _nodes_from_resids(G: nx.Graph, resids: List[int]) -> List[int]:
    """Devuelve índices de nodos cuyo atributo 'resid' pertenece a resids."""
    S = set(int(r) for r in resids)
    return [n for n, d in G.nodes(data=True) if int(d.get("resid", 10**9)) in S]

def _top_hub_nodes_from_ranking_csv(ranking_csv: Path, top_n: int) -> List[int]:
    df = pd.read_csv(ranking_csv)
    if "node_idx" not in df.columns:
        raise ValueError("centrality_ranking.csv no tiene 'node_idx'")
    return df.head(int(top_n))["node_idx"].astype(int).tolist()

def identify_bottleneck_residues(paths: List[List[int]], G: nx.Graph, threshold: float) -> pd.DataFrame:
    """Cuenta frecuencia de residuos (por resid) a través de todos los caminos y filtra por umbral."""
    from collections import Counter
    cnt = Counter()
    for path in paths:
        for n in path:
            resid = int(G.nodes[n].get("resid", 10**9))
            cnt[resid] += 1
    total = max(1, len(paths))
    rows = []
    for resid, c in cnt.items():
        freq = c / total
        if freq >= float(threshold):
            rows.append(dict(resid=resid, appearances=c, frequency=freq))
    df = pd.DataFrame(rows).sort_values("frequency", ascending=False)
    return df

# -------------------------------- orquestador --------------------------------

def run_pathways_from_yaml(yaml_path: Path):
    cfg = _read_yaml(yaml_path)
    verbose = bool(cfg.get("run", {}).get("verbose", True))

    # cargar grafo (M1) y energías (M2)
    net_pkl = Path(cfg["artifacts"]["network_graph"])
    if not net_pkl.exists():
        raise FileNotFoundError(f"No existe grafo de red: {net_pkl}")
    G_base = _load_graph(net_pkl)

    # para hubs necesitaremos el ranking de 01_network (mismo folder que el pkl)
    ranking_csv = net_pkl.with_name("centrality_ranking.csv")
    if not ranking_csv.exists():
        raise FileNotFoundError(f"No existe centrality_ranking.csv junto al grafo: {ranking_csv}")

    edge_policy = str(cfg["pathways"]["edge_weight"]).lower()
    if edge_policy == "interaction_energy":
        inter_csv = Path(cfg["artifacts"]["inter_energy"])
        if not inter_csv.exists():
            raise FileNotFoundError(f"No existe inter_energy CSV: {inter_csv}")
        energy_df = _load_inter_energy(inter_csv)
        G = _apply_energy_weights(G_base, energy_df)
        weight_key = "energy_weight"
    else:  # "persistence"
        G = _apply_persistence_weights(G_base)
        weight_key = "persistence_weight"

    if verbose:
        print("=" * 80)
        print("MÓDULO 5: PATHWAY ANALYSIS")
        print("=" * 80)
        print(f"Grafo base: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
        print(f"Estrategia de pesos: {edge_policy}  (clave='{weight_key}')")

    # Universe para interfaz (si aplica)
    pdb = Path(cfg["inputs"]["pdb"])
    dcd = Path(cfg["inputs"]["dcd"])
    if not pdb.exists():
        raise FileNotFoundError(f"No existe PDB: {pdb}")
    if not dcd.exists():
        raise FileNotFoundError(f"No existe DCD: {dcd}")
    u = mda.Universe(str(pdb), str(dcd))

    # (opcional) avanzar al primer frame seleccionado para medir interfaz
    frames = _select_frames(u, cfg.get("frames", {}), verbose=verbose)
    if frames:
        u.trajectory[frames[0]]  # frame representativo

    chainA = cfg["inputs"]["chains"]["nrp1"]  # "A"
    chainB = cfg["inputs"]["chains"]["xylt1"] # "B"

    # obtener nodos fuente (top hubs)
    topN = int(cfg["pathways"]["n_source_residues"])
    hub_nodes = _top_hub_nodes_from_ranking_csv(ranking_csv, topN)
    if verbose:
        print(f"Nodos fuente (hubs): {len(hub_nodes)}")

    # targets por interfaz
    target_nodes: List[int] = []
    tgt_cfg = cfg["pathways"]["targets"]
    if str(tgt_cfg.get("type", "interface")).lower() == "interface":
        cutoff = float(tgt_cfg.get("cutoff", 5.0))
        iface_resids = identify_interface_residues(u, chainA, chainB, cutoff)
        target_nodes = _nodes_from_resids(G, iface_resids)
        if verbose:
            print(f"Nodos target (interfaz, cutoff={cutoff} Å): {len(target_nodes)}")
    else:
        # aquí podrías implementar otras estrategias de target si hiciera falta
        pass

    # búsqueda de caminos
    n_paths = int(cfg["pathways"]["n_paths_per_source"])
    max_len = int(cfg["pathways"]["max_path_length"])
    if verbose:
        print("\nBuscando pathways...")
    all_paths: List[Dict[str, Any]] = []
    simple_paths_list: List[List[int]] = []

    for src in hub_nodes:
        for tgt in target_nodes:
            if src == tgt:
                continue
            paths = _shortest_and_alternatives(G, src, tgt, n_paths=n_paths, max_len=max_len, weight_key=weight_key)
            for p in paths:
                residues = [int(G.nodes[n]["resid"]) for n in p]
                score = _score_path(G, p, weight_key)
                all_paths.append(dict(
                    source=int(src),
                    target=int(tgt),
                    path=[int(n) for n in p],
                    length=int(len(p)),
                    residues=residues,
                    score=float(score)
                ))
                simple_paths_list.append(p)

    if verbose:
        print(f"  Pathways encontrados: {len(all_paths)}")

    # bottlenecks
    if verbose:
        print("\nIdentificando bottlenecks...")
    thr = float(cfg["pathways"]["bottleneck_threshold"])
    bottlenecks_df = identify_bottleneck_residues(simple_paths_list, G, threshold=thr)
    if verbose:
        print(f"  Bottleneck residues: {len(bottlenecks_df)}")

    # importancia de rutas (ordenarlas por score ascendente)
    if all_paths:
        imp_df = pd.DataFrame(all_paths).sort_values("score", ascending=True)
    else:
        imp_df = pd.DataFrame(columns=["source","target","path","length","residues","score"])

    # escribir salidas
    outdir = Path(cfg["output"]["base_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    # weighted_paths.json
    paths_json = outdir / cfg["pathways"]["output"]["pathways"]
    _ensure_parent(paths_json)
    with paths_json.open("w", encoding="utf-8") as f:
        json.dump(all_paths, f, indent=2)
    # bottlenecks.csv
    bott_csv = outdir / cfg["pathways"]["output"]["bottlenecks"]
    _ensure_parent(bott_csv)
    bottlenecks_df.to_csv(bott_csv, index=False)
    # pathway_importance.csv
    imp_csv = outdir / cfg["pathways"]["output"]["importance"]
    _ensure_parent(imp_csv)
    imp_df.to_csv(imp_csv, index=False)

    if verbose:
        print(f"\n✅ Pathways guardados: {paths_json}")
        print(f"✅ Bottlenecks:        {bott_csv}")
        print(f"✅ Importance:         {imp_csv}")
        print("\nTop 10 bottlenecks:")
        print(bottlenecks_df.head(10))

    return dict(
        graph=G,
        paths_json=str(paths_json),
        bottlenecks_csv=str(bott_csv),
        importance_csv=str(imp_csv)
    )

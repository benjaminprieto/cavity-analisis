# 01_src/cavity_analysis/core/pathways_optimized.py
"""
Module 5a: Pathway Analysis (OPTIMIZED)

Features:
- Loads graph from Module 1 (pickle) and inter-energies from Module 2 (CSV)
- Optionally inspects PDB+DCD to identify interface A↔B
- Weights edges by energy (|E|) or contact persistence
- Searches paths (shortest + alternatives) from hubs → interface
- Identifies bottlenecks (residues appearing in many paths)
- Exports:
    * weighted_paths.json
    * bottleneck_residues.csv
    * pathway_importance.csv

OPTIMIZATIONS:
- Progress bars (tqdm)
- Detailed logging with timestamps
- More efficient search with timeout
- Incremental checkpoint saving
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import pickle
import time
from datetime import datetime
import numpy as np
import pandas as pd
import networkx as nx
import MDAnalysis as mda

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  WARNING: tqdm not available. Install with: pip install tqdm")


# ---------------------------- General utilities ------------------------------

def _read_yaml(path: Path) -> Dict[str, Any]:
    """Read YAML configuration file."""
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_parent(p: Path) -> None:
    """Ensure parent directory exists."""
    p.parent.mkdir(parents=True, exist_ok=True)


def _log(msg: str, verbose: bool = True):
    """Log message with timestamp."""
    if verbose:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")


def _select_frames(u, frames_cfg: Dict[str, Any], verbose: bool = True) -> List[int]:
    """Select frames from trajectory based on config."""
    n = len(u.trajectory)
    mode = (frames_cfg or {}).get("mode", "stride")
    if mode == "list":
        lst = frames_cfg.get("list") or []
        frames = [int(f) for f in lst if 0 <= int(f) < n]
    else:
        stride = int(frames_cfg.get("stride", 1) or 1)
        start = int(frames_cfg.get("start", 0) or 0)
        stop = frames_cfg.get("stop", None)
        stop = int(stop) if stop is not None else n
        frames = list(range(max(0, start), min(n, stop), max(1, stride)))
    if verbose:
        _log(f"Frames selected: {len(frames)}", verbose)
    return frames


# -------------------------- Module-specific functions ------------------------

def identify_interface_residues(u, chain1: str, chain2: str, cutoff: float = 5.0) -> List[int]:
    """
    Identify interface residues between two chains.

    Returns residues (from either chain A or B) where any atom pair A-B
    is within cutoff distance (Angstroms) in the current frame.
    """
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
    """Load NetworkX graph from pickle file."""
    with pkl_path.open("rb") as f:
        G = pickle.load(f)
    if not isinstance(G, nx.Graph):
        raise TypeError("Loaded object is not a networkx.Graph")
    return G


def _load_inter_energy(csv_path: Path) -> pd.DataFrame:
    """Load interaction energy CSV."""
    df = pd.read_csv(csv_path)
    # Expected columns: resid1,resname1,segid1,resid2,resname2,segid2,e_coulomb,e_lj,e_total
    if "e_total" not in df.columns:
        raise ValueError("inter_energy CSV missing 'e_total' column")
    return df


def _apply_energy_weights(G_in: nx.Graph, energy_df: pd.DataFrame, verbose: bool = True) -> nx.Graph:
    """
    Add 'energy_weight' to each edge: cost = 1/(|E_total|+0.1).
    Lower energy (stronger interaction) = lower cost = preferred path.
    """
    _log("Applying energy weights...", verbose)
    G = G_in.copy()
    energy_map = {}
    for _, r in energy_df.iterrows():
        key = tuple(sorted((int(r["resid1"]), int(r["resid2"]))))
        energy_map[key] = float(abs(r["e_total"]))

    for u, v, data in G.edges(data=True):
        ru = int(G.nodes[u].get("resid", -999999))
        rv = int(G.nodes[v].get("resid", -999999))
        e = energy_map.get(tuple(sorted((ru, rv))), 0.0)
        data["energy_weight"] = 1.0 / (abs(e) + 0.1)
    return G


def _apply_persistence_weights(G_in: nx.Graph, verbose: bool = True) -> nx.Graph:
    """
    Use 'weight' (contact persistence) from base graph as cost.
    Higher persistence = lower cost = preferred path.
    """
    _log("Applying persistence weights...", verbose)
    G = G_in.copy()
    for _, _, data in G.edges(data=True):
        persist = float(data.get("weight", 0.0))  # From module 1, 'weight' = persistence ∈ [0,1]
        data["persistence_weight"] = 1.0 / (persist + 1e-6)
    return G


def _shortest_and_alternatives_fast(
        G: nx.Graph,
        source: int,
        target: int,
        n_paths: int,
        max_len: int,
        weight_key: str,
        max_time_seconds: float = 2.0
) -> List[List[int]]:
    """
    Find shortest path + alternatives with TIME LIMIT.
    Much faster than naive all_simple_paths approach.

    Args:
        G: NetworkX graph
        source: Source node index
        target: Target node index
        n_paths: Maximum number of paths to find
        max_len: Maximum path length
        weight_key: Edge weight attribute to use for shortest path
        max_time_seconds: Maximum time to spend on alternative paths

    Returns:
        List of paths (each path is a list of node indices)
    """
    out = []

    # 1. Weighted shortest path
    try:
        sp = nx.shortest_path(G, source, target, weight=weight_key)
        if len(sp) <= max_len:
            out.append(sp)
    except nx.NetworkXNoPath:
        return []

    if n_paths <= 1:
        return out

    # 2. Alternative paths with time limit
    start_time = time.time()
    count = 1

    try:
        for path in nx.all_simple_paths(G, source, target, cutoff=max_len):
            if time.time() - start_time > max_time_seconds:
                break
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
    """Calculate total cost of a path (sum of edge weights)."""
    s = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        s += float(G[u][v].get(weight_key, 1.0))
    return s


def _nodes_from_resids(G: nx.Graph, resids: List[int]) -> List[int]:
    """Get node indices whose 'resid' attribute belongs to resids list."""
    S = set(int(r) for r in resids)
    return [n for n, d in G.nodes(data=True) if int(d.get("resid", 10 ** 9)) in S]


def _top_hub_nodes_from_ranking_csv(ranking_csv: Path, top_n: int) -> List[int]:
    """Get top N hub nodes from centrality ranking CSV."""
    df = pd.read_csv(ranking_csv)
    if "node_idx" not in df.columns:
        raise ValueError("centrality_ranking.csv missing 'node_idx' column")
    return df.head(int(top_n))["node_idx"].astype(int).tolist()


def identify_bottleneck_residues(paths: List[List[int]], G: nx.Graph, threshold: float) -> pd.DataFrame:
    """
    Count residue frequency across all paths and filter by threshold.

    Bottlenecks are residues that appear in many different paths,
    indicating they are critical connection points.
    """
    from collections import Counter
    cnt = Counter()
    for path in paths:
        for n in path:
            resid = int(G.nodes[n].get("resid", 10 ** 9))
            cnt[resid] += 1
    total = max(1, len(paths))
    rows = []
    for resid, c in cnt.items():
        freq = c / total
        if freq >= float(threshold):
            rows.append(dict(resid=resid, appearances=c, frequency=freq))
    df = pd.DataFrame(rows).sort_values("frequency", ascending=False) if rows else pd.DataFrame()
    return df


def _save_checkpoint(all_paths: List[Dict], checkpoint_path: Path):
    """Save incremental checkpoint."""
    _ensure_parent(checkpoint_path)
    with checkpoint_path.open("w", encoding="utf-8") as f:
        json.dump(all_paths, f, indent=2)


# -------------------------------- Orchestrator --------------------------------

def run_pathways_from_yaml(yaml_path: Path):
    """
    Main orchestrator function for pathway analysis.

    Loads configuration from YAML, processes the network graph with
    appropriate edge weights, identifies source (hubs) and target (interface)
    nodes, searches for paths, identifies bottlenecks, and exports results.
    """
    cfg = _read_yaml(yaml_path)
    verbose = bool(cfg.get("run", {}).get("verbose", True))

    _log("=" * 80, verbose)
    _log("MODULE 5a: PATHWAY ANALYSIS (OPTIMIZED)", verbose)
    _log("=" * 80, verbose)

    # Load graph
    net_pkl = Path(cfg["artifacts"]["network_graph"])
    if not net_pkl.exists():
        raise FileNotFoundError(f"Network graph not found: {net_pkl}")

    _log(f"Loading graph from {net_pkl.name}...", verbose)
    G_base = _load_graph(net_pkl)
    _log(f"  → {G_base.number_of_nodes()} nodes, {G_base.number_of_edges()} edges", verbose)

    # Hub ranking
    ranking_csv = net_pkl.with_name("centrality_ranking.csv")
    if not ranking_csv.exists():
        raise FileNotFoundError(f"centrality_ranking.csv not found")

    # Apply weights
    edge_policy = str(cfg["pathways"]["edge_weight"]).lower()
    if edge_policy == "interaction_energy":
        inter_csv = Path(cfg["artifacts"]["inter_energy"])
        if not inter_csv.exists():
            raise FileNotFoundError(f"Interaction energy CSV not found: {inter_csv}")
        energy_df = _load_inter_energy(inter_csv)
        G = _apply_energy_weights(G_base, energy_df, verbose)
        weight_key = "energy_weight"
    else:
        G = _apply_persistence_weights(G_base, verbose)
        weight_key = "persistence_weight"

    _log(f"Edge weight strategy: {edge_policy} (key='{weight_key}')", verbose)

    # Universe for interface detection
    pdb = Path(cfg["inputs"]["pdb"])
    dcd = Path(cfg["inputs"]["dcd"])
    if not pdb.exists() or not dcd.exists():
        raise FileNotFoundError("PDB or DCD file not found")

    _log("Loading MDAnalysis Universe...", verbose)
    u = mda.Universe(str(pdb), str(dcd))

    frames = _select_frames(u, cfg.get("frames", {}), verbose=verbose)
    if frames:
        u.trajectory[frames[0]]

    chainA = cfg["inputs"]["chains"]["nrp1"]
    chainB = cfg["inputs"]["chains"]["xylt1"]

    # Get source hubs
    topN = int(cfg["pathways"]["n_source_residues"])
    hub_nodes = _top_hub_nodes_from_ranking_csv(ranking_csv, topN)
    _log(f"Source nodes (top {topN} hubs): {len(hub_nodes)}", verbose)

    # Get target nodes (interface)
    tgt_cfg = cfg["pathways"]["targets"]
    target_nodes: List[int] = []

    if str(tgt_cfg.get("type", "interface")).lower() == "interface":
        _log("Identifying interface residues...", verbose)
        cutoff = float(tgt_cfg.get("cutoff", 5.0))
        iface_resids = identify_interface_residues(u, chainA, chainB, cutoff)
        target_nodes = _nodes_from_resids(G, iface_resids)
        _log(f"  → {len(iface_resids)} interface residues → {len(target_nodes)} nodes", verbose)

    # Estimation
    n_pairs = len(hub_nodes) * len(target_nodes)
    _log(f"\n{'─' * 60}", verbose)
    _log(f"PATHWAY SEARCH:", verbose)
    _log(f"  Pairs to process: {len(hub_nodes)} × {len(target_nodes)} = {n_pairs}", verbose)

    n_paths = int(cfg["pathways"]["n_paths_per_source"])
    max_len = int(cfg["pathways"]["max_path_length"])
    _log(f"  Max paths per pair: {n_paths}", verbose)
    _log(f"  Max path length: {max_len}", verbose)

    # Checkpoint configuration
    save_every = cfg["pathways"].get("save_checkpoint_every", 100)
    outdir = Path(cfg["output"]["base_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / "checkpoint_paths.json"

    _log(f"  Timeout per search: 2s", verbose)
    _log(f"  Checkpoint every {save_every} pairs", verbose)
    _log(f"{'─' * 60}\n", verbose)

    # OPTIMIZED SEARCH WITH PROGRESS
    all_paths: List[Dict[str, Any]] = []
    simple_paths_list: List[List[int]] = []

    start_time = time.time()
    pair_count = 0
    found_count = 0

    # Progress bar
    pbar = tqdm(total=n_pairs, desc="Processing pairs", unit="pair",
                disable=not (verbose and TQDM_AVAILABLE), ncols=100)

    try:
        for src in hub_nodes:
            for tgt in target_nodes:
                pair_count += 1

                if src == tgt:
                    pbar.update(1)
                    continue

                # Optimized search with timeout
                paths = _shortest_and_alternatives_fast(
                    G, src, tgt,
                    n_paths=n_paths,
                    max_len=max_len,
                    weight_key=weight_key,
                    max_time_seconds=2.0
                )

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
                    found_count += 1

                pbar.update(1)
                pbar.set_postfix({"paths": found_count})

                # Incremental save
                if pair_count % save_every == 0 and all_paths:
                    _save_checkpoint(all_paths, checkpoint_path)
    finally:
        pbar.close()

    elapsed = time.time() - start_time
    _log(f"\n✓ Search completed in {elapsed:.1f}s", verbose)
    _log(f"  Pathways found: {len(all_paths)}", verbose)
    _log(f"  Average: {len(all_paths) / max(1, pair_count):.2f} paths/pair", verbose)

    # Identify bottlenecks
    _log("\nIdentifying bottlenecks...", verbose)
    thr = float(cfg["pathways"]["bottleneck_threshold"])
    bottlenecks_df = identify_bottleneck_residues(simple_paths_list, G, threshold=thr)
    _log(f"  → {len(bottlenecks_df)} bottleneck residues (freq ≥ {thr:.1%})", verbose)

    # Path importance ranking
    if all_paths:
        imp_df = pd.DataFrame(all_paths).sort_values("score", ascending=True)
    else:
        imp_df = pd.DataFrame(columns=["source", "target", "path", "length", "residues", "score"])

    # Save final outputs
    _log("\nSaving results...", verbose)

    paths_json = outdir / cfg["pathways"]["output"]["pathways"]
    _ensure_parent(paths_json)
    with paths_json.open("w", encoding="utf-8") as f:
        json.dump(all_paths, f, indent=2)

    bott_csv = outdir / cfg["pathways"]["output"]["bottlenecks"]
    _ensure_parent(bott_csv)
    bottlenecks_df.to_csv(bott_csv, index=False)

    imp_csv = outdir / cfg["pathways"]["output"]["importance"]
    _ensure_parent(imp_csv)
    imp_df.to_csv(imp_csv, index=False)

    # Remove checkpoint if exists
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    _log(f"\n{'=' * 80}", verbose)
    _log("RESULTS:", verbose)
    _log(f"  ✓ Pathways:    {paths_json}", verbose)
    _log(f"  ✓ Bottlenecks: {bott_csv}", verbose)
    _log(f"  ✓ Importance:  {imp_csv}", verbose)

    if len(bottlenecks_df) > 0:
        _log(f"\n{'─' * 60}", verbose)
        _log("Top 10 Bottlenecks:", verbose)
        _log(f"{'─' * 60}", verbose)
        print(bottlenecks_df.head(10).to_string(index=False))

    _log(f"{'=' * 80}\n", verbose)

    return dict(
        graph=G,
        paths_json=str(paths_json),
        bottlenecks_csv=str(bott_csv),
        importance_csv=str(imp_csv),
        n_paths=len(all_paths),
        elapsed_seconds=elapsed
    )
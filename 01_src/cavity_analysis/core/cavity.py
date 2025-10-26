# 01_src/cavity_analysis/core/cavity.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import MDAnalysis as mda


# -------------------------------------------------------------------
# Helpers de lectura de resultados previos (archivos → estructuras)
# -------------------------------------------------------------------
def _csv(p: Path):
    return pd.read_csv(p) if p and p.exists() else None

def _json(p: Path):
    if p and p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _npy(p: Path):
    return np.load(p) if p and p.exists() else None

def _pkl(p: Path):
    if p and p.exists():
        import pickle
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def load_previous_results(cfg: Dict[str, Any]) -> Dict[str, Any]:
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
        assign = cjson.get("assignments", cjson)  # compatibilidad
        out["communities"] = {"communities": {int(k): int(v) for k, v in assign.items()}}

    # --- INTEGRATION (módulo 7)
    integ = prev.get("integration", {})
    idir  = Path(integ.get("dir", ""))
    crit  = _csv(idir / integ.get("critical_csv", ""))
    if crit is not None:
        out["integration"] = {"integrated": crit}

    return out


# -------------------------------------------------------------
# Núcleo del análisis (idéntico en intención a tu versión)
# -------------------------------------------------------------
def extract_cavity_residues_data(cavity_residues: List[int],
                                 all_results: Dict[str, Any],
                                 config: Dict[str, Any],
                                 verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print(f"  Extrayendo datos de {len(cavity_residues)} residuos de cavidad...")

    cavity_data = pd.DataFrame({'resid': cavity_residues})

    # 1. Network
    if 'network' in all_results and all_results['network']:
        rnk = all_results['network']['ranking']
        cols = [c for c in ['resid', 'resname', 'segid', 'betweenness', 'eigenvector', 'closeness', 'degree'] if c in rnk.columns]
        cavity_data = cavity_data.merge(rnk[cols], on='resid', how='left')

    # 2. Energy
    if 'energy' in all_results and all_results['energy'] and 'binding_per_residue' in all_results['energy']:
        e = all_results['energy']['binding_per_residue']
        keep = [c for c in ['resid', 'binding_energy'] if c in e.columns]
        cavity_data = cavity_data.merge(e[keep], on='resid', how='left')

    # 3. DCCM (cuenta de pares en alta correlación)
    if 'dccm' in all_results and all_results['dccm'] and 'high_correlations' in all_results['dccm']:
        hi = all_results['dccm']['high_correlations']
        from collections import Counter
        corr_counts = Counter()
        for _, row in hi.iterrows():
            if row['resid1'] in cavity_residues:
                corr_counts[row['resid1']] += 1
            if row['resid2'] in cavity_residues:
                corr_counts[row['resid2']] += 1
        cavity_data['n_high_correlations'] = cavity_data['resid'].map(corr_counts).fillna(0)

    # 4. PCA
    if 'pca' in all_results and all_results['pca']:
        p = all_results['pca']['participation']
        keep = [c for c in ['resid', 'PC1', 'PC2', 'PC3'] if c in p.columns]
        if keep:
            cavity_data = cavity_data.merge(p[keep], on='resid', how='left')

    # 5. Pathways
    if 'pathways' in all_results and all_results['pathways'] and 'bottlenecks' in all_results['pathways']:
        b = all_results['pathways']['bottlenecks']
        keep = [c for c in ['resid', 'frequency', 'appearances'] if c in b.columns]
        cavity_data = cavity_data.merge(b[keep], on='resid', how='left')

    # 6. Communities
    if 'communities' in all_results and all_results['communities'] and 'network' in all_results and all_results['network']:
        assignments = all_results['communities']['communities']  # node_idx -> comm
        ranking = all_results['network']['ranking']
        node_to_resid = dict(zip(ranking['node_idx'], ranking['resid']))
        resid_to_comm = {node_to_resid[n]: c for n, c in assignments.items() if n in node_to_resid}
        cavity_data['community'] = cavity_data['resid'].map(resid_to_comm)

    # 7. Integration (critical_score)
    if 'integration' in all_results and all_results['integration']:
        integ = all_results['integration']['integrated']
        keep = [c for c in ['resid', 'critical_score'] if c in integ.columns]
        cavity_data = cavity_data.merge(integ[keep], on='resid', how='left')

    return cavity_data


def compare_cavity_vs_rest(cavity_data: pd.DataFrame,
                           all_residues_data: pd.DataFrame,
                           metrics: List[str],
                           verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print("\n  Comparando cavidad vs resto de proteína...")

    cavity_resids = set(cavity_data['resid'])
    rest_data = all_residues_data[~all_residues_data['resid'].isin(cavity_resids)].copy()

    comparison: Dict[str, Any] = {}

    for metric in metrics:
        if metric not in cavity_data.columns or metric not in rest_data.columns:
            continue

        cav_vals = pd.to_numeric(cavity_data[metric], errors='coerce').dropna()
        rst_vals = pd.to_numeric(rest_data[metric], errors='coerce').dropna()
        if len(cav_vals) == 0 or len(rst_vals) == 0:
            continue

        cav_mean, cav_std = cav_vals.mean(), cav_vals.std()
        rst_mean, rst_std = rst_vals.mean(), rst_vals.std()
        try:
            stat, pval = stats.mannwhitneyu(cav_vals, rst_vals, alternative='greater')
            signif = bool(pval < 0.05)
        except Exception:
            stat, pval, signif = None, None, False

        fold = float(cav_mean / rst_mean) if rst_mean != 0 else np.inf
        interp = ('Cavidad significativamente mayor' if signif and fold > 1
                  else 'Cavidad significativamente menor' if signif and fold < 1
                  else 'No diferencia significativa')

        comparison[metric] = {
            'cavity_mean': float(cav_mean),
            'cavity_std': float(cav_std),
            'rest_mean': float(rst_mean),
            'rest_std': float(rst_std),
            'fold_change': float(fold),
            'p_value': float(pval) if pval is not None else None,
            'is_significant': signif,
            'interpretation': interp
        }

        if verbose:
            print(f"    {metric}: cav={cav_mean:.3f}±{cav_std:.3f} | rest={rst_mean:.3f}±{rst_std:.3f} | "
                  f"FC={fold:.2f} | p={pval:.3g if pval is not None else 'NA'} {'✅' if signif else '❌'}")

    return comparison


def analyze_cavity_connectivity(cavity_residues: List[int],
                                network_graph: nx.Graph,
                                verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print("\n  Analizando conectividad interna de cavidad...")

    resid_to_node = {}
    for node, data in network_graph.nodes(data=True):
        resid_to_node[data.get('resid')] = node

    cav_nodes = [resid_to_node[r] for r in cavity_residues if r in resid_to_node]
    sub = network_graph.subgraph(cav_nodes).copy()

    n_nodes = sub.number_of_nodes()
    n_edges = sub.number_of_edges()
    density = nx.density(sub) if n_nodes > 1 else 0.0
    n_comp = nx.number_connected_components(sub) if n_nodes > 0 else 0
    connected = bool(n_comp == 1)
    clustering = nx.average_clustering(sub) if n_nodes > 0 else 0.0

    if verbose:
        print(f"    Nodos: {n_nodes} | Aristas: {n_edges} | Densidad: {density:.3f} | "
              f"Componentes: {n_comp} | Conectada: {'Sí' if connected else 'No'} | "
              f"Clustering: {clustering:.3f}")

    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density': float(density),
        'n_components': n_comp,
        'is_fully_connected': connected,
        'avg_clustering': float(clustering)
    }


def analyze_cavity_to_ser612_communication(cavity_residues: List[int],
                                           ser612_resid: int,
                                           pathway_results: Dict[str, Any] | None,
                                           dccm_results: Dict[str, Any] | None,
                                           verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print("\n  Analizando comunicación Cavidad → Ser612...")

    comm: Dict[str, Any] = {}

    # 1) Pathways
    if pathway_results and 'pathways' in pathway_results:
        paths = pathway_results['pathways']
        relevant = []
        for p in paths:
            pr = set(p.get('residues', []))
            if pr & set(cavity_residues) and ser612_resid in pr:
                relevant.append(p)

        comm['n_paths_via_cavity'] = len(relevant)
        cav_in_paths = set()
        for p in relevant:
            cav_in_paths.update(set(p.get('residues', [])) & set(cavity_residues))
        comm['cavity_residues_in_paths'] = list(cav_in_paths)
        comm['pct_cavity_in_paths'] = 100.0 * (len(cav_in_paths) / max(1, len(cavity_residues)))

        if verbose:
            print(f"    Pathways vía cavidad: {comm['n_paths_via_cavity']} | "
                  f"% cavidad en paths: {comm['pct_cavity_in_paths']:.1f}%")

    # 2) DCCM
    if dccm_results and 'dccm' in dccm_results and 'atoms' in dccm_results:
        M = dccm_results['dccm']
        atoms = dccm_results['atoms']  # opcional: si tu DCCM core guardó AtomGroup

        # Si no hay AtomGroup, no podemos mapear resids→índices
        if hasattr(atoms, '__len__'):
            cav_idx = [i for i, a in enumerate(atoms) if a.resid in cavity_residues]
            ser_idx = [i for i, a in enumerate(atoms) if a.resid == ser612_resid]
            if cav_idx and ser_idx:
                vals = [abs(M[i, j]) for i in cav_idx for j in ser_idx]
                if vals:
                    comm['avg_correlation_to_ser612'] = float(np.mean(vals))
                    comm['max_correlation_to_ser612'] = float(np.max(vals))
                    if verbose:
                        print(f"    Correlación promedio: {comm['avg_correlation_to_ser612']:.3f} | "
                              f"máxima: {comm['max_correlation_to_ser612']:.3f}")

    return comm


def _identify_interface_residues(u: mda.Universe, chain1: str, chain2: str, cutoff: float = 5.0) -> List[int]:
    """Versión local (evitamos dependencia cruzada)."""
    from MDAnalysis.analysis import distances
    a1 = u.select_atoms(f"segid {chain1}")
    a2 = u.select_atoms(f"segid {chain2}")
    if a1.n_atoms == 0 or a2.n_atoms == 0:
        return []
    D = distances.distance_array(a1.positions, a2.positions, box=u.dimensions)
    i, j = np.where(D < cutoff)
    res1 = set(a1[i].resids.tolist())
    res2 = set(a2[j].resids.tolist())
    return sorted(res1 | res2)


def analyze_cavity_to_interface_communication(cavity_residues: List[int],
                                              interface_residues: List[int],
                                              pathway_results: Dict[str, Any] | None,
                                              energy_results: Dict[str, Any] | None,
                                              verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print("\n  Analizando comunicación Cavidad → Interfaz NRP1-XYLT1...")

    out: Dict[str, Any] = {}

    # 1) Pathways cavidad→interfaz
    if pathway_results and 'pathways' in pathway_results:
        paths = pathway_results['pathways']
        sel = []
        for p in paths:
            pr = set(p.get('residues', []))
            if (pr & set(cavity_residues)) and (pr & set(interface_residues)):
                sel.append(p)
        out['n_paths_to_interface'] = len(sel)
        if verbose:
            print(f"    Pathways cavidad→interfaz: {len(sel)}")

    # 2) Energía cavidad–interfaz
    if energy_results and 'inter' in energy_results:
        inter = energy_results['inter']
        if all(c in inter.columns for c in ('resid1', 'resid2', 'e_total')):
            filt = inter[
                (inter['resid1'].isin(cavity_residues) & inter['resid2'].isin(interface_residues)) |
                (inter['resid2'].isin(cavity_residues) & inter['resid1'].isin(interface_residues))
            ]
            if len(filt) > 0:
                out['n_interactions_to_interface'] = int(len(filt))
                out['avg_energy_to_interface'] = float(filt['e_total'].mean())
                if verbose:
                    print(f"    Interacciones energéticas: {len(filt)} | "
                          f"E(prom): {out['avg_energy_to_interface']:.2f} kcal/mol")

    return out


def generate_cavity_report(cavity_data: pd.DataFrame,
                           comparison: Dict[str, Any],
                           connectivity: Dict[str, Any],
                           ser612_comm: Dict[str, Any],
                           interface_comm: Dict[str, Any],
                           verdict: Dict[str, Any],
                           config: Dict[str, Any]) -> str:
    r = []
    r.append("="*80)
    r.append("ANÁLISIS DE ALOSTERISMO DE LA CAVIDAD")
    r.append("="*80 + "\n")
    r.append(f"Proyecto: {config['project_name']}")
    r.append(f"Cavidad: {len(cavity_data)} residuos en {config['regions']['cavity']['name']}\n")
    r.append("="*80)
    r.append("1. VEREDICTO FINAL")
    r.append("="*80 + "\n")
    r.append(verdict['verdict'])
    r.append(f"Score: {verdict['score']:.1f}/100\n")
    r.append(verdict['conclusion'] + "\n")
    r.append("Evidencias:")
    for e in verdict['evidences']:
        r.append(f"  {e}")
    r.append("\n" + "="*80)
    r.append("2. COMPARACIÓN CAVIDAD VS RESTO DE PROTEÍNA")
    r.append("="*80)
    for metric, data in comparison.items():
        pv = f"{data['p_value']:.4f}" if data['p_value'] is not None else "N/A"
        r.append(
            f"\n{metric.upper()}:\n"
            f"  Cavidad: {data['cavity_mean']:.3f} ± {data['cavity_std']:.3f}\n"
            f"  Resto:   {data['rest_mean']:.3f} ± {data['rest_std']:.3f}\n"
            f"  Fold change: {data['fold_change']:.2f}x\n"
            f"  p-value: {pv}\n"
            f"  {data['interpretation']}\n"
        )
    r.append("\n" + "="*80)
    r.append("3. CONECTIVIDAD INTERNA")
    r.append("="*80 + "\n")
    r.append(f"Densidad: {connectivity['density']:.3f}")
    r.append(f"Componentes: {connectivity['n_components']}")
    r.append(f"Clustering promedio: {connectivity['avg_clustering']:.3f}")
    r.append(f"¿Totalmente conectada?: {'Sí' if connectivity['is_fully_connected'] else 'No'}\n")
    r.append("="*80)
    r.append("4. COMUNICACIÓN CON SER612")
    r.append("="*80 + "\n")
    if 'n_paths_via_cavity' in ser612_comm:
        r.append(f"Pathways vía cavidad: {ser612_comm['n_paths_via_cavity']}")
        if 'pct_cavity_in_paths' in ser612_comm:
            r.append(f"Residuos cavidad en paths: {ser612_comm.get('cavity_residues_in_paths', [])} "
                     f"({ser612_comm['pct_cavity_in_paths']:.1f}%)")
    if 'avg_correlation_to_ser612' in ser612_comm:
        r.append(f"Correlación promedio: {ser612_comm['avg_correlation_to_ser612']:.3f}")
        r.append(f"Correlación máxima: {ser612_comm['max_correlation_to_ser612']:.3f}\n")
    r.append("="*80)
    r.append("5. COMUNICACIÓN CON INTERFAZ NRP1-XYLT1")
    r.append("="*80 + "\n")
    if 'n_paths_to_interface' in interface_comm:
        r.append(f"Pathways a interfaz: {interface_comm['n_paths_to_interface']}")
    if 'n_interactions_to_interface' in interface_comm:
        r.append(f"Interacciones energéticas: {interface_comm['n_interactions_to_interface']}")
        r.append(f"Energía promedio: {interface_comm['avg_energy_to_interface']:.2f} kcal/mol\n")
    r.append("="*80)
    r.append("6. TOP 10 RESIDUOS MÁS CRÍTICOS DE LA CAVIDAD")
    r.append("="*80 + "\n")
    if 'critical_score' in cavity_data.columns:
        top10 = cavity_data.sort_values('critical_score', ascending=False).head(10)
        cols = [c for c in ['resid', 'resname', 'critical_score', 'betweenness'] if c in top10.columns]
        r.append(top10[cols].to_string(index=False))
    else:
        r.append("No hay 'critical_score' para ordenar.")
    r.append("\n" + "="*80 + "\n")
    return "\n".join(r)


def generate_cavity_importance_verdict(cavity_data: pd.DataFrame,
                                       comparison: Dict[str, Any],
                                       connectivity: Dict[str, Any],
                                       ser612_comm: Dict[str, Any],
                                       interface_comm: Dict[str, Any],
                                       verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print("\n" + "=" * 80)
        print("VEREDICTO FINAL: IMPORTANCIA ALOSTÉRICA DE LA CAVIDAD")
        print("=" * 80)

    evidences: List[str] = []
    score = 0
    max_score = 0

    # 1) Centralidad (20)
    max_score += 20
    if 'betweenness' in comparison:
        fc = comparison['betweenness']['fold_change']
        sig = comparison['betweenness']['is_significant']
        if sig and fc > 1:
            score += 20; evidences.append("✅ Alta centralidad (betweenness significativamente mayor)")
        elif fc > 1:
            score += 10; evidences.append("⚠️ Centralidad elevada (no significativa)")
        else:
            evidences.append("❌ Centralidad no destacable")

    # 2) Energía (20)
    max_score += 20
    if 'binding_energy' in comparison:
        sig = comparison['binding_energy']['is_significant']
        fc  = comparison['binding_energy']['fold_change']
        if sig:
            score += 20; evidences.append("✅ Contribución energética significativa al binding")
        elif abs(fc - 1) > 0.2:
            score += 10; evidences.append("⚠️ Contribución energética moderada")
        else:
            evidences.append("❌ Contribución energética no destacable")

    # 3) DCCM (15)
    max_score += 15
    if 'n_high_correlations' in comparison:
        sig = comparison['n_high_correlations']['is_significant']
        fc  = comparison['n_high_correlations']['fold_change']
        if sig and fc > 1:
            score += 15; evidences.append("✅ Alta participación en correlaciones dinámicas")
        elif fc > 1:
            score += 8; evidences.append("⚠️ Participación moderada en correlaciones")
        else:
            evidences.append("❌ Baja participación en correlaciones")

    # 4) Conectividad (10)
    max_score += 10
    if connectivity['is_fully_connected'] and connectivity['density'] > 0.3:
        score += 10; evidences.append(f"✅ Cavidad conectada (densidad={connectivity['density']:.2f})")
    elif connectivity['is_fully_connected']:
        score += 5; evidences.append(f"⚠️ Conectada pero densidad baja ({connectivity['density']:.2f})")
    else:
        evidences.append(f"❌ Cavidad fragmentada ({connectivity['n_components']} componentes)")

    # 5) Comunicación con Ser612 (20)
    max_score += 20
    if ser612_comm.get('n_paths_via_cavity', 0) > 0:
        if ser612_comm.get('pct_cavity_in_paths', 0) > 30:
            score += 20; evidences.append("✅ Fuerte comunicación con Ser612")
        else:
            score += 10; evidences.append("⚠️ Comunicación moderada con Ser612")
    else:
        evidences.append("❌ Sin comunicación detectable con Ser612")

    # 6) Comunicación con interfaz (15)
    max_score += 15
    if interface_comm.get('n_paths_to_interface', 0) > 0:
        score += 15; evidences.append("✅ Comunicación con la interfaz NRP1–XYLT1")
    else:
        evidences.append("❌ No se detectó comunicación con la interfaz")

    final = (score / max_score * 100.0) if max_score > 0 else 0.0
    if final >= 70:
        verdict = "✅ SITIO ALOSTÉRICO CRÍTICO"
        concl = "Evidencia fuerte de rol alostérico funcional de la cavidad."
    elif final >= 50:
        verdict = "⚠️ POSIBLE SITIO ALOSTÉRICO"
        concl = "Evidencia moderada; requiere validación experimental."
    else:
        verdict = "❌ NO ES SITIO ALOSTÉRICO CRÍTICO"
        concl = "La evidencia no soporta un rol alostérico crítico."

    if verbose:
        print(f"\n{verdict}\nScore: {final:.1f}/100")

    return {
        "verdict": verdict,
        "score": float(final),
        "max_score": int(max_score),
        "evidences": evidences,
        "conclusion": concl
    }


def run_cavity_allostery_analysis(u: mda.Universe,
                                  config: Dict[str, Any],
                                  all_results: Dict[str, Any],
                                  verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print("=" * 80)
        print("MÓDULO 9: CAVITY ALLOSTERY ANALYSIS")
        print("=" * 80)

    # Cavidad y Ser612
    if 'cavity' not in config.get('regions', {}):
        if verbose:
            print("  ⚠️  No se definió 'regions.cavity' en el YAML.")
        return {}

    cavity_residues = list(map(int, config['regions']['cavity']['residues']))
    ser612_resid = int(config['regions']['ser612']['residue'])

    # 1) Tabla de cavidad
    if verbose:
        print("\n1) Extrayendo datos de cavidad …")
    cavity_data = extract_cavity_residues_data(cavity_residues, all_results, config, verbose)

    # 2) Comparación
    if verbose:
        print("\n2) Comparando cavidad vs resto …")
    if 'integration' not in all_results or 'integrated' not in all_results['integration']:
        raise RuntimeError("Falta 'integration.integrated' (módulo 7) para comparar contra todo el proteoma.")
    all_df = all_results['integration']['integrated']
    metrics = ['betweenness', 'binding_energy', 'n_high_correlations', 'PC1', 'frequency', 'critical_score']
    comparison = compare_cavity_vs_rest(cavity_data, all_df, metrics, verbose)

    # 3) Conectividad interna
    if verbose:
        print("\n3) Conectividad interna …")
    if 'network' not in all_results or 'network' not in all_results['network']:
        raise RuntimeError("Falta 'network.network' (módulo 1) con el grafo para conectividad.")
    G = all_results['network']['network']
    connectivity = analyze_cavity_connectivity(cavity_residues, G, verbose)

    # 4) Comunicación con Ser612
    if verbose:
        print("\n4) Comunicación con Ser612 …")
    ser612_comm = analyze_cavity_to_ser612_communication(
        cavity_residues, ser612_resid,
        all_results.get('pathways'), all_results.get('dccm'), verbose
    )

    # 5) Comunicación con interfaz
    if verbose:
        print("\n5) Comunicación con interfaz NRP1–XYLT1 …")
    chain1 = config['inputs']['chains']['nrp1']
    chain2 = config['inputs']['chains']['xylt1']
    iface_res = _identify_interface_residues(u, chain1, chain2, cutoff=5.0)
    interface_comm = analyze_cavity_to_interface_communication(
        cavity_residues, iface_res,
        all_results.get('pathways'), all_results.get('energy'), verbose
    )

    # 6) Veredicto
    verdict = generate_cavity_importance_verdict(
        cavity_data, comparison, connectivity, ser612_comm, interface_comm, verbose
    )

    # 7) Salvar outputs
    out_base = Path(config['output']['base_dir'])
    (out_base / "tables").mkdir(parents=True, exist_ok=True)

    cavity_csv = out_base / "tables" / "9_cavity_residues_data.csv"
    cavity_data.to_csv(cavity_csv, index=False)

    comp_json = out_base / "tables" / "9_cavity_comparison.json"
    with open(comp_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    verdict_json = out_base / "tables" / "9_cavity_verdict.json"
    with open(verdict_json, "w", encoding="utf-8") as f:
        json.dump({
            "verdict": verdict,
            "connectivity": connectivity,
            "ser612_communication": ser612_comm,
            "interface_communication": interface_comm
        }, f, indent=2)

    report_txt = out_base / "tables" / "9_cavity_allostery_report.txt"
    report = generate_cavity_report(cavity_data, comparison, connectivity, ser612_comm, interface_comm, verdict, config)
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(report)

    if verbose:
        print(f"\n✅ Resultados guardados en {out_base}/tables/9_cavity_*")

    return {
        "cavity_data": cavity_data,
        "comparison": comparison,
        "connectivity": connectivity,
        "ser612_communication": ser612_comm,
        "interface_communication": interface_comm,
        "verdict": verdict
    }

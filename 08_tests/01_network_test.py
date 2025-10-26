# 08_tests/01_network_fulltest.py
"""
Test integral del PASO 01 (network):
- Valida existencia y consistencia de 3 outputs: .pkl (grafo), .csv (ranking), .json (hubs)
- Calcula métricas globales y por cadena (A=NRP1, B=XYLT1, U=UDP)
- Verifica concordancia hubs_json vs ranking.csv
- Analiza contactos de interfaz A<->B
- Genera plots en 05_results/<tag>/01_network/plots/
- Escribe un resumen en test_report.txt
- Exporta nodes.csv y edges.csv

Uso:
    python 08_tests/01_network_fulltest.py --config 03_configs/analyses/01_network.yaml
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import pickle
import math
import warnings

import yaml
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_graph(path_guess: Path) -> nx.Graph:
    """Carga pickle del grafo. Si el nombre del YAML no es .pkl pero existe .pkl, lo usa."""
    if path_guess.suffix.lower() != ".pkl":
        alt = path_guess.with_suffix(".pkl")
        if alt.exists():
            path_guess = alt
    if not path_guess.exists():
        # fallback si alguien lo dejó como .gpickle
        alt2 = path_guess.with_suffix(".gpickle")
        if alt2.exists():
            path_guess = alt2
    if not path_guess.exists():
        raise FileNotFoundError(f"No encuentro el grafo: {path_guess}")
    with open(path_guess, "rb") as fh:
        G = pickle.load(fh)
    if not isinstance(G, nx.Graph):
        raise TypeError("El objeto cargado no es un networkx.Graph.")
    return G, path_guess


def basic_graph_metrics(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degs = [d for _, d in G.degree()]
    avg_deg = float(np.mean(degs)) if degs else 0.0
    density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0

    # componentes
    comps = list(nx.connected_components(G))
    n_components = len(comps)
    largest = max((len(c) for c in comps), default=0)

    # pesos (persistencia)
    ws = [float(d.get("weight", 0.0)) for _, _, d in G.edges(data=True)]
    wmean = float(np.mean(ws)) if ws else 0.0
    wmed = float(np.median(ws)) if ws else 0.0

    return dict(
        n_nodes=n, n_edges=m, avg_degree=avg_deg, density=density,
        n_components=n_components, largest_component=largest,
        weight_mean=wmean, weight_median=wmed
    )


def check_ranking_df(df: pd.DataFrame) -> list[str]:
    required = ["node_idx", "resid", "resname", "segid", "protein"]
    # al menos 1 métrica
    metrics = [c for c in df.columns if c in ["betweenness", "eigenvector", "closeness", "degree"]]
    missing = [c for c in required if c not in df.columns]
    issues = []
    if missing:
        issues.append(f"Faltan columnas obligatorias: {missing}")
    if not metrics:
        issues.append("No se encontró ninguna columna de centralidad (betweenness/eigenvector/closeness/degree).")
    if df.empty:
        issues.append("El ranking está vacío.")
    if df.isna().sum().sum() > 0:
        issues.append("Hay valores NaN en ranking.csv.")
    return issues


def compare_hubs_json_vs_df(hubs_json: dict, df: pd.DataFrame, top_n: int = 50) -> dict:
    """
    Compara las listas 'hubs' (por métrica) contra el top-N de ranking.csv (mismo criterio por métrica).
    Devuelve acuerdo (%) por métrica.
    """
    out = {}
    # Para comparar, usamos node_idx como id
    for metric, items in hubs_json.items():
        # normalizar nombre de métrica del ranking
        metric_name = metric
        if metric_name not in df.columns and metric_name == "betweenness_centrality":
            metric_name = "betweenness"
        if metric_name not in df.columns and metric_name == "eigenvector_centrality":
            metric_name = "eigenvector"

        if metric_name not in df.columns:
            out[metric] = {"status": "skip", "reason": f"'{metric_name}' no está en ranking.csv"}
            continue

        df_sorted = df.sort_values(metric_name, ascending=False)
        top_df = set(df_sorted.head(top_n)["node_idx"].tolist())
        top_js = set(int(x["node_idx"]) for x in items[:top_n])

        inter = len(top_df & top_js)
        union = len(top_df | top_js) if (top_df or top_js) else 1
        jacc = inter / union
        out[metric] = {
            "overlap_count": inter,
            "topN": top_n,
            "overlap_ratio": round(jacc, 3)
        }
    return out


def interface_analysis(G: nx.Graph) -> dict:
    """
    Analiza interfaz A<->B:
      - porcentaje de aristas cruzadas
      - top nodos por grado de interfaz
    """
    # etiqueta de cadena por nodo
    seg = nx.get_node_attributes(G, "segid")

    cross_edges = []
    for u, v, d in G.edges(data=True):
        su, sv = seg.get(u, ""), seg.get(v, "")
        if su and sv and su != sv and {su, sv} <= {"A", "B"}:
            cross_edges.append((u, v, d))
    cross_ratio = len(cross_edges) / max(1, G.number_of_edges())

    # grado de interfaz por nodo
    cross_deg = {}
    for u, v, d in cross_edges:
        cross_deg[u] = cross_deg.get(u, 0) + 1
        cross_deg[v] = cross_deg.get(v, 0) + 1

    # top 20 por grado de interfaz
    top_nodes = sorted(cross_deg.items(), key=lambda kv: kv[1], reverse=True)[:20]
    # empaquetar con info de resid
    rows = []
    for idx, val in top_nodes:
        nd = G.nodes[idx]
        rows.append(dict(
            node_idx=int(idx),
            interfacial_degree=int(val),
            resid=int(nd.get("resid", -1)),
            resname=str(nd.get("resname", "")),
            segid=str(nd.get("segid", "")),
        ))
    return {
        "cross_edges_ratio": round(cross_ratio, 4),
        "top20_interfacial": rows
    }


def plot_all(G: nx.Graph, df: pd.DataFrame, outplots: Path) -> list[Path]:
    """
    Genera plots sin seaborn (solo matplotlib) y devuelve las rutas guardadas.
    """
    ensure_dir(outplots)
    saved = []

    # --- histograma de grados
    fig = plt.figure()
    degs = [d for _, d in G.degree()]
    plt.hist(degs, bins=min(50, max(10, int(math.sqrt(len(degs))))) )
    plt.title("Distribución de grado")
    plt.xlabel("grado")
    plt.ylabel("conteo")
    p = outplots / "degree_hist.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # --- histograma de pesos/persistencia
    ws = [float(d.get("weight", 0.0)) for _, _, d in G.edges(data=True)]
    fig = plt.figure()
    plt.hist(ws, bins=30)
    plt.title("Distribución de persistencia (peso de arista)")
    plt.xlabel("persistencia [0..1]")
    plt.ylabel("conteo")
    p = outplots / "weight_hist.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # --- scatter: betweenness vs degree
    if "betweenness" in df.columns and "degree" in df.columns:
        fig = plt.figure()
        plt.scatter(df["degree"].values, df["betweenness"].values, s=10)
        plt.title("Betweenness vs Degree")
        plt.xlabel("degree")
        plt.ylabel("betweenness")
        p = outplots / "scatter_degree_vs_betweenness.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # --- heatmap de correlación entre centralidades (si hay >=2)
    centr_cols = [c for c in ["betweenness", "eigenvector", "closeness", "degree"] if c in df.columns]
    if len(centr_cols) >= 2:
        corr = df[centr_cols].corr().values
        fig = plt.figure()
        im = plt.imshow(corr, interpolation="nearest")
        plt.xticks(range(len(centr_cols)), centr_cols, rotation=45, ha="right")
        plt.yticks(range(len(centr_cols)), centr_cols)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Correlación entre centralidades")
        p = outplots / "centralities_correlation.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # --- barras: top 15 por betweenness
    if "betweenness" in df.columns:
        dfb = df.sort_values("betweenness", ascending=False).head(15).copy()
        labels = [f"{r.segid}{int(r.resid)}-{r.resname}" for _, r in dfb.iterrows()]
        fig = plt.figure(figsize=(10, 4))
        plt.bar(range(len(dfb)), dfb["betweenness"].values)
        plt.xticks(range(len(dfb)), labels, rotation=45, ha="right")
        plt.title("Top 15 por betweenness")
        plt.xlabel("residuo")
        plt.ylabel("betweenness")
        p = outplots / "top15_betweenness.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    return saved


def main():
    ap = argparse.ArgumentParser(description="Full test del PASO 01 (network) con plots y reporte.")
    ap.add_argument("--config", required=True, help="Ruta al YAML (03_configs/analyses/01_network.yaml)")
    args = ap.parse_args()

    PROJ_ROOT = Path(__file__).resolve().parents[1]
    cfg_path = (PROJ_ROOT / args.config).resolve() if not args.config.startswith(str(PROJ_ROOT)) else Path(args.config)
    cfg = load_yaml(cfg_path)

    base_dir = (PROJ_ROOT / cfg["output"]["base_dir"]).resolve()
    ensure_dir(base_dir)

    # rutas declaradas en YAML (el core pudo haber forzado .pkl; aquí lo resolvemos)
    net_decl = base_dir / cfg["global_network"]["output"]["network_graph"]
    rank_path = base_dir / cfg["global_network"]["output"]["centrality_ranking"]
    hubs_path = base_dir / cfg["global_network"]["output"]["hub_residues"]

    # carga de archivos
    G, net_path = load_graph(net_decl)
    if not rank_path.exists():
        raise FileNotFoundError(f"No encuentro ranking: {rank_path}")
    if not hubs_path.exists():
        raise FileNotFoundError(f"No encuentro hubs json: {hubs_path}")

    df = pd.read_csv(rank_path)
    hubs = json.loads(hubs_path.read_text(encoding="utf-8"))

    # 1) métricas del grafo
    gstats = basic_graph_metrics(G)

    # 2) checks de ranking
    rank_issues = check_ranking_df(df)

    # 3) resumen por cadenas (A/B/U)
    by_chain = df["segid"].value_counts().to_dict()

    # 4) comparación hubs_json vs ranking
    top_n = int(cfg["global_network"]["top_n_hubs"])
    hubs_match = compare_hubs_json_vs_df(hubs, df, top_n=top_n)

    # 5) análisis de interfaz A<->B
    inter = interface_analysis(G)

    # 6) exports + plots
    outplots = base_dir / "plots"
    ensure_dir(outplots)

    # export nodos/aristas para Gephi/Cytoscape
    nodes_csv = base_dir / "nodes.csv"
    edges_csv = base_dir / "edges.csv"
    pd.DataFrame([{"node_idx": n, **d} for n, d in G.nodes(data=True)]).to_csv(nodes_csv, index=False)
    pd.DataFrame([{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)]).to_csv(edges_csv, index=False)

    plot_paths = plot_all(G, df, outplots)

    # 7) reporte
    report = []
    report.append("TEST REPORTE — PASO 01 (network)")
    report.append("=" * 72)
    report.append(f"Grafo: {net_path}")
    report.append(f"Ranking: {rank_path}")
    report.append(f"Hubs JSON: {hubs_path}\n")

    report.append("MÉTRICAS DEL GRAFO")
    for k, v in gstats.items():
        report.append(f"  - {k}: {v}")

    report.append("\nRANKING — sanity checks")
    if rank_issues:
        for x in rank_issues:
            report.append(f"  ⚠ {x}")
    else:
        report.append("  OK: columnas y datos válidos")

    report.append("\nRESUMEN POR CADENA (segid)")
    for k, v in by_chain.items():
        report.append(f"  - {k}: {v}")

    report.append("\nCONCORDANCIA hubs.json vs ranking.csv (top-N por métrica)")
    for m, info in hubs_match.items():
        if info.get("status") == "skip":
            report.append(f"  - {m}: skip ({info.get('reason')})")
        else:
            report.append(f"  - {m}: overlap={info['overlap_count']}/{info['topN']}  (Jaccard={info['overlap_ratio']})")

    report.append("\nINTERFAZ A<->B")
    report.append(f"  - porcentaje de aristas cruzadas: {round(100*inter['cross_edges_ratio'],2)}%")
    report.append(f"  - top 10 nodos interfaciales:")
    for row in inter["top20_interfacial"][:10]:
        report.append(f"      {row['segid']}{row['resid']} {row['resname']}  deg_if={row['interfacial_degree']} (node_idx={row['node_idx']})")

    report.append("\nPLOTS Y EXPORTS")
    report.append(f"  - nodes.csv: {nodes_csv}")
    report.append(f"  - edges.csv: {edges_csv}")
    for p in plot_paths:
        report.append(f"  - plot: {p}")

    report_txt = base_dir / "test_report.txt"
    report_txt.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print(f"\n✅ Reporte escrito en: {report_txt}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()

# 08_tests/09_cavity_test.py
from __future__ import annotations
import sys
from pathlib import Path
import yaml
import MDAnalysis as mda

# importar core
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "01_src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cavity_analysis.core.cavity import load_previous_results, run_cavity_allostery_analysis

def main():
    yaml_path = ROOT / "03_configs" / "analyses" / "09_cavity.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No encuentro YAML: {yaml_path}")

    cfg = yaml.safe_load(open(yaml_path, "r", encoding="utf-8"))

    pdb = cfg["inputs"]["pdb"]
    dcd = cfg["inputs"].get("dcd", None)

    print(f"[TEST] PDB: {pdb}")
    if dcd: print(f"[TEST] DCD: {dcd}")

    u = mda.Universe(pdb, dcd) if (dcd and Path(dcd).exists()) else mda.Universe(pdb)

    prev = load_previous_results(cfg)

    print("="*80)
    print("TEST: MÓDULO 9 — CAVITY ALLOSTERY")
    print("="*80)

    res = run_cavity_allostery_analysis(u, cfg, prev, verbose=True)

    # Sanity checks
    out_base = Path(cfg["output"]["base_dir"])
    expected = [
        out_base / "tables" / "9_cavity_residues_data.csv",
        out_base / "tables" / "9_cavity_comparison.json",
        out_base / "tables" / "9_cavity_verdict.json",
        out_base / "tables" / "9_cavity_allostery_report.txt",
    ]
    print("\n[TEST] Archivos esperados:")
    for p in expected:
        print(f" - {p}  [{'OK' if p.exists() else 'FALTA'}]")

    if res and "verdict" in res:
        print(f"\n[TEST] Veredicto: {res['verdict']['verdict']} | Score: {res['verdict']['score']:.1f}/100")

    if res and "cavity_data" in res:
        cd = res["cavity_data"]
        if not cd.empty and "critical_score" in cd.columns:
            top = cd.sort_values("critical_score", ascending=False).head(10)
            cols = [c for c in ["resid", "resname", "critical_score", "betweenness"] if c in top.columns]
            print("\n[TEST] Top 10 cavidad por critical_score:")
            print(top[cols].to_string(index=False))

if __name__ == "__main__":
    main()

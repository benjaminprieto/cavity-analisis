# 08_tests/08_ser612_test.py
from __future__ import annotations
import sys, json
from pathlib import Path
import yaml
import pandas as pd
import MDAnalysis as mda

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "01_src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cavity_analysis.core.ser612 import run_ser612_analysis

def main():
    yaml_path = ROOT / "03_configs" / "analyses" / "08_ser612.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No encuentro YAML: {yaml_path}")

    cfg = yaml.safe_load(open(yaml_path, "r", encoding="utf-8"))
    pdb = cfg["inputs"]["pdb"]
    dcd = cfg["inputs"].get("dcd", None)

    print("="*80)
    print("TEST: MÓDULO 8 — SER612 (FreeSASA→MDTraj→Fallback)")
    print("="*80)
    print(f"[TEST] PDB: {pdb}")
    if dcd: print(f"[TEST] DCD: {dcd}")

    u = mda.Universe(pdb, dcd) if (dcd and Path(dcd).exists()) else mda.Universe(pdb)
    res = run_ser612_analysis(u, cfg, verbose=True)

    out_base = Path(cfg["output"]["base_dir"])
    dist_csv = out_base / "tables" / "8_ser612_distance.csv"
    sasa_csv = out_base / "tables" / "8_ser612_sasa.csv"
    acc_json = out_base / "tables" / "8_ser612_accessibility.json"

    print("\n[TEST] Archivos esperados:")
    for p in [dist_csv, sasa_csv, acc_json]:
        print(f" - {p} [{'OK' if p.exists() else 'FALTA'}]")

    if dist_csv.exists():
        d = pd.read_csv(dist_csv)
        print("\n[TEST] Distancias (head):")
        print(d.head(5).to_string(index=False))
        assert {"frame","time_ps","distance_A"} <= set(d.columns)

    if sasa_csv.exists():
        s = pd.read_csv(sasa_csv)
        print("\n[TEST] SASA/Accesibilidad (head):")
        print(s.head(5).to_string(index=False))
        assert {"frame","time_ps","sasa_A2"} <= set(s.columns)

    if acc_json.exists():
        acc = json.loads(acc_json.read_text(encoding="utf-8"))
        print("\n[TEST] Accesibilidad resumen:")
        for k in ["glycosylation_score","state","avg_distance_A","avg_sasa_A2",
                  "pct_favorable_distance","pct_favorable_sasa","pct_favorable_both"]:
            if k in acc:
                print(f"  {k}: {acc[k]}")
        assert "glycosylation_score" in acc

if __name__ == "__main__":
    main()

# 02_scripts/08_ser612.py
from __future__ import annotations
import argparse, sys
from pathlib import Path
import yaml
import MDAnalysis as mda

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "01_src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cavity_analysis.core.ser612 import run_ser612_analysis

def main():
    ap = argparse.ArgumentParser(description="Módulo 8: Ser612 Analysis (FreeSASA→MDTraj→Fallback)")
    ap.add_argument("--config", required=True, help="Ruta a 03_configs/analyses/08_ser612.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"No existe YAML: {cfg_path}")

    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    verbose = bool(cfg.get("run", {}).get("verbose", True))

    pdb = cfg["inputs"]["pdb"]
    dcd = cfg["inputs"].get("dcd", None)

    print(f"[INFO] PDB: {pdb}")
    if dcd:
        print(f"[INFO] DCD: {dcd}")

    u = mda.Universe(pdb, dcd) if (dcd and Path(dcd).exists()) else mda.Universe(pdb)
    run_ser612_analysis(u, cfg, verbose=verbose)

if __name__ == "__main__":
    main()

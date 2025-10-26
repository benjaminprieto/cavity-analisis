# 02_scripts/09_cavity.py
from __future__ import annotations
import argparse, sys
from pathlib import Path
import yaml
import MDAnalysis as mda

# permitir importar el core
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "01_src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cavity_analysis.core.cavity import (
    load_previous_results,
    run_cavity_allostery_analysis,
)

def main():
    ap = argparse.ArgumentParser(description="Módulo 9: Cavity Allostery CLI")
    ap.add_argument("--config", required=True, help="Ruta a 03_configs/analyses/09_cavity.yaml")
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

    if dcd and Path(dcd).exists():
        u = mda.Universe(pdb, dcd)
    else:
        u = mda.Universe(pdb)

    print("[INFO] Cargando resultados previos (módulos 1–7)…")
    prev = load_previous_results(cfg)

    run_cavity_allostery_analysis(u, cfg, prev, verbose=verbose)

if __name__ == "__main__":
    main()

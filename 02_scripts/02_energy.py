# 02_scripts/02_energy.py
import argparse
from pathlib import Path
import sys

# habilitar import local si 01_src no es Sources Root
proj_root = Path(__file__).resolve().parents[1]
src_dir = proj_root / "01_src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from cavity_analysis.core.energy import run_energy_from_yaml

def main():
    ap = argparse.ArgumentParser(description="PASO 02: Energy decomposition (CLI)")
    ap.add_argument("--config", required=True, help="03_configs/analyses/02_energy.yaml")
    args = ap.parse_args()
    run_energy_from_yaml((proj_root / args.config) if not Path(args.config).is_absolute() else Path(args.config))

if __name__ == "__main__":
    main()

#python 02_scripts/02_energy.py --config 03_configs/analyses/02_energy.yaml

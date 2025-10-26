# 02_scripts/01_network.py
import argparse
from pathlib import Path
import sys

# habilitar imports desde 01_src si no marcaste Sources Root en PyCharm
proj_root = Path(__file__).resolve().parents[1]
src_dir = proj_root / "01_src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from cavity_analysis.core.network import run_global_network_from_yaml

def main():
    ap = argparse.ArgumentParser(description="PASO 01: Global contact network (CLI).")
    ap.add_argument("--config", required=True, help="Ruta a YAML (03_configs/analyses/01_network.yaml)")
    args = ap.parse_args()
    run_global_network_from_yaml(Path(args.config))

if __name__ == "__main__":
    main()

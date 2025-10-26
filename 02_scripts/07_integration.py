# 02_scripts/07_integration.py
import argparse
from pathlib import Path
import sys

proj_root = Path(__file__).resolve().parents[1]
src_dir = proj_root / "01_src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from cavity_analysis.core.integration import run_integration_from_yaml

def main():
    ap = argparse.ArgumentParser(description="PASO 07: Integration (CLI)")
    ap.add_argument("--config", required=True, help="03_configs/analyses/07_integration.yaml")
    args = ap.parse_args()

    cfg_path = (proj_root / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    run_integration_from_yaml(cfg_path)

if __name__ == "__main__":
    main()

#python 02_scripts/07_integration.py --config 03_configs/analyses/07_integration.yaml

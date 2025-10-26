# 02_scripts/04_pca.py
import argparse
from pathlib import Path
import sys

proj_root = Path(__file__).resolve().parents[1]
src_dir = proj_root / "01_src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from cavity_analysis.core.pca import run_pca_from_yaml

def main():
    ap = argparse.ArgumentParser(description="PASO 04: PCA (CLI)")
    ap.add_argument("--config", required=True, help="03_configs/analyses/04_pca.yaml")
    args = ap.parse_args()
    yaml_path = (proj_root / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    run_pca_from_yaml(yaml_path)

if __name__ == "__main__":
    main()

#python 02_scripts/04_pca.py --config 03_configs/analyses/04_pca.yaml
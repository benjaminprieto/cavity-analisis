# 02_scripts/05a_pathways.py
"""
CLI script for pathway analysis.
Simply loads the YAML config and calls the core module.
"""
import argparse
from pathlib import Path
import sys

proj_root = Path(__file__).resolve().parents[1]
src_dir = proj_root / "01_src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from cavity_analysis.core.pathways_optimized import run_pathways_from_yaml


def main():
    ap = argparse.ArgumentParser(
        description="STEP 05a: Pathway Analysis (Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python 02_scripts/05a_pathways.py --config 03_configs/analyses/05a_pathways.yaml
        """
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file"
    )
    args = ap.parse_args()

    # Resolve absolute or relative path
    yaml_path = Path(args.config)
    if not yaml_path.is_absolute():
        yaml_path = proj_root / yaml_path

    if not yaml_path.exists():
        print(f"❌ ERROR: Config file not found: {yaml_path}")
        sys.exit(1)

    # Run analysis
    try:
        results = run_pathways_from_yaml(yaml_path)
        print(f"\n✅ Analysis completed successfully")
        print(f"   Pathways found: {results['n_paths']}")
        print(f"   Elapsed time: {results['elapsed_seconds']:.1f}s")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR during analysis:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
python 02_scripts/05a_pathways.py --config 03_configs/analyses/05a_pathways.yaml
"""

import argparse
from pathlib import Path
import sys
from datetime import datetime
import os
from RoadGuardianAI.batch.runner import BatchPredictor
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def parse_args():
    p = argparse.ArgumentParser(description="Run batch predictions for all segments")
    p.add_argument("--ts", type=str, default=None, help="ISO timestamp (UTC) for window start. Defaults to current UTC hour.")
    p.add_argument("--out", type=str, default=None, help="Output path. Defaults to out/batch_predictions_<ts>.parquet")
    p.add_argument("--fmt", type=str, default="parquet", choices=["parquet", "csv", "json"], help="Output format")
    p.add_argument("--cfg", type=str, default="config/config.yml", help="Path to config file")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    p.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress prints")
    return p.parse_args()

def main():
    args = parse_args()
    bp = BatchPredictor(cfg_path=args.cfg)
    out_path = bp.run_once(ts=args.ts, out_path=args.out, fmt=args.fmt, overwrite=args.overwrite, progress=args.progress)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()

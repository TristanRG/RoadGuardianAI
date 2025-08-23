import argparse
from pathlib import Path
import sys
from datetime import datetime
import os
import traceback

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from RoadGuardianAI.batch.runner import BatchPredictor
import pandas as pd
from RoadGuardianAI.utils.db import save_predictions

def parse_args():
    p = argparse.ArgumentParser(description="Run batch predictions for all segments")
    p.add_argument("--ts", type=str, default=None, help="ISO timestamp (UTC) for window start. Defaults to current UTC hour.")
    p.add_argument("--out", type=str, default=None, help="Output path. Defaults to out/batch_predictions_<ts>.parquet")
    p.add_argument("--fmt", type=str, default="parquet", choices=["parquet", "csv", "json"], help="Output format")
    p.add_argument("--cfg", type=str, default="config/config.yml", help="Path to config file")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    p.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress prints")
    return p.parse_args()

def _read_output_to_df(path: Path, fmt: str) -> pd.DataFrame:
    fmt = fmt.lower()
    if fmt in ("parquet", "pq"):
        return pd.read_parquet(path)
    elif fmt in ("csv", "txt"):
        return pd.read_csv(path)
    elif fmt == "json":
        return pd.read_json(path, orient="records", lines=False)
    else:
        raise ValueError("Unsupported format: " + fmt)

def main():
    args = parse_args()
    bp = BatchPredictor(cfg_path=args.cfg)
    out_path = bp.run_once(ts=args.ts, out_path=args.out, fmt=args.fmt, overwrite=args.overwrite, progress=args.progress)
    print("Wrote:", out_path)

    try:
        df = _read_output_to_df(Path(out_path), args.fmt)
    except Exception as e:
        print("Failed reading output file for DB save:", e)
        traceback.print_exc()
        raise

    try:
        n = save_predictions(df)
        print(f"Saved {n} rows to DB.")
    except Exception as e:
        print("Failed to save predictions to DB:", e)
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import math
import time
import pandas as pd
import numpy as np
from ..api.model_serving import load_config, ModelServer
from ..api.metrics import metrics_response


class BatchPredictor:
    def __init__(self, cfg_path: Optional[str] = None, cfg: Optional[dict] = None):
        if cfg is None:
            self.cfg = load_config(cfg_path or "config/config.yml")
        else:
            self.cfg = cfg
        self.server = ModelServer(self.cfg)
        self.batch_size = int(self.cfg["api"].get("batch_size", 2000))

    @staticmethod
    def _normalize_ts(ts_in: Optional[str]) -> datetime:
        if ts_in is None:
            ts = datetime.utcnow()
        else:
            ts = pd.to_datetime(ts_in)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None) if hasattr(ts, "tz_convert") else ts.tz_localize(None)
        ts = ts.replace(minute=0, second=0, microsecond=0)
        return ts

    def run_once(
        self,
        ts: Optional[str] = None,
        out_path: Optional[str] = None,
        fmt: str = "parquet",
        overwrite: bool = False,
        progress: bool = True,
    ) -> Path:
        ts_use = self._normalize_ts(ts)
        segs = self.server.list_all_segments_from_history()
        if not segs:
            raise RuntimeError("No segment history files found in history_dir: " + str(self.server.history_dir))

        results = []
        start = time.time()
        total = len(segs)
        for i in range(0, total, self.batch_size):
            batch = segs[i : i + self.batch_size]
            df_feats = self.server.assemble_features_batch(batch, ts_use)
            seg_ids = df_feats["segment_id"].tolist()
            X = df_feats.drop(columns=["segment_id"])
            probs = self.server.predict_proba_vect(X)
            for sid, p in zip(seg_ids, probs):
                results.append((sid, float(p)))
            if progress:
                done = min(i + self.batch_size, total)
                pct = 100.0 * done / total
                print(f"Predicted {done}/{total} ({pct:.1f}%)", end="\r")

        elapsed = time.time() - start
        print(f"\nPrediction pass done in {elapsed:.1f}s, rows: {len(results)}")

        df_out = pd.DataFrame(results, columns=["segment_id", "probability"])
        df_out["window_start"] = pd.to_datetime(ts_use)
        df_out["probability_pct"] = (df_out["probability"] * 100.0).round(4).astype(str) + "%"


        hist_rows = []

        hist_map = self.server.read_history_for_segments(df_out["segment_id"].tolist(), ts_use)
        lats = []
        lons = []
        seg_te_list = []
        sev1 = []
        sev6 = []
        tot6 = []
        for sid in df_out["segment_id"].tolist():
            hr = hist_map.get(sid, {}) or {}
            lats.append(hr.get("LATITUDE"))
            lons.append(hr.get("LONGITUDE"))
            seg_te_list.append(self.server.seg_te_map.get(sid, self.server.global_seg_te))
            sev1.append(int(hr.get("sev_1h", 0)))
            sev6.append(int(hr.get("sev_6h", 0)))
            tot6.append(int(hr.get("tot_6h", 0)))

        df_out["lat"] = lats
        df_out["lon"] = lons
        df_out["seg_te"] = seg_te_list
        df_out["sev_1h"] = sev1
        df_out["sev_6h"] = sev6
        df_out["tot_6h"] = tot6

        cols = ["segment_id", "window_start", "probability", "probability_pct", "seg_te", "lat", "lon", "sev_1h", "sev_6h", "tot_6h"]
        df_out = df_out[cols]

        if out_path is None:
            stamp = ts_use.strftime("%Y%m%dT%H%M%SZ")
            out_dir = Path("out")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"batch_predictions_{stamp}.{fmt}"
        else:
            out_path = Path(out_path)

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Output file exists: {out_path} (set overwrite=True to replace)")

        if fmt.lower() in ("parquet", "pq"):
            try:
                df_out.to_parquet(out_path, index=False)
            except Exception:
                df_out.to_parquet(out_path, index=False, engine="pyarrow")
        elif fmt.lower() in ("csv", "txt"):
            df_out.to_csv(out_path, index=False)
        elif fmt.lower() in ("json",):
            df_out.to_json(out_path, orient="records", lines=False)
        else:
            raise ValueError("Unsupported format: " + fmt)

        print("Saved batch output to:", out_path)
        return out_path

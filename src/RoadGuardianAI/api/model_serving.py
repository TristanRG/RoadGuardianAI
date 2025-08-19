import os
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import joblib
import yaml
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import lightgbm as lgb

load_dotenv()

ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONFIG = {
    "model": {
        "model_path": str(ROOT / "models" / "lgb_baseline_model.txt"),
        "joblib_path": str(ROOT / "models" / "lgb_baseline_model.joblib"),
        "encoders_path": str(ROOT / "models" / "label_encoders.joblib"),
        "expected_features": [
            "hour_sin","hour_cos","weekday_sin","weekday_cos",
            "LATITUDE","LONGITUDE",
            "sev_1h","sev_6h","sev_24h","tot_1h","tot_6h","tot_24h",
            "seg_te","cluster_te"
        ]
    },
    "history": {
        "dir": str(ROOT / "data" / "processed" / "segment_history_files_shifted_v2"),
        "compute_on_missing": True
    },
    "api": {
        "default_top_k": 10,
        "cache_ttl_seconds": 300,
        "batch_size": 2000
    }
}


def load_config(path: Optional[str] = None) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if path:
        p = Path(path)
        if p.exists():
            try:
                with open(p, "rt") as fh:
                    user = yaml.safe_load(fh) or {}
                for k, v in user.items():
                    if isinstance(v, dict) and k in cfg:
                        cfg[k].update(v)
                    else:
                        cfg[k] = v
            except Exception:
                pass
    return cfg


class SimpleTTLCache:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str):
        t = time.time()
        row = self._store.get(key)
        if row is None:
            return None
        ts, value = row
        if t - ts > self.ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any):
        self._store[key] = (time.time(), value)

    def clear(self):
        self._store.clear()


class ModelServer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None
        self.booster = None
        self.enc = {}
        self.seg_te_map = {}
        self.cluster_te_map = {}
        self.global_seg_te = 0.0
        self.expected_features = cfg["model"].get("expected_features", [])
        self.history_dir = Path(cfg["history"]["dir"])
        self.cache = SimpleTTLCache(ttl_seconds=cfg["api"].get("cache_ttl_seconds", 300))
        self.batch_size = cfg["api"].get("batch_size", 2000)
        self._load_model_and_encoders()

    def _load_model_and_encoders(self):
        model_path_txt = Path(self.cfg["model"].get("model_path"))
        model_path_joblib = Path(self.cfg["model"].get("joblib_path"))
        enc_path = Path(self.cfg["model"].get("encoders_path"))

        print("CONFIG model_path:", model_path_txt)
        print("CONFIG joblib_path:", model_path_joblib)
        print("CONFIG encoders_path:", enc_path)
        print("ROOT resolved to:", ROOT)

        loaded = None
        if model_path_joblib.exists():
            try:
                loaded = joblib.load(str(model_path_joblib))
                if isinstance(loaded, lgb.Booster):
                    self.booster = loaded
                    self.model = loaded
                    print("Loaded LightGBM Booster via joblib:", model_path_joblib)
                else:
                    self.model = loaded
                    print("Loaded model object via joblib:", type(loaded), model_path_joblib)
            except Exception as e:
                print("joblib model load failed:", e)

        if self.booster is None and (not self.model) and model_path_txt.exists():
            try:
                self.booster = lgb.Booster(model_file=str(model_path_txt))
                self.model = self.booster
                print("Loaded LightGBM Booster from txt:", model_path_txt)
            except Exception as e:
                print("failed to load booster from text file:", e)

        if enc_path.exists():
            try:
                enc = joblib.load(str(enc_path))
                if isinstance(enc, dict):
                    self.enc = enc
                    self.seg_te_map = enc.get("seg_te_map", {})
                    self.cluster_te_map = enc.get("cluster_te_map", {})
                    if len(self.seg_te_map) > 0:
                        self.global_seg_te = float(np.nanmean(list(self.seg_te_map.values())))
                    else:
                        self.global_seg_te = 0.0
                    print("Loaded encoders/maps:", enc_path, "seg_te keys:", len(self.seg_te_map))
                else:
                    print("Encoders file loaded but not a dict. Type:", type(enc))
            except Exception as e:
                print("Failed to load encoders:", e)
        else:
            print("No encoders file found at:", enc_path)

        if self.model is None:
            missing = []
            if not model_path_joblib.exists() and not model_path_txt.exists():
                missing.append(f"model (joblib_exists={model_path_joblib.exists()}, txt_exists={model_path_txt.exists()})")
            if not enc_path.exists():
                missing.append(f"encoders_path={enc_path}")
            raise RuntimeError("Model/encoder load failed. Missing: " + "; ".join(missing))

        if self.booster is not None:
            try:
                model_features = self.booster.feature_name()
                self.expected_features = model_features
            except Exception:
                pass

    def predict_proba_vect(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is not None:
            feat_names = self.booster.feature_name()
            for f in feat_names:
                if f not in X.columns:
                    X[f] = 0
            X_ord = X[feat_names]
            pred = self.booster.predict(X_ord, num_iteration=getattr(self.booster, "best_iteration", None))
            return np.asarray(pred)
        else:
            if hasattr(self.model, "predict_proba"):
                for f in self.expected_features:
                    if f not in X.columns:
                        X[f] = 0
                X_ord = X[self.expected_features]
                probs = self.model.predict_proba(X_ord)
                return probs[:, 1]
            elif hasattr(self.model, "predict"):
                for f in self.expected_features:
                    if f not in X.columns:
                        X[f] = 0
                X_ord = X[self.expected_features]
                raw = self.model.predict(X_ord)
                return raw.astype(float)
            else:
                raise RuntimeError("Loaded model is not callable/predictable.")

    @staticmethod
    def compute_time_feats(ts: datetime) -> Dict[str, float]:
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        hour = ts.hour
        weekday = ts.weekday()
        return {
            "hour": hour,
            "weekday": weekday,
            "hour_sin": math.sin(2 * math.pi * hour / 24.0),
            "hour_cos": math.cos(2 * math.pi * hour / 24.0),
            "weekday_sin": math.sin(2 * math.pi * weekday / 7.0),
            "weekday_cos": math.cos(2 * math.pi * weekday / 7.0),
        }

    def read_history_for_segments(self, segments: List[str], window_start: datetime) -> Dict[str, dict]:
        out = {}
        p0 = self.history_dir
        win = pd.to_datetime(window_start)
        for seg in segments:
            safe_name = str(seg).replace("/", "_").replace("\\", "_").replace(" ", "_")[:200]
            path = p0 / f"{safe_name}.parquet"
            if not path.exists():
                out[seg] = {}
                continue
            try:
                hf = pd.read_parquet(path)
                if "window_start" in hf.columns:
                    hf["window_start"] = pd.to_datetime(hf["window_start"])
                    match = hf[hf["window_start"] == win]
                    if match.empty:
                        out[seg] = {}
                    else:
                        row = match.iloc[0].to_dict()
                        out[seg] = row
                else:
                    out[seg] = {}
            except Exception:
                out[seg] = {}
        return out

    def assemble_features_batch(self, segments: List[str], ts_pred: datetime) -> pd.DataFrame:
        rows = []
        hist_map = self.read_history_for_segments(segments, ts_pred)
        time_feats = self.compute_time_feats(ts_pred)
        for seg in segments:
            h = hist_map.get(seg, {}) or {}
            r = {}
            r.update({k: time_feats[k] for k in ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]})
            r["LATITUDE"] = h.get("LATITUDE") if h.get("LATITUDE") is not None else np.nan
            r["LONGITUDE"] = h.get("LONGITUDE") if h.get("LONGITUDE") is not None else np.nan
            for c in ["sev_1h","sev_6h","sev_24h","tot_1h","tot_6h","tot_24h"]:
                r[c] = int(h.get(c, 0)) if h.get(c) is not None else 0
            seg_te = self.seg_te_map.get(seg)
            if seg_te is None:
                seg_te = self.global_seg_te
            r["seg_te"] = float(seg_te)
            if self.cluster_te_map:
                cluster_val = h.get("cluster") or h.get("cluster_id")
                if cluster_val is not None and cluster_val in self.cluster_te_map:
                    r["cluster_te"] = float(self.cluster_te_map[cluster_val])
                else:
                    r["cluster_te"] = float(np.nanmean(list(self.cluster_te_map.values()))) if len(self.cluster_te_map)>0 else 0.0
            rows.append((seg, r))

        df_rows = []
        for seg, r in rows:
            rowdict = r.copy()
            rowdict["segment_id"] = seg
            df_rows.append(rowdict)
        df = pd.DataFrame(df_rows)
        if "LATITUDE" in df.columns:
            df["LATITUDE"] = df["LATITUDE"].fillna(0.0)
        if "LONGITUDE" in df.columns:
            df["LONGITUDE"] = df["LONGITUDE"].fillna(0.0)

        for f in self.expected_features:
            if f not in df.columns:
                df[f] = 0
        df_model = df[self.expected_features].copy()
        df_model.insert(0, "segment_id", df["segment_id"].values)
        return df_model

    def list_all_segments_from_history(self) -> List[str]:
        segs = []
        p = Path(self.history_dir)
        if not p.exists():
            return segs
        for f in p.glob("*.parquet"):
            name = f.stem
            segs.append(name)
        return segs

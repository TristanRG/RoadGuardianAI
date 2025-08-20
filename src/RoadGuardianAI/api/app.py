from fastapi import FastAPI, HTTPException, Depends, Header, Query
from pydantic import BaseModel, Field, conint
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import uvicorn
import os
from pydantic import conint

from .model_serving import load_config, ModelServer

CFG = load_config("config/config.yml")
MODEL_SERVER = ModelServer(CFG)

app = FastAPI(
    title="RoadGuardian API",
    version="0.1",
    description="Risk prediction API for road segments",
)

API_KEY = os.getenv("API_KEY")


def require_api_key(x_api_key: Optional[str] = Header(None)):
    """Validate API key if configured."""
    if API_KEY is None:
        return True
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key (X-Api-Key header).")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return True


class PredictItem(BaseModel):
    segment_id: str = Field(..., description="Unique road segment ID")
    ts: Optional[datetime] = Field(
        None, description="Prediction timestamp (UTC). Defaults to current UTC hour"
    )


class PredictRequest(BaseModel):
    items: List[PredictItem] = Field(..., description="List of segments to predict for")


class RiskRequest(BaseModel):
    ts: Optional[datetime] = Field(None, description="Timestamp to predict risk for")
    top_k: Optional[int] = Query(
    None, gt=0, description="Number of top risky segments"
)

def _format_item(
    sid: str,
    prob: float,
    hist_row: dict,
    rank: Optional[int] = None,
    ts: Optional[datetime] = None,
):
    p = float(prob)
    item = {
        "segment_id": sid,
        "rank": rank,
        "probability": round(p, 6),
        "probability_pct": f"{p*100:.2f}%",
        "lat": float(hist_row.get("LATITUDE")) if hist_row.get("LATITUDE") else None,
        "lon": float(hist_row.get("LONGITUDE")) if hist_row.get("LONGITUDE") else None,
        "seg_te": round(
            float(MODEL_SERVER.seg_te_map.get(sid, MODEL_SERVER.global_seg_te)), 6
        ),
        "recent": {
            "sev_1h": int(hist_row.get("sev_1h", 0)),
            "sev_6h": int(hist_row.get("sev_6h", 0)),
            "tot_6h": int(hist_row.get("tot_6h", 0)),
        },
        "has_history": bool(hist_row),
    }
    if ts is not None:
        item["ts"] = ts.isoformat()
    return item

@app.get("/health")
def health():
    """Health check for model + history data."""
    info = {
        "model_loaded": MODEL_SERVER.model is not None,
        "model_type": str(type(MODEL_SERVER.model)),
        "n_seg_te_keys": len(MODEL_SERVER.seg_te_map),
        "history_dir_exists": Path(MODEL_SERVER.history_dir).exists(),
    }
    return {"status": "ok", "info": info}


@app.post("/predict")
def predict(
    req: PredictRequest,
    authorized: bool = Depends(require_api_key),
    pretty: bool = True,
):
    """Predict risk probability for given road segments."""
    if len(req.items) == 0:
        raise HTTPException(status_code=400, detail="No items provided.")

    results = []
    grouped = {}

    for it in req.items:
        ts_here = (
            it.ts
            if it.ts is not None
            else datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        )
        ts_here = ts_here.replace(tzinfo=None)
        grouped.setdefault(ts_here, []).append(it.segment_id)

    for ts_pred, segs in grouped.items():
        df_feats = MODEL_SERVER.assemble_features_batch(segs, ts_pred)
        seg_ids = df_feats["segment_id"].tolist()
        X = df_feats.drop(columns=["segment_id"])
        probs = MODEL_SERVER.predict_proba_vect(X)
        hist_rows = MODEL_SERVER.read_history_for_segments(seg_ids, ts_pred)
        for sid, p in zip(seg_ids, probs):
            if pretty:
                results.append(
                    _format_item(sid, p, hist_rows.get(sid, {}) or {}, ts=ts_pred)
                )
            else:
                results.append(
                    {"segment_id": sid, "ts": ts_pred.isoformat(), "probability": float(p)}
                )

    return {"count": len(results), "predictions": results}


@app.post("/risk")
def risk_post(
    req: RiskRequest,
    authorized: bool = Depends(require_api_key),
    pretty: bool = True,
):
    """Get top-k risky segments at a given time (POST)."""
    ts = (
        req.ts
        if req.ts is not None
        else datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    ).replace(tzinfo=None)

    top_k = req.top_k or CFG["api"].get("default_top_k", 10)
    cache_key = f"risk:{ts.isoformat()}:{top_k}"
    cached = MODEL_SERVER.cache.get(cache_key)
    if cached is not None:
        return {"ts": ts.isoformat(), "top_k": top_k, "results": cached, "cached": True}

    segs = MODEL_SERVER.list_all_segments_from_history()
    if not segs:
        raise HTTPException(status_code=500, detail="No segment history files found.")

    all_results = []
    for i in range(0, len(segs), MODEL_SERVER.batch_size):
        batch = segs[i : i + MODEL_SERVER.batch_size]
        df_feats = MODEL_SERVER.assemble_features_batch(batch, ts)
        seg_ids = df_feats["segment_id"].tolist()
        X = df_feats.drop(columns=["segment_id"])
        probs = MODEL_SERVER.predict_proba_vect(X)
        for sid, p in zip(seg_ids, probs):
            all_results.append({"segment_id": sid, "probability": float(p)})

    top_sorted = sorted(all_results, key=lambda x: x["probability"], reverse=True)[:top_k]
    top_seg_ids = [r["segment_id"] for r in top_sorted]
    hist_rows = MODEL_SERVER.read_history_for_segments(top_seg_ids, ts)

    enriched = [
        _format_item(r["segment_id"], r["probability"], hist_rows.get(r["segment_id"], {}), rank_idx, ts)
        if pretty
        else {"segment_id": r["segment_id"], "probability": r["probability"], **hist_rows.get(r["segment_id"], {})}
        for rank_idx, r in enumerate(top_sorted, start=1)
    ]

    MODEL_SERVER.cache.set(cache_key, enriched)
    return {"ts": ts.isoformat(), "top_k": top_k, "results": enriched, "cached": False}


@app.get("/risk")
def risk_get(
    top_k: Optional[int] = Query(
        None, gt=0, description="Number of top risky segments"
    ),
    ts: Optional[datetime] = Query(
        None, description="Timestamp to predict for"
    ),
    authorized: bool = Depends(require_api_key),
    pretty: bool = True,
):
    """Get top-k risky segments at a given time (GET)."""
    ts_use = (
        ts if ts is not None
        else datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    ).replace(tzinfo=None)

    top_k_use = top_k or CFG["api"].get("default_top_k", 10)
    cache_key = f"risk:{ts_use.isoformat()}:{top_k_use}"
    cached = MODEL_SERVER.cache.get(cache_key)
    if cached is not None:
        return {"ts": ts_use.isoformat(), "top_k": top_k_use, "results": cached, "cached": True}

    segs = MODEL_SERVER.list_all_segments_from_history()
    if not segs:
        raise HTTPException(status_code=500, detail="No segment history files found.")

    all_results = []
    for i in range(0, len(segs), MODEL_SERVER.batch_size):
        batch = segs[i : i + MODEL_SERVER.batch_size]
        df_feats = MODEL_SERVER.assemble_features_batch(batch, ts_use)
        seg_ids = df_feats["segment_id"].tolist()
        X = df_feats.drop(columns=["segment_id"])
        probs = MODEL_SERVER.predict_proba_vect(X)
        for sid, p in zip(seg_ids, probs):
            all_results.append({"segment_id": sid, "probability": float(p)})

    top_sorted = sorted(all_results, key=lambda x: x["probability"], reverse=True)[:top_k_use]
    top_seg_ids = [r["segment_id"] for r in top_sorted]
    hist_rows = MODEL_SERVER.read_history_for_segments(top_seg_ids, ts_use)

    enriched = [
        _format_item(r["segment_id"], r["probability"], hist_rows.get(r["segment_id"], {}), rank_idx, ts_use)
        if pretty
        else {"segment_id": r["segment_id"], "probability": r["probability"], **hist_rows.get(r["segment_id"], {})}
        for rank_idx, r in enumerate(top_sorted, start=1)
    ]

    MODEL_SERVER.cache.set(cache_key, enriched)
    return {"ts": ts_use.isoformat(), "top_k": top_k_use, "results": enriched, "cached": False}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("src.RoadGuardianAI.api.app:app", host="0.0.0.0", port=port, reload=True)

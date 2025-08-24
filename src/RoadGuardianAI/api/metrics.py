from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry
from starlette.responses import Response

HTTP_REQUESTS_TOTAL = Counter(
    "rg_http_requests_total",
    "Total HTTP requests processed by the API",
    ["method", "endpoint", "status_code"],
)
HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "rg_http_request_latency_seconds", "HTTP request latency seconds", ["method", "endpoint"]
)

PREDICTIONS_TOTAL = Counter(
    "rg_predictions_total", "Total number of predictions returned by predict endpoints"
)
BATCH_RUNS_TOTAL = Counter(
    "rg_batch_runs_total", "Total number of batch prediction runs executed"
)
BATCH_PREDICTIONS_TOTAL = Counter(
    "rg_batch_predictions_total", "Total number of predictions produced by batch runs"
)
MODEL_LOADED = Gauge("rg_model_loaded", "Is model loaded (1=yes,0=no)")

def metrics_response() -> Response:
    payload = generate_latest()  
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

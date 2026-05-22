from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
from pathlib import Path
import time
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

try:
    from api.predictor import ClaimPredictor
    from api.ui import PREDICT_UI_HTML
except ModuleNotFoundError:
    from predictor import ClaimPredictor
    from ui import PREDICT_UI_HTML

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_PATH = PROJECT_DIR / "data" / "ours" / "test.csv"

predictor = ClaimPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup
    predictor.load()
    logger.info("Model ready")
    yield
    # Cleanup on shutdown (nothing needed here)


app = FastAPI(
    title="Claim Detection API",
    summary="Sentence-level factual claim classification.",
    description=(
        "Classify text as a factual claim and inspect model readiness for "
        "the local Llama claim detector."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "displayRequestDuration": True,
        "filter": True,
        "syntaxHighlight": {"theme": "arta"},
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

APP_CSS = """
<style>
  :root {
    --bg: #f4efe6;
    --card: rgba(255, 255, 255, 0.84);
    --ink: #1c2a24;
    --muted: #5c6c63;
    --line: rgba(28, 42, 36, 0.12);
    --accent: #1f6f5f;
    --accent-2: #d9a441;
    --good: #1f7a4f;
    --bad: #a6432b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    color: var(--ink);
    background:
      radial-gradient(circle at top left, rgba(217, 164, 65, 0.22), transparent 28%),
      radial-gradient(circle at bottom right, rgba(31, 111, 95, 0.18), transparent 30%),
      linear-gradient(135deg, #f7f2ea 0%, #efe7da 100%);
  }
  .page {
    max-width: 1100px;
    margin: 0 auto;
    padding: 40px 20px 64px;
  }
  .hero {
    display: grid;
    gap: 18px;
    margin-bottom: 28px;
  }
  .eyebrow {
    font-size: 12px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
  }
  h1, h2, h3 {
    margin: 0;
    line-height: 1.05;
  }
  h1 {
    font-size: clamp(2.4rem, 5vw, 4.8rem);
    max-width: 10ch;
  }
  p {
    margin: 0;
    color: var(--muted);
    line-height: 1.6;
  }
  .grid {
    display: grid;
    gap: 18px;
  }
  .grid.two {
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  }
  .grid.three {
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  }
  .card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 22px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(28, 42, 36, 0.06);
  }
  .metric {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent);
  }
  .muted { color: var(--muted); }
  .row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    align-items: center;
  }
  .pill {
    display: inline-flex;
    align-items: center;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(31, 111, 95, 0.08);
    color: var(--accent);
    font-size: 0.95rem;
  }
  a.button, button {
    appearance: none;
    border: 0;
    border-radius: 14px;
    padding: 12px 16px;
    background: var(--ink);
    color: #fff;
    font: inherit;
    text-decoration: none;
    cursor: pointer;
    transition: transform 160ms ease, opacity 160ms ease;
  }
  a.button.secondary, button.secondary {
    background: #fff;
    color: var(--ink);
    border: 1px solid var(--line);
  }
  a.button:hover, button:hover { transform: translateY(-1px); }
  input, textarea {
    width: 100%;
    padding: 12px 14px;
    border-radius: 14px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.92);
    color: var(--ink);
    font: inherit;
  }
  textarea {
    min-height: 120px;
    resize: vertical;
  }
  label {
    display: grid;
    gap: 8px;
    font-size: 0.95rem;
    color: var(--muted);
  }
  pre {
    margin: 0;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-word;
    padding: 16px;
    border-radius: 16px;
    background: #18221e;
    color: #eff7f0;
    font-size: 0.92rem;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95rem;
  }
  th, td {
    text-align: left;
    padding: 10px 0;
    border-bottom: 1px solid var(--line);
  }
  .good { color: var(--good); }
  .bad { color: var(--bad); }
  .footer-note {
    margin-top: 18px;
    font-size: 0.92rem;
    color: var(--muted);
  }
</style>
"""


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    sentence: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        examples=["The Empire State Building is the tallest building in New York City."],
    )

    @field_validator("sentence")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class PredictResponse(BaseModel):
    sentence: str
    is_claim: bool
    confidence: float
    claim_probability: float
    latency_ms: float


class EvaluationReportResponse(BaseModel):
    sample_size: int
    sample_seed: int
    accuracy: float
    f1: float
    label_distribution: dict[str, int]
    prediction_distribution: dict[str, int]
    classification_report: dict


# ── Middleware: request timing ────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = str(round(elapsed, 2))
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

DOCS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Claim Detection API Docs</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
  <style>
    :root {
      --ink: #17231f;
      --muted: #51635b;
      --paper: #f5f2eb;
      --panel: #fffdf9;
      --line: #d8d0c2;
      --claim: #087f5b;
      --signal: #1f5fbb;
      --alert: #c2410c;
    }

    * {
      box-sizing: border-box;
      letter-spacing: 0;
    }

    body {
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(180deg, #fff9ef 0, var(--paper) 300px),
        var(--paper);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }

    .docs-header {
      border-bottom: 1px solid var(--line);
      background: rgba(255, 253, 249, 0.94);
    }

    .docs-header__inner {
      width: min(1180px, calc(100% - 40px));
      margin: 0 auto;
      padding: 30px 0 24px;
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 24px;
    }

    .docs-header h1 {
      margin: 0 0 8px;
      font-size: clamp(1.8rem, 3vw, 2.7rem);
      line-height: 1.05;
    }

    .docs-header p {
      max-width: 620px;
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }

    .docs-status {
      min-width: 170px;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-left: 4px solid var(--claim);
      border-radius: 8px;
      color: var(--muted);
      background: var(--panel);
      box-shadow: 0 12px 36px rgba(23, 35, 31, 0.08);
    }

    .docs-status strong {
      display: block;
      margin-bottom: 4px;
      color: var(--ink);
    }

    #swagger-ui {
      width: min(1180px, calc(100% - 40px));
      margin: 0 auto;
      padding: 24px 0 48px;
    }

    .swagger-ui,
    .swagger-ui .info .title,
    .swagger-ui .opblock-tag {
      font-family: inherit;
      color: var(--ink);
    }

    .swagger-ui .topbar,
    .swagger-ui .info .title small,
    .swagger-ui .models {
      display: none;
    }

    .swagger-ui .info {
      margin: 0 0 22px;
      padding: 22px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }

    .swagger-ui .info .title {
      margin-bottom: 10px;
      font-size: 1.45rem;
    }

    .swagger-ui .scheme-container,
    .swagger-ui .opblock,
    .swagger-ui .dialog-ux .modal-ux {
      border-radius: 8px;
      box-shadow: none;
    }

    .swagger-ui .scheme-container {
      margin: 0 0 18px;
      padding: 18px;
      border: 1px solid var(--line);
      background: var(--panel);
    }

    .swagger-ui .opblock {
      overflow: hidden;
      border-width: 1px;
      background: var(--panel);
    }

    .swagger-ui .opblock.opblock-get {
      border-color: rgba(31, 95, 187, 0.34);
      background: rgba(31, 95, 187, 0.08);
    }

    .swagger-ui .opblock.opblock-post {
      border-color: rgba(8, 127, 91, 0.36);
      background: rgba(8, 127, 91, 0.08);
    }

    .swagger-ui .opblock.opblock-post .opblock-summary-method {
      background: var(--claim);
    }

    .swagger-ui .opblock.opblock-get .opblock-summary-method {
      background: var(--signal);
    }

    .swagger-ui button,
    .swagger-ui select,
    .swagger-ui input,
    .swagger-ui textarea {
      font-family: inherit;
    }

    .swagger-ui .btn.execute {
      border-color: var(--alert);
      background: var(--alert);
    }

    @media (max-width: 720px) {
      .docs-header__inner {
        width: min(100% - 24px, 1180px);
        align-items: stretch;
        flex-direction: column;
        padding-top: 22px;
      }

      #swagger-ui {
        width: min(100% - 24px, 1180px);
      }
    }
  </style>
</head>
<body>
  <header class="docs-header">
    <div class="docs-header__inner">
      <div>
        <h1>Claim Detection API</h1>
        <p>Inspect model readiness and classify sentences by factual claim likelihood.</p>
      </div>
      <div class="docs-status">
        <strong>Local model API</strong>
        Llama claim detector
      </div>
    </div>
  </header>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    SwaggerUIBundle({
      url: "/openapi.json",
      dom_id: "#swagger-ui",
      deepLinking: true,
      displayRequestDuration: true,
      filter: true,
      defaultModelsExpandDepth: -1,
      presets: [SwaggerUIBundle.presets.apis],
    });
  </script>
</body>
</html>
"""


@app.get("/docs", include_in_schema=False)
def docs():
    return HTMLResponse(DOCS_HTML)


@app.get("/", include_in_schema=False)
def prediction_ui():
    return HTMLResponse(PREDICT_UI_HTML)


@app.get("/health", tags=["Operations"], summary="Check model readiness")
def health():
    return {"status": "ok", "model_loaded": predictor.model is not None}


@app.post("/reload-model", tags=["Operations"], summary="Reload the local model")
def reload_model():
    predictor.load()
    return {"status": "ok", "model_loaded": predictor.model is not None}


def run_prediction(body: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    try:
        result = predictor.predict(body.sentence)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")

    latency = round((time.perf_counter() - start) * 1000, 2)

    return PredictResponse(
        sentence=body.sentence,
        latency_ms=latency,
        **result,
    )


def format_prediction(result: PredictResponse) -> str:
    verdict = "a factual claim" if result.is_claim else "not a factual claim"
    return (
        f"Sentence: {result.sentence}\n"
        f"Verdict: This is {verdict}.\n"
        f"Claim probability: {result.claim_probability:.1%}\n"
        f"Confidence: {result.confidence:.1%}\n"
        f"Latency: {result.latency_ms} ms"
    )


@app.post(
    "/predict",
    response_class=PlainTextResponse,
    tags=["Predictions"],
    summary="Classify one sentence as text",
)
def predict_text(body: PredictRequest):
    return format_prediction(run_prediction(body))


@app.post(
    "/predict/json",
    response_model=PredictResponse,
    tags=["Predictions"],
    summary="Classify one sentence as JSON",
)
def predict_json(body: PredictRequest):
    return run_prediction(body)


@app.post("/predict/batch", tags=["Predictions"], summary="Classify up to 32 sentences")
def predict_batch(sentences: list[str]):
    """Predict on multiple sentences at once (max 32)."""
    if len(sentences) > 32:
        raise HTTPException(status_code=400, detail="Max 32 sentences per batch")
    return [predictor.predict(s) for s in sentences]


@app.get("/report/test", response_model=EvaluationReportResponse)
def report_test(sample_size: int = 100, sample_seed: int = 42):
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if sample_size <= 0:
        raise HTTPException(status_code=400, detail="sample_size must be greater than 0")
    if not TEST_DATA_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Test data not found at {TEST_DATA_PATH}")

    test_df = pd.read_csv(TEST_DATA_PATH)
    test_df = test_df.rename(columns={"text": "sentence"})
    test_df = test_df.dropna(subset=["sentence", "label"]).copy()
    test_df["sentence"] = test_df["sentence"].astype(str)
    test_df["label"] = test_df["label"].astype(int)

    if sample_size > len(test_df):
        raise HTTPException(
            status_code=400,
            detail=f"sample_size ({sample_size}) exceeds test set size ({len(test_df)})",
        )

    sample_df = test_df.sample(n=sample_size, random_state=sample_seed).reset_index(drop=True)

    rows = []
    for row in sample_df.to_dict(orient="records"):
        prediction = predictor.predict(row["sentence"])
        rows.append(
            {
                "label": int(row["label"]),
                "pred": int(prediction["is_claim"]),
            }
        )

    results_df = pd.DataFrame(rows)
    y_true = results_df["label"]
    y_pred = results_df["pred"]

    return EvaluationReportResponse(
        sample_size=sample_size,
        sample_seed=sample_seed,
        accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
        f1=round(float(f1_score(y_true, y_pred)), 4),
        label_distribution={str(k): int(v) for k, v in y_true.value_counts().sort_index().items()},
        prediction_distribution={str(k): int(v) for k, v in y_pred.value_counts().sort_index().items()},
        classification_report=classification_report(y_true, y_pred, digits=4, output_dict=True),
    )


@app.get("/report/test/ui", response_class=HTMLResponse)
def report_test_ui():
    return f"""
    <html>
      <head>
        <title>Test Report UI</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        {APP_CSS}
      </head>
      <body>
        <main class="page">
          <section class="hero">
            <div class="eyebrow">Evaluation</div>
            <h1>Test-split report without raw JSON first.</h1>
            <p>
              This view queries <code>/report/test</code> and renders the metrics,
              label balance, and classification report in a readable layout.
            </p>
            <div class="row">
              <a class="button secondary" href="/">Back Home</a>
              <a class="button secondary" href="/report/test">View Raw JSON</a>
            </div>
          </section>

          <section class="card">
            <div class="grid three">
              <label>
                Sample Size
                <input id="sample-size" type="number" min="1" value="100" />
              </label>
              <label>
                Sample Seed
                <input id="sample-seed" type="number" value="42" />
              </label>
              <div class="row" style="align-self:end">
                <button onclick="loadReport()">Run Report</button>
              </div>
            </div>
          </section>

          <div style="height:18px"></div>

          <section class="grid three">
            <div class="card">
              <div class="muted">Accuracy</div>
              <div id="accuracy" class="metric">-</div>
            </div>
            <div class="card">
              <div class="muted">F1</div>
              <div id="f1" class="metric">-</div>
            </div>
            <div class="card">
              <div class="muted">Sample</div>
              <div id="sample-meta" class="metric" style="font-size:1.35rem">-</div>
            </div>
          </section>

          <div style="height:18px"></div>

          <section class="grid two">
            <div class="card">
              <h2>Distributions</h2>
              <div style="height:14px"></div>
              <div id="distributions" class="grid"></div>
            </div>

            <div class="card">
              <h2>Classification Report</h2>
              <div style="height:14px"></div>
              <div id="report-table" class="muted">Run the report to populate this table.</div>
            </div>
          </section>

          <div style="height:18px"></div>

          <section class="card">
            <h2>Raw Payload</h2>
            <div style="height:14px"></div>
            <pre id="raw-json">Run the report to inspect the underlying JSON.</pre>
          </section>
        </main>

        <script>
          function renderPairs(title, obj) {{
            const rows = Object.entries(obj || {{}})
              .map(([key, value]) => `<div class="pill">${{title}} ${{key}}: ${{value}}</div>`)
              .join("");
            return rows || '<div class="muted">No data</div>';
          }}

          function renderReportTable(report) {{
            const keys = ["0", "1", "macro avg", "weighted avg"];
            const rows = keys
              .filter((key) => report[key])
              .map((key) => {{
                const row = report[key];
                return `
                  <tr>
                    <td>${{key}}</td>
                    <td>${{Number(row.precision).toFixed(4)}}</td>
                    <td>${{Number(row.recall).toFixed(4)}}</td>
                    <td>${{Number(row["f1-score"]).toFixed(4)}}</td>
                    <td>${{Number(row.support).toFixed(0)}}</td>
                  </tr>
                `;
              }})
              .join("");

            return `
              <table>
                <thead>
                  <tr>
                    <th>Label</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>Support</th>
                  </tr>
                </thead>
                <tbody>${{rows}}</tbody>
              </table>
            `;
          }}

          async function loadReport() {{
            const sampleSize = document.getElementById("sample-size").value;
            const sampleSeed = document.getElementById("sample-seed").value;
            document.getElementById("raw-json").textContent = "Running...";
            document.getElementById("report-table").innerHTML = '<span class="muted">Running...</span>';
            try {{
              const response = await fetch(`/report/test?sample_size=${{sampleSize}}&sample_seed=${{sampleSeed}}`);
              const data = await response.json();
              document.getElementById("accuracy").textContent = data.accuracy.toFixed(4);
              document.getElementById("f1").textContent = data.f1.toFixed(4);
              document.getElementById("sample-meta").textContent = `${{data.sample_size}} / seed ${{data.sample_seed}}`;
              document.getElementById("distributions").innerHTML =
                renderPairs("True", data.label_distribution) +
                renderPairs("Pred", data.prediction_distribution);
              document.getElementById("report-table").innerHTML =
                renderReportTable(data.classification_report);
              document.getElementById("raw-json").textContent =
                JSON.stringify(data, null, 2);
            }} catch (err) {{
              document.getElementById("raw-json").textContent = String(err);
              document.getElementById("report-table").innerHTML =
                '<span class="bad">Failed to load report.</span>';
            }}
          }}

          loadReport();
        </script>
      </body>
    </html>
    """
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

PREDICT_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Claim Detection</title>
  <style>
    :root {
      --bg: #f4f7f5;
      --paper: #ffffff;
      --ink: #101b18;
      --muted: #53665f;
      --line: #cfd8d4;
      --teal: #006d62;
      --teal-soft: #d9f1eb;
      --blue: #1b5fcc;
      --blue-soft: #dfeaff;
      --coral: #b73e21;
      --coral-soft: #ffe2d8;
      --gold: #8b6100;
      --gold-soft: #ffefbf;
      --shadow: 0 24px 70px rgba(16, 27, 24, 0.12);
    }

    * {
      box-sizing: border-box;
      letter-spacing: 0;
    }

    body {
      min-height: 100vh;
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(130deg, #eff8f5 0, #f7f9ff 44%, #fff6f2 100%);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }

    button,
    textarea {
      font: inherit;
    }

    .shell {
      width: min(1240px, calc(100% - 48px));
      margin: 0 auto;
      padding: 28px 0 52px;
    }

    .topbar {
      min-height: 50px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      border-bottom: 1px solid rgba(16, 27, 24, 0.13);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 720;
    }

    .brand-mark {
      width: 34px;
      height: 34px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: white;
      background: var(--teal);
      box-shadow: 6px 6px 0 rgba(27, 95, 204, 0.16);
    }

    .top-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .link-button,
    .status {
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 0 13px;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.78);
      text-decoration: none;
    }

    .status-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--gold);
    }

    .status[data-state="ready"] .status-dot {
      background: var(--teal);
    }

    .status[data-state="error"] .status-dot {
      background: var(--coral);
    }

    main {
      min-height: calc(100vh - 130px);
      display: grid;
      align-items: center;
      gap: 34px;
      padding-top: 34px;
    }

    .intro {
      max-width: 760px;
    }

    h1 {
      max-width: 720px;
      margin: 0;
      font-size: clamp(2.15rem, 4.9vw, 5rem);
      line-height: 0.98;
    }

    .intro p {
      max-width: 560px;
      margin: 18px 0 0;
      color: var(--muted);
      font-size: 1.04rem;
      line-height: 1.5;
    }

    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1.12fr) minmax(340px, 0.88fr);
      overflow: hidden;
      border: 1px solid rgba(16, 27, 24, 0.15);
      border-radius: 8px;
      background: var(--paper);
      box-shadow: var(--shadow);
    }

    .compose,
    .result {
      min-height: 482px;
      padding: clamp(22px, 3vw, 38px);
    }

    .compose {
      display: flex;
      flex-direction: column;
      border-right: 1px solid var(--line);
    }

    label,
    .eyebrow {
      display: block;
      margin-bottom: 11px;
      color: var(--muted);
      font-size: 0.83rem;
      font-weight: 760;
      text-transform: uppercase;
    }

    textarea {
      width: 100%;
      min-height: 190px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 19px;
      color: var(--ink);
      background: #fbfdfc;
      font-size: 1.12rem;
      line-height: 1.5;
      outline: none;
    }

    textarea:focus {
      border-color: var(--blue);
      box-shadow: 0 0 0 4px rgba(27, 95, 204, 0.15);
    }

    .examples {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 14px 0 0;
    }

    .example {
      min-height: 36px;
      max-width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 12px;
      color: var(--ink);
      background: var(--bg);
      cursor: pointer;
    }

    .example:hover {
      border-color: var(--teal);
      background: var(--teal-soft);
    }

    .compose-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-top: auto;
      padding-top: 28px;
    }

    .counter {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }

    .submit {
      min-width: 156px;
      min-height: 48px;
      border: 0;
      border-radius: 8px;
      color: white;
      background: var(--ink);
      cursor: pointer;
      font-weight: 720;
    }

    .submit:hover:not(:disabled) {
      background: var(--teal);
    }

    .submit:disabled {
      cursor: wait;
      opacity: 0.58;
    }

    .result {
      display: grid;
      align-content: space-between;
      gap: 28px;
      background:
        linear-gradient(145deg, rgba(217, 241, 235, 0.78), rgba(223, 234, 255, 0.72) 55%, rgba(255, 226, 216, 0.66));
    }

    .verdict {
      display: grid;
      gap: 20px;
    }

    .verdict-title {
      margin: 0;
      font-size: clamp(1.8rem, 3vw, 3rem);
      line-height: 1;
    }

    .probability {
      height: 18px;
      overflow: hidden;
      border: 1px solid rgba(16, 27, 24, 0.18);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.72);
    }

    .probability-fill {
      width: 0;
      height: 100%;
      min-width: 8px;
      background: linear-gradient(90deg, var(--blue), var(--teal));
      transition: width 380ms ease;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .metric {
      min-height: 86px;
      border: 1px solid rgba(16, 27, 24, 0.14);
      border-radius: 8px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.76);
    }

    .metric span {
      display: block;
      margin-bottom: 9px;
      color: var(--muted);
      font-size: 0.82rem;
    }

    .metric strong {
      display: block;
      font-size: 1.48rem;
      font-variant-numeric: tabular-nums;
    }

    .message {
      min-height: 52px;
      margin: 0;
      border-left: 4px solid var(--gold);
      padding: 10px 0 10px 14px;
      color: var(--muted);
      line-height: 1.45;
    }

    .message[data-tone="error"] {
      border-color: var(--coral);
      color: var(--coral);
    }

    .message[data-tone="success"] {
      border-color: var(--teal);
      color: var(--ink);
    }

    .result[data-verdict="claim"] .verdict-title {
      color: var(--teal);
    }

    .result[data-verdict="not-claim"] .verdict-title {
      color: var(--coral);
    }

    @media (max-width: 900px) {
      .shell {
        width: min(100% - 24px, 1240px);
        padding-top: 16px;
      }

      .topbar,
      .top-actions,
      .compose-actions {
        align-items: stretch;
      }

      .topbar {
        flex-direction: column;
        padding-bottom: 16px;
      }

      .top-actions {
        width: 100%;
      }

      .link-button,
      .status {
        flex: 1;
      }

      main {
        gap: 24px;
        padding-top: 24px;
      }

      .workspace {
        grid-template-columns: 1fr;
      }

      .compose {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
    }

    @media (max-width: 560px) {
      .top-actions,
      .compose-actions {
        flex-direction: column;
      }

      .submit {
        width: 100%;
      }

      .metrics {
        grid-template-columns: 1fr;
      }

      .compose,
      .result {
        min-height: auto;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">C</div>
        <span>Claim Detection</span>
      </div>
      <div class="top-actions">
        <div class="status" id="model-status" data-state="checking">
          <span class="status-dot" aria-hidden="true"></span>
          <span id="status-label">Checking model</span>
        </div>
        <a class="link-button" href="/docs">API docs</a>
      </div>
    </header>
    <main>
      <section class="intro">
        <h1>Check whether a sentence makes a factual claim.</h1>
        <p>Enter one sentence and review the checkpoint verdict, claim probability, and response time.</p>
      </section>
      <section class="workspace" aria-label="Claim classifier">
        <form class="compose" id="predict-form">
          <div>
            <label for="sentence">Sentence</label>
            <textarea id="sentence" maxlength="1000" required>The Empire State Building was completed in 1931.</textarea>
            <div class="examples" aria-label="Examples">
              <button class="example" type="button" data-sentence="Water boils at 100 degrees Celsius.">Fact</button>
              <button class="example" type="button" data-sentence="I think this movie is beautiful.">Opinion</button>
              <button class="example" type="button" data-sentence="Mount Everest is the tallest mountain above sea level.">Named entity</button>
            </div>
          </div>
          <div class="compose-actions">
            <span class="counter" id="counter">0 / 1000</span>
            <button class="submit" id="submit" type="submit">Analyze</button>
          </div>
        </form>
        <section class="result" id="result-panel" data-verdict="idle" aria-live="polite">
          <div class="verdict">
            <span class="eyebrow">Verdict</span>
            <h2 class="verdict-title" id="verdict">Ready</h2>
            <div class="probability" aria-label="Claim probability">
              <div class="probability-fill" id="probability-fill"></div>
            </div>
            <div class="metrics">
              <div class="metric">
                <span>Claim probability</span>
                <strong id="probability">--</strong>
              </div>
              <div class="metric">
                <span>Latency</span>
                <strong id="latency">--</strong>
              </div>
            </div>
          </div>
          <p class="message" id="message">Submit a sentence to run the step-500 checkpoint.</p>
        </section>
      </section>
    </main>
  </div>
  <script>
    const sentence = document.querySelector("#sentence");
    const form = document.querySelector("#predict-form");
    const submit = document.querySelector("#submit");
    const counter = document.querySelector("#counter");
    const verdict = document.querySelector("#verdict");
    const resultPanel = document.querySelector("#result-panel");
    const probability = document.querySelector("#probability");
    const probabilityFill = document.querySelector("#probability-fill");
    const latency = document.querySelector("#latency");
    const message = document.querySelector("#message");
    const modelStatus = document.querySelector("#model-status");
    const statusLabel = document.querySelector("#status-label");

    function updateCounter() {
      counter.textContent = `${sentence.value.length} / 1000`;
    }

    function setStatus(state, label) {
      modelStatus.dataset.state = state;
      statusLabel.textContent = label;
    }

    function setMessage(text, tone = "idle") {
      message.textContent = text;
      message.dataset.tone = tone;
    }

    async function checkHealth() {
      try {
        const response = await fetch("/health");
        const data = await response.json();
        setStatus(data.model_loaded ? "ready" : "checking", data.model_loaded ? "Model ready" : "Model loading");
      } catch (error) {
        setStatus("error", "API unavailable");
      }
    }

    document.querySelectorAll("[data-sentence]").forEach((button) => {
      button.addEventListener("click", () => {
        sentence.value = button.dataset.sentence;
        updateCounter();
        sentence.focus();
      });
    });

    sentence.addEventListener("input", updateCounter);

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const value = sentence.value.trim();
      if (!value) {
        sentence.focus();
        setMessage("Enter a sentence before analyzing.", "error");
        return;
      }

      submit.disabled = true;
      submit.textContent = "Analyzing...";
      verdict.textContent = "Running";
      resultPanel.dataset.verdict = "idle";
      setMessage("The checkpoint is evaluating the sentence.");

      try {
        const response = await fetch("/predict/json", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({sentence: value}),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Prediction failed.");
        }

        const percent = Math.round(data.claim_probability * 1000) / 10;
        verdict.textContent = data.is_claim ? "Factual claim" : "Not a factual claim";
        resultPanel.dataset.verdict = data.is_claim ? "claim" : "not-claim";
        probability.textContent = `${percent}%`;
        probabilityFill.style.width = `${Math.max(percent, 1)}%`;
        latency.textContent = `${data.latency_ms} ms`;
        setMessage(data.sentence, "success");
      } catch (error) {
        verdict.textContent = "Unavailable";
        resultPanel.dataset.verdict = "not-claim";
        probability.textContent = "--";
        probabilityFill.style.width = "0";
        latency.textContent = "--";
        setMessage(error.message, "error");
      } finally {
        submit.disabled = false;
        submit.textContent = "Analyze";
      }
    });

    updateCounter();
    checkHealth();
  </script>
</body>
</html>
"""

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from api.predictor import ClaimPredictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the step-500 claim predictor on labeled sentences."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/ours/test.csv"),
        help="CSV with text and label columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/llama_1b_step500_eval.csv"),
        help="CSV path for row-level predictions.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/reports/llama_1b_step500_metrics.csv"),
        help="CSV path for aggregate metrics.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of sentences per model forward pass.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional row limit for a smoke test.",
    )
    return parser.parse_args()


def messages(sentence: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an AI agent used to determine whether or not a "
                "sentence is a factual claim. Only respond with Yes or No"
            ),
        },
        {
            "role": "user",
            "content": f"Is the following sentence a factual claim? {sentence}",
        },
        {"role": "assistant", "content": ""},
    ]


def tokenized_rows(predictor: ClaimPredictor, sentences: list[str]) -> dict[str, torch.Tensor]:
    rows = []
    for sentence in sentences:
        chat_inputs = predictor.tokenizer.apply_chat_template(
            messages(sentence),
            tokenize=True,
            continue_final_message=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        input_ids = chat_inputs.input_ids if hasattr(chat_inputs, "input_ids") else chat_inputs
        rows.append({"input_ids": input_ids[0, :-1]})

    batch = predictor.tokenizer.pad(rows, padding=True, return_tensors="pt")
    return {key: value.to(predictor.device) for key, value in batch.items()}


def predict_batch(predictor: ClaimPredictor, sentences: list[str]) -> list[dict]:
    batch = tokenized_rows(predictor, sentences)

    with torch.no_grad():
        outputs = predictor.model(**batch, use_cache=False)
        next_token_logits = outputs.logits[:, -1]
        no_logits = torch.logsumexp(next_token_logits[:, predictor.no_token_ids], dim=1)
        yes_logits = torch.logsumexp(next_token_logits[:, predictor.yes_token_ids], dim=1)
        claim_probs = torch.softmax(torch.stack((no_logits, yes_logits), dim=1), dim=1)[:, 1]

    predictions = []
    for claim_prob in claim_probs.cpu().tolist():
        is_claim = claim_prob >= 0.5
        predictions.append(
            {
                "pred": int(is_claim),
                "claim_probability": round(claim_prob, 6),
                "confidence": round(claim_prob if is_claim else 1 - claim_prob, 6),
            }
        )
    return predictions


def load_rows(path: Path, limit: int | None) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"text", "label"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path}: missing required column(s): {', '.join(sorted(missing))}.")
    if limit is not None:
        data = data.head(limit)
    return data.dropna(subset=["text", "label"]).copy()


def metrics_frame(results: pd.DataFrame) -> pd.DataFrame:
    y_true = results["label"].astype(int)
    y_pred = results["pred"].astype(int)
    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    true_negative = int(((y_true == 0) & (y_pred == 0)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())

    total = len(results)
    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1 = 2 * precision * recall / f1_denominator if f1_denominator else 0.0

    summary = {
        "rows": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return pd.DataFrame([summary])


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    data = load_rows(args.data, args.limit)
    predictor = ClaimPredictor()
    predictor.load()

    results = []
    batches = range(0, len(data), args.batch_size)
    for start in tqdm(batches, total=(len(data) + args.batch_size - 1) // args.batch_size):
        rows = data.iloc[start : start + args.batch_size]
        predictions = predict_batch(predictor, rows["text"].astype(str).tolist())
        for source, prediction in zip(rows.to_dict(orient="records"), predictions):
            results.append(
                {
                    "text": source["text"],
                    "label": int(source["label"]),
                    **prediction,
                }
            )

    results_frame = pd.DataFrame(results)
    summary = metrics_frame(results_frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    results_frame.to_csv(args.output, index=False)
    summary.to_csv(args.summary, index=False, float_format="%.6f")

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nPredictions: {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()

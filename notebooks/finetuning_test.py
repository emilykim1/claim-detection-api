import argparse

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from finetuning_common import (
    DEFAULT_RESULTS_PATH,
    OUTPUT_DIR,
    load_finetuned_model_and_tokenizer,
    load_test_data,
    predict_claim_probability,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned claim model on the test split.")
    parser.add_argument(
        "--model-path",
        default=OUTPUT_DIR,
        help="Path to the saved fine-tuned model or checkpoint directory.",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_RESULTS_PATH,
        help="CSV path for saving test predictions.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of randomly selected test examples to evaluate.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used for test sampling.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Starting evaluation script")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = load_finetuned_model_and_tokenizer(args.model_path)
    model.to(device)
    model.eval()

    test_df = load_test_data()
    print(f"Loaded {len(test_df)} test rows")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be greater than 0")
    if args.sample_size > len(test_df):
        raise ValueError(
            f"--sample-size ({args.sample_size}) exceeds test set size ({len(test_df)})"
        )

    test_df = test_df.sample(n=args.sample_size, random_state=args.sample_seed).reset_index(drop=True)
    print(
        f"Evaluating random sample of {len(test_df)} test rows "
        f"(seed={args.sample_seed})"
    )

    rows = []
    for idx, row in enumerate(test_df.to_dict(orient="records"), start=1):
        prediction = predict_claim_probability(
            model=model,
            tokenizer=tokenizer,
            sentence=row["sentence"],
            device=device,
        )
        rows.append(
            {
                "sentence": row["sentence"],
                "label": row["label"],
                "pred": prediction["pred"],
                "claim_probability": round(prediction["claim_probability"], 6),
            }
        )

        if idx == 1 or idx % 100 == 0 or idx == len(test_df):
            print(f"Evaluated {idx}/{len(test_df)} test examples")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.output_path, index=False)
    print(f"Saved predictions to {args.output_path}")

    y_true = results_df["label"]
    y_pred = results_df["pred"]
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1: {f1_score(y_true, y_pred):.4f}")
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


METRICS = {
    "Accuracy": accuracy_score,
    "Precision": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
    "Recall": lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
    "F1": lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a classification metrics chart from result CSV files."
    )
    parser.add_argument(
        "results",
        nargs="*",
        type=Path,
        default=[
            Path("results/llama_1b_ft_eval.csv"),
            Path("results/llama_1b_ft_transfer_eval.csv"),
        ],
        help="Result CSV files containing label and pred columns.",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        default=Path("results/reports/classification_metrics.png"),
        help="PNG path for the chart output.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/reports/classification_metrics.csv"),
        help="CSV path for the metric summary.",
    )
    return parser.parse_args()


def display_name(path: Path) -> str:
    words = path.stem.replace("_", " ").replace("-", " ").split()
    aliases = {"ft": "FT", "eval": "Eval", "llama": "Llama"}
    return " ".join(aliases.get(word.lower(), word.title()) for word in words)


def binary_values(frame: pd.DataFrame, column: str, path: Path) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{path}: column '{column}' contains non-numeric values.")

    unique_values = set(values.astype(int).unique())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"{path}: column '{column}' must contain only 0 and 1 values.")

    return values.astype(int)


def summarize(path: Path) -> dict:
    frame = pd.read_csv(path)
    required = {"label", "pred"}
    missing = required.difference(frame.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"{path}: missing required column(s): {columns}.")

    clean = frame.dropna(subset=["label", "pred"]).copy()
    if clean.empty:
        raise ValueError(f"{path}: no rows contain both label and pred values.")

    y_true = binary_values(clean, "label", path)
    y_pred = binary_values(clean, "pred", path)

    summary = {"Run": display_name(path), "Rows": len(clean), "Source": str(path)}
    summary.update({name: metric(y_true, y_pred) for name, metric in METRICS.items()})
    return summary


def save_chart(summary: pd.DataFrame, output: Path):
    chart_data = summary.set_index("Run")[list(METRICS)]
    fig_width = max(9, len(chart_data) * 2.4)
    figure, axis = plt.subplots(figsize=(fig_width, 6.2))

    colors = ["#15616d", "#1d4ed8", "#c2410c", "#7c3aed"]
    chart_data.plot.bar(ax=axis, color=colors, width=0.78)

    axis.set_title("Classification Metrics", fontsize=16, fontweight="bold", pad=18)
    axis.set_ylabel("Score")
    axis.set_xlabel("")
    axis.set_ylim(0, 1.08)
    axis.grid(axis="y", color="#d7dee8", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(title="Metric", frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    axis.tick_params(axis="x", rotation=0)

    for container in axis.containers:
        labels = [f"{bar.get_height():.3f}" for bar in container]
        axis.bar_label(container, labels=labels, fontsize=8, padding=3)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    summaries = [summarize(path) for path in args.results]
    summary = pd.DataFrame(summaries)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False, float_format="%.6f")
    save_chart(summary, args.chart)

    columns = ["Run", "Rows", *METRICS]
    print(summary[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nChart: {args.chart}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()

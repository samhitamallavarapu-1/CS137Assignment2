import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PART2_OUTPUTS = PROJECT_ROOT / "part2_outputs"
SUMMARY_JSON_PATH = PART2_OUTPUTS / "confidence_analysis_summary.json"

CORRECT_DIR = PART2_OUTPUTS / "confidently_correct"
INCORRECT_DIR = PART2_OUTPUTS / "confidently_incorrect"

OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "confidence_saliency_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def load_summary_json():
    if not SUMMARY_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing summary JSON: {SUMMARY_JSON_PATH}")

    with open(SUMMARY_JSON_PATH, "r") as f:
        return json.load(f)


def load_group_csvs(group_name, examples):
    """
    examples: list of dicts from confidence_analysis_summary.json
    returns:
      combined long dataframe with columns like:
      [example_id, t_idx, timestamp, feature, saliency, group]
    """
    rows = []

    for i, ex in enumerate(examples, start=1):
        csv_rel_path = ex.get("csv_path")
        if csv_rel_path is None:
            print(f"Skipping {group_name} example {i}: no csv_path in JSON")
            continue

        csv_path = PROJECT_ROOT / csv_rel_path
        if not csv_path.exists():
            print(f"Skipping missing CSV: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # standardize columns just in case
        expected_cols = {"feature", "saliency"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(
                f"CSV {csv_path} is missing expected columns {expected_cols}. "
                f"Found: {list(df.columns)}"
            )

        df = df.copy()
        df["group"] = group_name
        df["example_rank_in_group"] = i
        df["t_idx"] = ex.get("t_idx")
        df["timestamp"] = ex.get("timestamp")
        df["true_apcp_value"] = ex.get("true_apcp_value")
        df["pred_apcp_value"] = ex.get("pred_apcp_value")
        df["true_label"] = ex.get("true_label")
        df["pred_label"] = ex.get("pred_label")
        df["confidence"] = ex.get("confidence")
        df["correct"] = ex.get("correct")
        df["example_id"] = f"{i:02d}_{group_name}_tidx_{ex.get('t_idx')}"

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    return combined


def summarize_group(feature_df):
    """
    feature_df = long dataframe of all features across all examples in one group

    returns:
      per-feature summary dataframe
    """
    if feature_df.empty:
        return pd.DataFrame()

    summary = (
        feature_df.groupby("feature", as_index=False)
        .agg(
            mean_saliency=("saliency", "mean"),
            median_saliency=("saliency", "median"),
            max_saliency=("saliency", "max"),
            min_saliency=("saliency", "min"),
            std_saliency=("saliency", "std"),
            n_rows=("saliency", "size"),
        )
        .sort_values("mean_saliency", ascending=False)
        .reset_index(drop=True)
    )

    summary["std_saliency"] = summary["std_saliency"].fillna(0.0)
    return summary


def top_feature_frequency(feature_df, top_k=10):
    """
    Count how often each feature appears in the top_k within each example.
    """
    if feature_df.empty:
        return pd.DataFrame(columns=["feature", "top_k_count"])

    counts = Counter()

    for example_id, ex_df in feature_df.groupby("example_id"):
        top_df = ex_df.sort_values("saliency", ascending=False).head(top_k)
        for feat in top_df["feature"]:
            counts[feat] += 1

    freq_df = pd.DataFrame(
        [{"feature": feat, "top_k_count": count} for feat, count in counts.items()]
    ).sort_values(["top_k_count", "feature"], ascending=[False, True]).reset_index(drop=True)

    return freq_df


def make_comparison_table(correct_summary, incorrect_summary):
    """
    Merge per-feature summaries from correct and incorrect groups.
    """
    merged = pd.merge(
        correct_summary[["feature", "mean_saliency", "median_saliency"]],
        incorrect_summary[["feature", "mean_saliency", "median_saliency"]],
        on="feature",
        how="outer",
        suffixes=("_correct", "_incorrect"),
    ).fillna(0.0)

    merged["mean_saliency_diff_correct_minus_incorrect"] = (
        merged["mean_saliency_correct"] - merged["mean_saliency_incorrect"]
    )
    merged["abs_mean_saliency_diff"] = merged[
        "mean_saliency_diff_correct_minus_incorrect"
    ].abs()

    merged = merged.sort_values("abs_mean_saliency_diff", ascending=False).reset_index(drop=True)
    return merged


def plot_top_features(summary_df, title, output_path, top_n=20):
    if summary_df.empty:
        print(f"Skipping empty plot: {title}")
        return

    plot_df = summary_df.head(top_n).copy()
    plot_df = plot_df.iloc[::-1]  # reverse for nicer horizontal bars

    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["feature"], plot_df["mean_saliency"])
    plt.xlabel("Mean saliency")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_frequency(freq_df, title, output_path, top_n=20):
    if freq_df.empty:
        print(f"Skipping empty plot: {title}")
        return

    plot_df = freq_df.head(top_n).copy()
    plot_df = plot_df.iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["feature"], plot_df["top_k_count"])
    plt.xlabel("Count of appearances in top-k")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_difference_table(diff_df, title, output_path, top_n=20):
    if diff_df.empty:
        print(f"Skipping empty plot: {title}")
        return

    plot_df = diff_df.head(top_n).copy()
    plot_df = plot_df.iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["feature"], plot_df["mean_saliency_diff_correct_minus_incorrect"])
    plt.xlabel("Mean saliency difference (correct - incorrect)")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_text_summary(payload, correct_features, incorrect_features,
                      correct_summary, incorrect_summary,
                      correct_freq, incorrect_freq,
                      comparison_df):
    out_path = OUTPUT_DIR / "analysis_summary.txt"

    n_correct = len(payload.get("confidently_correct", []))
    n_incorrect = len(payload.get("confidently_incorrect", []))

    with open(out_path, "w") as f:
        f.write("CONFIDENCE SALIENCY ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Validation accuracy APCP: {payload.get('validation_accuracy_apcp')}\n")
        f.write(f"Validation examples: {payload.get('n_validation_examples')}\n")
        f.write(f"Confidently correct examples: {n_correct}\n")
        f.write(f"Confidently incorrect examples: {n_incorrect}\n\n")

        f.write("Top 10 features by mean saliency: confidently_correct\n")
        f.write(correct_summary.head(10).to_string(index=False))
        f.write("\n\n")

        f.write("Top 10 features by mean saliency: confidently_incorrect\n")
        f.write(incorrect_summary.head(10).to_string(index=False))
        f.write("\n\n")

        f.write("Top 10 most frequent top-10 features: confidently_correct\n")
        f.write(correct_freq.head(10).to_string(index=False))
        f.write("\n\n")

        f.write("Top 10 most frequent top-10 features: confidently_incorrect\n")
        f.write(incorrect_freq.head(10).to_string(index=False))
        f.write("\n\n")

        f.write("Top 15 features with largest absolute mean saliency difference\n")
        f.write(comparison_df.head(15).to_string(index=False))
        f.write("\n")

    print(f"Saved text summary to {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    payload = load_summary_json()

    correct_examples = payload.get("confidently_correct", [])
    incorrect_examples = payload.get("confidently_incorrect", [])

    print(f"Loaded {len(correct_examples)} confidently_correct examples")
    print(f"Loaded {len(incorrect_examples)} confidently_incorrect examples")

    correct_features = load_group_csvs("confidently_correct", correct_examples)
    incorrect_features = load_group_csvs("confidently_incorrect", incorrect_examples)

    if correct_features.empty:
        print("WARNING: No confidently_correct CSVs were loaded.")
    if incorrect_features.empty:
        print("WARNING: No confidently_incorrect CSVs were loaded.")

    # Save raw combined tables
    if not correct_features.empty:
        correct_features.to_csv(OUTPUT_DIR / "correct_all_feature_saliency_long.csv", index=False)
    if not incorrect_features.empty:
        incorrect_features.to_csv(OUTPUT_DIR / "incorrect_all_feature_saliency_long.csv", index=False)

    # Summaries
    correct_summary = summarize_group(correct_features)
    incorrect_summary = summarize_group(incorrect_features)

    correct_freq = top_feature_frequency(correct_features, top_k=10)
    incorrect_freq = top_feature_frequency(incorrect_features, top_k=10)

    comparison_df = make_comparison_table(correct_summary, incorrect_summary)

    # Save CSVs
    correct_summary.to_csv(OUTPUT_DIR / "correct_feature_summary.csv", index=False)
    incorrect_summary.to_csv(OUTPUT_DIR / "incorrect_feature_summary.csv", index=False)
    correct_freq.to_csv(OUTPUT_DIR / "correct_top10_frequency.csv", index=False)
    incorrect_freq.to_csv(OUTPUT_DIR / "incorrect_top10_frequency.csv", index=False)
    comparison_df.to_csv(OUTPUT_DIR / "correct_vs_incorrect_feature_comparison.csv", index=False)

    # Plots
    plot_top_features(
        correct_summary,
        "Top Features by Mean Saliency: Confidently Correct",
        OUTPUT_DIR / "top_features_correct.png",
        top_n=20,
    )

    plot_top_features(
        incorrect_summary,
        "Top Features by Mean Saliency: Confidently Incorrect",
        OUTPUT_DIR / "top_features_incorrect.png",
        top_n=20,
    )

    plot_top_frequency(
        correct_freq,
        "Most Frequent Top-10 Features: Confidently Correct",
        OUTPUT_DIR / "top10_frequency_correct.png",
        top_n=20,
    )

    plot_top_frequency(
        incorrect_freq,
        "Most Frequent Top-10 Features: Confidently Incorrect",
        OUTPUT_DIR / "top10_frequency_incorrect.png",
        top_n=20,
    )

    plot_difference_table(
        comparison_df,
        "Largest Mean Saliency Differences (Correct - Incorrect)",
        OUTPUT_DIR / "feature_difference_correct_minus_incorrect.png",
        top_n=20,
    )

    save_text_summary(
        payload=payload,
        correct_features=correct_features,
        incorrect_features=incorrect_features,
        correct_summary=correct_summary,
        incorrect_summary=incorrect_summary,
        correct_freq=correct_freq,
        incorrect_freq=incorrect_freq,
        comparison_df=comparison_df,
    )

    print("\nDone.")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
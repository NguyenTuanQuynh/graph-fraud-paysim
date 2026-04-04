import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.io_utils import load_config, resolve_path


def save_runtime_summary(df: pd.DataFrame, output_dir: Path):
    keep_cols = [
        "experiment_id",
        "master",
        "step_max",
        "subset_rows",
        "vertex_count",
        "edge_count",
        "total_runtime_sec",
        "02_make_dev_subset_sec",
        "03_build_graph_tables_sec",
        "04_extract_tabular_features_sec",
        "05_extract_graph_features_sec",
        "06_build_training_dataset_sec",
        "07_train_model_sec",
        "status",
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]
    out_df = df[existing_cols].sort_values(["master", "step_max"])
    out_df.to_csv(output_dir / "scaling_runtime_summary.csv", index=False)


def save_model_metrics_summary(day3_metrics: list, output_dir: Path):
    df = pd.DataFrame(day3_metrics)
    df.to_csv(output_dir / "day3_model_metrics_summary.csv", index=False)


def plot_total_runtime(df: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(10, 6))
    for master, grp in df.groupby("master"):
        grp = grp.sort_values("step_max")
        plt.plot(grp["step_max"], grp["total_runtime_sec"], marker="o", label=master)

    plt.xlabel("Max step in dev subset")
    plt.ylabel("Total runtime (seconds)")
    plt.title("Scaling experiment: total runtime vs data size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_total_runtime_vs_step.png")
    plt.close()


def plot_graph_feature_runtime(df: pd.DataFrame, output_dir: Path):
    col = "05_extract_graph_features_sec"
    if col not in df.columns:
        return

    plt.figure(figsize=(10, 6))
    for master, grp in df.groupby("master"):
        grp = grp.sort_values("step_max")
        plt.plot(grp["step_max"], grp[col], marker="o", label=master)

    plt.xlabel("Max step in dev subset")
    plt.ylabel("Job 05 runtime (seconds)")
    plt.title("Scaling experiment: graph feature extraction runtime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_job05_runtime_vs_step.png")
    plt.close()


def plot_model_metric(day3_metrics: list, metric_name: str, filename: str, title: str, output_dir: Path):
    df = pd.DataFrame(day3_metrics)

    plt.figure(figsize=(10, 6))
    plt.bar(df["run_name"], df[metric_name])
    plt.xticks(rotation=15)
    plt.ylabel(metric_name)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    reports_dir = Path(resolve_path(PROJECT_ROOT, cfg["paths"]["reports"]))
    output_dir = reports_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    scaling_csv = reports_dir / "scaling_results.csv"
    day3_json = reports_dir / "day3_graph_enhanced_metrics.json"

    if not scaling_csv.exists():
        raise FileNotFoundError(f"Scaling results CSV not found: {scaling_csv}")

    if not day3_json.exists():
        raise FileNotFoundError(f"Day 3 metrics JSON not found: {day3_json}")

    scaling_df = pd.read_csv(scaling_csv)
    with day3_json.open("r", encoding="utf-8") as f:
        day3_metrics = json.load(f)

    save_runtime_summary(scaling_df, output_dir)
    save_model_metrics_summary(day3_metrics, output_dir)

    plot_total_runtime(scaling_df, output_dir)
    plot_graph_feature_runtime(scaling_df, output_dir)

    plot_model_metric(
        day3_metrics,
        metric_name="auc_pr",
        filename="day3_auc_pr_comparison.png",
        title="Day 3 model comparison: AUC-PR",
        output_dir=output_dir
    )

    plot_model_metric(
        day3_metrics,
        metric_name="recall_at_0_5",
        filename="day3_recall_comparison.png",
        title="Day 3 model comparison: Recall@0.5",
        output_dir=output_dir
    )

    plot_model_metric(
        day3_metrics,
        metric_name="precision_at_0_5",
        filename="day3_precision_comparison.png",
        title="Day 3 model comparison: Precision@0.5",
        output_dir=output_dir
    )

    print("[DONE] Export report assets finished.")
    print(f"[INFO] Assets written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/config_dev.yaml")
    args = parser.parse_args()

    main(args.config)
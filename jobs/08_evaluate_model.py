import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.io_utils import load_config, resolve_path


def load_json_if_exists(path: Path):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    reports_dir = Path(resolve_path(PROJECT_ROOT, cfg["paths"]["reports"]))
    reports_dir.mkdir(parents=True, exist_ok=True)

    day2_path = reports_dir / "day2_baseline_metrics.json"
    day3_path = reports_dir / "day3_graph_enhanced_metrics.json"

    day2_metrics = load_json_if_exists(day2_path)
    day3_metrics = load_json_if_exists(day3_path)

    if day2_metrics is None and day3_metrics is None:
        raise FileNotFoundError(
            "Không tìm thấy day2_baseline_metrics.json hoặc day3_graph_enhanced_metrics.json trong data/reports"
        )

    rows = []

    if day2_metrics is not None:
        for item in day2_metrics:
            row = dict(item)
            row["source"] = "day2"
            rows.append(row)

    if day3_metrics is not None:
        for item in day3_metrics:
            row = dict(item)
            row["source"] = "day3"
            rows.append(row)

    df = pd.DataFrame(rows)

    # bỏ trùng theo run_name, ưu tiên kết quả day3 nếu cùng tên
    if "run_name" in df.columns:
        df["source_priority"] = df["source"].map({"day2": 1, "day3": 2}).fillna(0)
        df = df.sort_values(["run_name", "source_priority"]).drop_duplicates(
            subset=["run_name"], keep="last"
        )
        df = df.drop(columns=["source_priority"])

    comparison_csv = reports_dir / "final_model_comparison.csv"
    df.to_csv(comparison_csv, index=False)

    best_auc_pr = df.loc[df["auc_pr"].idxmax()].to_dict() if not df.empty else None
    best_recall = df.loc[df["recall_at_0_5"].idxmax()].to_dict() if not df.empty else None
    best_f1 = df.loc[df["f1_at_0_5"].idxmax()].to_dict() if not df.empty else None

    summary = {
        "num_models_compared": int(len(df)),
        "best_by_auc_pr": best_auc_pr,
        "best_by_recall_at_0_5": best_recall,
        "best_by_f1_at_0_5": best_f1,
    }

    summary_json = reports_dir / "final_evaluation_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[INFO] Final model comparison table")
    print(df.to_string(index=False))

    print("\n[INFO] Best models")
    print(json.dumps(summary, indent=2))

    print(f"\n[DONE] Evaluation summary written to: {summary_json}")
    print(f"[DONE] Comparison CSV written to: {comparison_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/config_dev.yaml")
    args = parser.parse_args()

    main(args.config)
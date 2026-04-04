import argparse
import copy
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.io_utils import load_config, resolve_path


JOB_SEQUENCE = [
    "02_make_dev_subset.py",
    "03_build_graph_tables.py",
    "04_extract_tabular_features.py",
    "05_extract_graph_features.py",
    "06_build_training_dataset.py",
    "07_train_model.py",
]


def sanitize_master(master: str) -> str:
    return master.replace("[", "").replace("]", "").replace("*", "star").replace(":", "_")


def parse_float_from_log(pattern: str, text: str):
    match = re.search(pattern, text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def parse_int_from_log(pattern: str, text: str):
    match = re.search(pattern, text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def load_json_if_exists(path: Path):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def make_experiment_config(base_cfg: dict, exp_id: str, master: str, step_max: int, shuffle_partitions: int):
    cfg = copy.deepcopy(base_cfg)

    cfg["master"] = master
    cfg["dev"]["min_step"] = 1
    cfg["dev"]["max_step"] = step_max
    cfg["spark"]["shuffle_partitions"] = shuffle_partitions

    # time-based split proportional to step_max
    train_max = max(1, int(step_max * 0.67))
    val_max = max(train_max + 1, int(step_max * 0.83))
    val_max = min(val_max, step_max - 1) if step_max > 2 else step_max

    cfg["split"]["train_max_step"] = train_max
    cfg["split"]["val_max_step"] = val_max
    cfg["split"]["test_max_step"] = step_max

    exp_root = f"data/scaling/{exp_id}"

    cfg["paths"]["silver_dev"] = f"{exp_root}/silver/paysim_dev_subset"
    cfg["paths"]["vertices_dev"] = f"{exp_root}/graph/dev_vertices"
    cfg["paths"]["edges_dev"] = f"{exp_root}/graph/dev_edges"
    cfg["paths"]["tabular_features_dev"] = f"{exp_root}/features/dev_tabular_features"
    cfg["paths"]["degree_features_dev"] = f"{exp_root}/features/dev_degree_features"
    cfg["paths"]["pagerank_nodes_dev"] = f"{exp_root}/features/dev_pagerank_nodes"
    cfg["paths"]["pagerank_features_dev"] = f"{exp_root}/features/dev_pagerank_features"
    cfg["paths"]["training_dataset_dev"] = f"{exp_root}/train/dev_training_dataset"
    cfg["paths"]["reports"] = f"{exp_root}/reports"

    return cfg


def run_job(job_filename: str, config_path: Path, log_path: Path):
    cmd = [sys.executable, str(PROJECT_ROOT / "jobs" / job_filename), "--config", str(config_path)]
    start = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )

    elapsed = time.perf_counter() - start
    return proc.returncode, elapsed


def main(config_path: str) -> None:
    base_cfg = load_config(config_path)

    reports_root = Path(resolve_path(PROJECT_ROOT, base_cfg["paths"]["reports"]))
    reports_root.mkdir(parents=True, exist_ok=True)

    scaling_cfg = base_cfg["scaling"]
    masters = scaling_cfg["masters"]
    step_max_values = scaling_cfg["step_max_values"]
    shuffle_map = scaling_cfg["shuffle_partitions"]

    temp_cfg_dir = reports_root / "scaling_temp_configs"
    summary_dir = reports_root / "scaling_logs"
    temp_cfg_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for master in masters:
        for step_max in step_max_values:
            master_tag = sanitize_master(master)
            exp_id = f"exp_{master_tag}_step{step_max}"
            print(f"[INFO] Running experiment: {exp_id}")

            shuffle_partitions = int(shuffle_map.get(master, base_cfg["spark"]["shuffle_partitions"]))
            exp_cfg = make_experiment_config(
                base_cfg=base_cfg,
                exp_id=exp_id,
                master=master,
                step_max=step_max,
                shuffle_partitions=shuffle_partitions
            )

            exp_cfg_path = temp_cfg_dir / f"{exp_id}.yaml"
            with exp_cfg_path.open("w", encoding="utf-8") as f:
                import yaml
                yaml.safe_dump(exp_cfg, f, sort_keys=False)

            exp_log_dir = summary_dir / exp_id
            exp_log_dir.mkdir(parents=True, exist_ok=True)

            result_row = {
                "experiment_id": exp_id,
                "master": master,
                "step_max": step_max,
                "shuffle_partitions": shuffle_partitions,
                "status": "SUCCESS",
            }

            total_start = time.perf_counter()

            for job_filename in JOB_SEQUENCE:
                job_name = job_filename.replace(".py", "")
                log_path = exp_log_dir / f"{job_name}.log"

                return_code, elapsed = run_job(job_filename, exp_cfg_path, log_path)
                result_row[f"{job_name}_sec"] = round(elapsed, 4)

                if return_code != 0:
                    result_row["status"] = "FAILED"
                    result_row["failed_job"] = job_name
                    break

            total_elapsed = time.perf_counter() - total_start
            result_row["total_runtime_sec"] = round(total_elapsed, 4)

            # Parse summary info from logs if available
            log_02 = (exp_log_dir / "02_make_dev_subset.log")
            log_03 = (exp_log_dir / "03_build_graph_tables.log")
            log_07 = Path(resolve_path(PROJECT_ROOT, exp_cfg["paths"]["reports"])) / "day3_graph_enhanced_metrics.json"

            if log_02.exists():
                txt = log_02.read_text(encoding="utf-8", errors="ignore")
                result_row["subset_rows"] = parse_int_from_log(r"total_rows\s*=\s*(\d+)", txt)
                result_row["subset_fraud_rows"] = parse_int_from_log(r"fraud_rows\s*=\s*(\d+)", txt)

            if log_03.exists():
                txt = log_03.read_text(encoding="utf-8", errors="ignore")
                result_row["vertex_count"] = parse_int_from_log(r"vertex_count\s*=\s*(\d+)", txt)
                result_row["edge_count"] = parse_int_from_log(r"edge_count\s*=\s*(\d+)", txt)

            metrics = load_json_if_exists(log_07)
            if metrics:
                for item in metrics:
                    run_name = item["run_name"]
                    prefix = run_name
                    result_row[f"{prefix}_auc_pr"] = item["auc_pr"]
                    result_row[f"{prefix}_auc_roc"] = item["auc_roc"]
                    result_row[f"{prefix}_precision_at_0_5"] = item["precision_at_0_5"]
                    result_row[f"{prefix}_recall_at_0_5"] = item["recall_at_0_5"]
                    result_row[f"{prefix}_f1_at_0_5"] = item["f1_at_0_5"]

            all_results.append(result_row)
            print(f"[INFO] Finished {exp_id} | status={result_row['status']} | total_runtime_sec={result_row['total_runtime_sec']}")

    # Write JSON
    json_path = reports_root / "scaling_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Write CSV
    all_keys = sorted({k for row in all_results for k in row.keys()})
    csv_path = reports_root / "scaling_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"[DONE] Scaling experiments finished.")
    print(f"[INFO] JSON written to: {json_path}")
    print(f"[INFO] CSV written to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/config_dev.yaml")
    args = parser.parse_args()

    main(args.config)
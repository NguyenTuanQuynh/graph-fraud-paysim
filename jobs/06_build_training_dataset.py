import argparse
import sys
from pathlib import Path
from time import perf_counter

from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.io_utils import load_config, resolve_path, ensure_parent_dir
from libs.spark_utils import build_spark_session


def main(config_path: str) -> None:
    start = perf_counter()

    cfg = load_config(config_path)
    spark = build_spark_session(cfg)

    tabular_path = resolve_path(PROJECT_ROOT, cfg["paths"]["tabular_features_dev"])
    degree_path = resolve_path(PROJECT_ROOT, cfg["paths"]["degree_features_dev"])
    pagerank_path = resolve_path(PROJECT_ROOT, cfg["paths"]["pagerank_features_dev"])
    output_path = resolve_path(PROJECT_ROOT, cfg["paths"]["training_dataset_dev"])
    ensure_parent_dir(output_path)

    print(f"[INFO] Reading tabular features from: {tabular_path}")
    tabular_df = spark.read.parquet(tabular_path)

    print(f"[INFO] Reading degree features from: {degree_path}")
    degree_df = spark.read.parquet(degree_path)

    print(f"[INFO] Reading PageRank features from: {pagerank_path}")
    pagerank_df = spark.read.parquet(pagerank_path)

    train_df = (
        tabular_df.join(degree_df, on="txn_id", how="left")
        .join(pagerank_df, on="txn_id", how="left")
        .fillna(0, subset=[
            "src_total_out_degree",
            "dst_total_in_degree",
            "src_step_txn_count",
            "dst_step_txn_count",
            "src_prev_steps_out_count",
            "dst_prev_steps_in_count",
        ])
        .fillna(0.0, subset=[
            "src_pagerank",
            "dst_pagerank",
        ])
    )

    total_rows = train_df.count()
    fraud_rows = train_df.filter(F.col("isFraud") == 1).count()

    print("[INFO] Training dataset summary")
    print(f"       total_rows = {total_rows}")
    print(f"       fraud_rows = {fraud_rows}")

    print(f"[INFO] Writing training dataset to: {output_path}")
    train_df.write.mode("overwrite").parquet(output_path)

    spark.stop()
    elapsed = perf_counter() - start
    print(f"[DONE] 06_build_training_dataset finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/config_dev.yaml")
    args = parser.parse_args()

    main(args.config)
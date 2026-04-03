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

    bronze_path = resolve_path(PROJECT_ROOT, cfg["paths"]["bronze"])
    silver_dev_path = resolve_path(PROJECT_ROOT, cfg["paths"]["silver_dev"])
    ensure_parent_dir(silver_dev_path)

    min_step = int(cfg["dev"]["min_step"])
    max_step = int(cfg["dev"]["max_step"])

    if min_step > max_step:
        raise ValueError("dev.min_step must be <= dev.max_step")

    print(f"[INFO] Reading bronze parquet from: {bronze_path}")
    df = spark.read.parquet(bronze_path)

    subset_df = df.filter((F.col("step") >= min_step) & (F.col("step") <= max_step))
    subset_df.cache()

    total_rows = subset_df.count()
    fraud_rows = subset_df.filter(F.col("isFraud") == 1).count()
    actual_min_step, actual_max_step = subset_df.select(F.min("step"), F.max("step")).first()

    if total_rows == 0:
        raise ValueError(
            f"Dev subset is empty. Hãy kiểm tra lại khoảng step: [{min_step}, {max_step}]"
        )

    print("[INFO] Dev subset summary")
    print(f"       requested_step_range = [{min_step}, {max_step}]")
    print(f"       actual_step_range    = [{actual_min_step}, {actual_max_step}]")
    print(f"       total_rows           = {total_rows}")
    print(f"       fraud_rows           = {fraud_rows}")

    print(f"[INFO] Writing dev subset parquet to: {silver_dev_path}")
    subset_df.write.mode("overwrite").parquet(silver_dev_path)

    subset_df.unpersist()
    spark.stop()

    elapsed = perf_counter() - start
    print(f"[DONE] 02_make_dev_subset finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/config_dev.yaml")
    args = parser.parse_args()

    main(args.config)
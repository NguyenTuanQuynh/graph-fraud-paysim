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

    silver_dev_path = resolve_path(PROJECT_ROOT, cfg["paths"]["silver_dev"])
    output_path = resolve_path(PROJECT_ROOT, cfg["paths"]["tabular_features_dev"])
    ensure_parent_dir(output_path)

    print(f"[INFO] Reading dev subset from: {silver_dev_path}")
    df = spark.read.parquet(silver_dev_path)

    features_df = (
        df.select(
            "txn_id",
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
        )
        .withColumn("deltaOrig", F.col("oldbalanceOrg") - F.col("newbalanceOrig"))
        .withColumn("deltaDest", F.col("newbalanceDest") - F.col("oldbalanceDest"))
        .withColumn(
            "orig_zero_before_txn",
            F.when(F.col("oldbalanceOrg") == 0, F.lit(1)).otherwise(F.lit(0))
        )
        .withColumn(
            "dest_zero_before_txn",
            F.when(F.col("oldbalanceDest") == 0, F.lit(1)).otherwise(F.lit(0))
        )
        .withColumn(
            "amount_to_oldbalanceOrg_ratio",
            F.when(F.col("oldbalanceOrg") > 0, F.col("amount") / F.col("oldbalanceOrg"))
             .otherwise(F.lit(0.0))
        )
        .withColumn(
            "amount_to_oldbalanceDest_ratio",
            F.when(F.col("oldbalanceDest") > 0, F.col("amount") / F.col("oldbalanceDest"))
             .otherwise(F.lit(0.0))
        )
    )

    total_rows = features_df.count()
    fraud_rows = features_df.filter(F.col("isFraud") == 1).count()

    print("[INFO] Tabular feature summary")
    print(f"       total_rows = {total_rows}")
    print(f"       fraud_rows = {fraud_rows}")

    print(f"[INFO] Writing tabular features to: {output_path}")
    features_df.write.mode("overwrite").parquet(output_path)

    spark.stop()
    elapsed = perf_counter() - start
    print(f"[DONE] 04_extract_tabular_features finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/config_dev.yaml")
    args = parser.parse_args()

    main(args.config)
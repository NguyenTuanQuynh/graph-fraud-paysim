import argparse
import sys
from pathlib import Path
from time import perf_counter

from pyspark.sql import functions as F
from pyspark.sql import types as T

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.io_utils import load_config, resolve_path, ensure_parent_dir
from libs.spark_utils import build_spark_session


REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def get_paysim_schema() -> T.StructType:
    return T.StructType([
        T.StructField("step", T.IntegerType(), True),
        T.StructField("type", T.StringType(), True),
        T.StructField("amount", T.DoubleType(), True),
        T.StructField("nameOrig", T.StringType(), True),
        T.StructField("oldbalanceOrg", T.DoubleType(), True),
        T.StructField("newbalanceOrig", T.DoubleType(), True),
        T.StructField("nameDest", T.StringType(), True),
        T.StructField("oldbalanceDest", T.DoubleType(), True),
        T.StructField("newbalanceDest", T.DoubleType(), True),
        T.StructField("isFraud", T.IntegerType(), True),
        T.StructField("isFlaggedFraud", T.IntegerType(), True),
    ])


def main(config_path: str) -> None:
    start = perf_counter()

    cfg = load_config(config_path)
    spark = build_spark_session(cfg)

    raw_csv_path = resolve_path(PROJECT_ROOT, cfg["paths"]["raw_csv"])
    bronze_path = resolve_path(PROJECT_ROOT, cfg["paths"]["bronze"])
    ensure_parent_dir(bronze_path)

    if not Path(raw_csv_path).exists():
        raise FileNotFoundError(
            f"Raw CSV not found at: {raw_csv_path}\n"
            f"Hãy copy file PaySim vào đúng đường dẫn này hoặc sửa config."
        )

    print(f"[INFO] Reading raw CSV from: {raw_csv_path}")

    df = (
        spark.read
        .option("header", "true")
        .schema(get_paysim_schema())
        .csv(raw_csv_path)
    )

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = (
        df.select(*REQUIRED_COLUMNS)
        .withColumn("type", F.upper(F.trim(F.col("type"))))
        .filter(
            F.col("step").isNotNull() &
            F.col("nameOrig").isNotNull() &
            F.col("nameDest").isNotNull()
        )
    )

    hash_inputs = [
        F.coalesce(F.col(c).cast("string"), F.lit("NULL"))
        for c in REQUIRED_COLUMNS
    ]

    df = df.withColumn("txn_id", F.sha2(F.concat_ws("||", *hash_inputs), 256))
    df = df.select("txn_id", *REQUIRED_COLUMNS)

    print("[INFO] Computing lightweight dataset summary...")

    summary = df.agg(
        F.count("*").alias("total_rows"),
        F.sum(F.col("isFraud")).alias("fraud_rows"),
        F.min("step").alias("min_step"),
        F.max("step").alias("max_step"),
        F.approx_count_distinct("nameOrig").alias("approx_distinct_senders"),
        F.approx_count_distinct("nameDest").alias("approx_distinct_receivers"),
    ).collect()[0]

    print("[INFO] Raw dataset summary")
    print(f"       total_rows                = {summary['total_rows']}")
    print(f"       fraud_rows                = {summary['fraud_rows']}")
    print(f"       min_step                  = {summary['min_step']}")
    print(f"       max_step                  = {summary['max_step']}")
    print(f"       approx_distinct_senders   = {summary['approx_distinct_senders']}")
    print(f"       approx_distinct_receivers = {summary['approx_distinct_receivers']}")

    print(f"[INFO] Writing bronze parquet to: {bronze_path}")
    df.write.mode("overwrite").parquet(bronze_path)

    spark.stop()

    elapsed = perf_counter() - start
    print(f"[DONE] 01_ingest_raw finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/config_dev.yaml")
    args = parser.parse_args()

    main(args.config)
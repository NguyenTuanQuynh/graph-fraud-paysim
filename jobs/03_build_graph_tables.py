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
    vertices_path = resolve_path(PROJECT_ROOT, cfg["paths"]["vertices_dev"])
    edges_path = resolve_path(PROJECT_ROOT, cfg["paths"]["edges_dev"])

    ensure_parent_dir(vertices_path)
    ensure_parent_dir(edges_path)

    print(f"[INFO] Reading dev subset from: {silver_dev_path}")
    df = spark.read.parquet(silver_dev_path)

    # ---- Build vertices ----
    senders = (
        df.select(F.col("nameOrig").alias("id"))
        .distinct()
        .withColumn("is_sender", F.lit(1))
    )

    receivers = (
        df.select(F.col("nameDest").alias("id"))
        .distinct()
        .withColumn("is_receiver", F.lit(1))
    )

    vertices = (
        senders.join(receivers, on="id", how="full_outer")
        .fillna(0, subset=["is_sender", "is_receiver"])
        .withColumn("is_sender", F.col("is_sender").cast("int"))
        .withColumn("is_receiver", F.col("is_receiver").cast("int"))
        .withColumn(
            "vertex_role",
            F.when((F.col("is_sender") == 1) & (F.col("is_receiver") == 1), F.lit("both"))
             .when(F.col("is_sender") == 1, F.lit("sender_only"))
             .otherwise(F.lit("receiver_only"))
        )
    )

    # ---- Build edges ----
    edges = (
        df.select(
            "txn_id",
            F.col("nameOrig").alias("src"),
            F.col("nameDest").alias("dst"),
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
            "isFlaggedFraud",
        )
        .filter(F.col("src").isNotNull() & F.col("dst").isNotNull())
    )

    vertices.cache()
    edges.cache()

    vertex_count = vertices.count()
    edge_count = edges.count()
    self_loops = edges.filter(F.col("src") == F.col("dst")).count()
    distinct_pairs = edges.select("src", "dst").distinct().count()

    print("[INFO] Graph summary")
    print(f"       vertex_count   = {vertex_count}")
    print(f"       edge_count     = {edge_count}")
    print(f"       self_loops     = {self_loops}")
    print(f"       distinct_pairs = {distinct_pairs}")

    print(f"[INFO] Writing vertices to: {vertices_path}")
    vertices.write.mode("overwrite").parquet(vertices_path)

    print(f"[INFO] Writing edges to: {edges_path}")
    edges.write.mode("overwrite").parquet(edges_path)

    vertices.unpersist()
    edges.unpersist()
    spark.stop()

    elapsed = perf_counter() - start
    print(f"[DONE] 03_build_graph_tables finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/config_dev.yaml")
    args = parser.parse_args()

    main(args.config)
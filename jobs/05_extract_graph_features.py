import argparse
import sys
from pathlib import Path
from time import perf_counter

from pyspark.sql import Window
from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.io_utils import load_config, resolve_path, ensure_parent_dir
from libs.spark_utils import build_spark_session


def compute_pagerank(train_edges_df, damping: float, max_iter: int):
    """
    Compute PageRank on TRAIN graph only.
    train_edges_df must have columns: src, dst
    Returns: DataFrame(id, pagerank)
    """

    weighted_edges = (
        train_edges_df.groupBy("src", "dst")
        .agg(F.count("*").alias("edge_weight"))
    )

    vertices = (
        weighted_edges.select(F.col("src").alias("id"))
        .union(weighted_edges.select(F.col("dst").alias("id")))
        .distinct()
        .cache()
    )

    num_vertices = vertices.count()
    if num_vertices == 0:
        raise ValueError("No vertices found for PageRank.")

    out_weight = (
        weighted_edges.groupBy("src")
        .agg(F.sum("edge_weight").alias("src_total_weight"))
        .cache()
    )

    edge_probs = (
        weighted_edges.join(out_weight, on="src", how="left")
        .withColumn("transition_prob", F.col("edge_weight") / F.col("src_total_weight"))
        .select("src", "dst", "transition_prob")
        .cache()
    )

    out_nodes = out_weight.select(F.col("src").alias("id")).distinct().cache()

    init_rank = 1.0 / float(num_vertices)
    ranks = vertices.withColumn("pagerank", F.lit(init_rank)).cache()

    print("[INFO] PageRank setup")
    print(f"       num_vertices = {num_vertices}")
    print(f"       damping      = {damping}")
    print(f"       max_iter     = {max_iter}")

    for i in range(max_iter):
        dangling_mass = (
            ranks.join(out_nodes, on="id", how="left_anti")
            .agg(F.sum("pagerank").alias("dangling_mass"))
            .collect()[0]["dangling_mass"]
        )
        dangling_mass = float(dangling_mass) if dangling_mass is not None else 0.0

        dangling_share = damping * dangling_mass / float(num_vertices)
        base_rank = (1.0 - damping) / float(num_vertices) + dangling_share

        contribs = (
            edge_probs.join(
                ranks.withColumnRenamed("id", "src_id"),
                edge_probs["src"] == F.col("src_id"),
                how="inner"
            )
            .select(
                edge_probs["dst"].alias("id"),
                (edge_probs["transition_prob"] * F.col("pagerank")).alias("contrib")
            )
        )

        summed = (
            contribs.groupBy("id")
            .agg(F.sum("contrib").alias("contrib_sum"))
        )

        new_ranks = (
            vertices.join(summed, on="id", how="left")
            .fillna(0.0, subset=["contrib_sum"])
            .withColumn("pagerank", F.lit(base_rank) + F.lit(damping) * F.col("contrib_sum"))
            .select("id", "pagerank")
            .cache()
        )

        delta = (
            ranks.join(new_ranks.withColumnRenamed("pagerank", "new_pagerank"), on="id", how="inner")
            .agg(F.sum(F.abs(F.col("pagerank") - F.col("new_pagerank"))).alias("l1_delta"))
            .collect()[0]["l1_delta"]
        )

        print(f"[INFO] PageRank iteration {i + 1}/{max_iter} | l1_delta = {float(delta):.8f}")

        ranks.unpersist()
        ranks = new_ranks

    edge_probs.unpersist()
    out_weight.unpersist()
    out_nodes.unpersist()
    vertices.unpersist()

    return ranks


def main(config_path: str) -> None:
    start = perf_counter()

    cfg = load_config(config_path)
    spark = build_spark_session(cfg)

    silver_dev_path = resolve_path(PROJECT_ROOT, cfg["paths"]["silver_dev"])
    edges_path = resolve_path(PROJECT_ROOT, cfg["paths"]["edges_dev"])

    degree_output_path = resolve_path(PROJECT_ROOT, cfg["paths"]["degree_features_dev"])
    pagerank_nodes_path = resolve_path(PROJECT_ROOT, cfg["paths"]["pagerank_nodes_dev"])
    pagerank_features_path = resolve_path(PROJECT_ROOT, cfg["paths"]["pagerank_features_dev"])

    ensure_parent_dir(degree_output_path)
    ensure_parent_dir(pagerank_nodes_path)
    ensure_parent_dir(pagerank_features_path)

    train_max_step = int(cfg["split"]["train_max_step"])
    damping = float(cfg["pagerank"]["damping"])
    max_iter = int(cfg["pagerank"]["max_iter"])

    print(f"[INFO] Reading dev subset from: {silver_dev_path}")
    df = spark.read.parquet(silver_dev_path)

    print(f"[INFO] Reading dev edges from: {edges_path}")
    edges_df = spark.read.parquet(edges_path)

    # =========================
    # PART A: DEGREE FEATURES
    # =========================
    src_total_degree = (
        df.groupBy("nameOrig")
        .agg(F.count("*").alias("src_total_out_degree"))
        .withColumnRenamed("nameOrig", "src_account")
    )

    dst_total_degree = (
        df.groupBy("nameDest")
        .agg(F.count("*").alias("dst_total_in_degree"))
        .withColumnRenamed("nameDest", "dst_account")
    )

    src_step_counts = (
        df.groupBy("nameOrig", "step")
        .agg(F.count("*").alias("src_step_txn_count"))
        .withColumnRenamed("nameOrig", "src_account")
    )

    dst_step_counts = (
        df.groupBy("nameDest", "step")
        .agg(F.count("*").alias("dst_step_txn_count"))
        .withColumnRenamed("nameDest", "dst_account")
    )

    src_hist_base = (
        df.groupBy("nameOrig", "step")
        .agg(F.count("*").alias("src_txn_count_this_step"))
        .withColumnRenamed("nameOrig", "src_account")
    )

    src_hist_window = (
        Window.partitionBy("src_account")
        .orderBy("step")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    src_hist_features = (
        src_hist_base
        .withColumn(
            "src_prev_steps_out_count",
            F.coalesce(F.sum("src_txn_count_this_step").over(src_hist_window), F.lit(0))
        )
        .select("src_account", "step", "src_prev_steps_out_count")
    )

    dst_hist_base = (
        df.groupBy("nameDest", "step")
        .agg(F.count("*").alias("dst_txn_count_this_step"))
        .withColumnRenamed("nameDest", "dst_account")
    )

    dst_hist_window = (
        Window.partitionBy("dst_account")
        .orderBy("step")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    dst_hist_features = (
        dst_hist_base
        .withColumn(
            "dst_prev_steps_in_count",
            F.coalesce(F.sum("dst_txn_count_this_step").over(dst_hist_window), F.lit(0))
        )
        .select("dst_account", "step", "dst_prev_steps_in_count")
    )

    degree_features = (
        df.select(
            "txn_id",
            "step",
            F.col("nameOrig").alias("src_account"),
            F.col("nameDest").alias("dst_account"),
        )
        .join(src_total_degree, on="src_account", how="left")
        .join(dst_total_degree, on="dst_account", how="left")
        .join(src_step_counts, on=["src_account", "step"], how="left")
        .join(dst_step_counts, on=["dst_account", "step"], how="left")
        .join(src_hist_features, on=["src_account", "step"], how="left")
        .join(dst_hist_features, on=["dst_account", "step"], how="left")
        .fillna(0, subset=[
            "src_total_out_degree",
            "dst_total_in_degree",
            "src_step_txn_count",
            "dst_step_txn_count",
            "src_prev_steps_out_count",
            "dst_prev_steps_in_count",
        ])
        .select(
            "txn_id",
            "src_total_out_degree",
            "dst_total_in_degree",
            "src_step_txn_count",
            "dst_step_txn_count",
            "src_prev_steps_out_count",
            "dst_prev_steps_in_count",
        )
    )

    degree_count = degree_features.count()

    print("[INFO] Degree feature summary")
    print(f"       total_rows = {degree_count}")

    print(f"[INFO] Writing degree features to: {degree_output_path}")
    degree_features.write.mode("overwrite").parquet(degree_output_path)

    # =========================
    # PART B: PAGERANK FEATURES
    # =========================
    train_edges_df = edges_df.filter(F.col("step") <= train_max_step).select("src", "dst")

    pagerank_nodes = compute_pagerank(
        train_edges_df=train_edges_df,
        damping=damping,
        max_iter=max_iter
    )

    node_count = pagerank_nodes.count()

    print(f"[INFO] Writing PageRank nodes to: {pagerank_nodes_path}")
    pagerank_nodes.write.mode("overwrite").parquet(pagerank_nodes_path)

    src_ranks = (
        pagerank_nodes
        .withColumnRenamed("id", "src_account")
        .withColumnRenamed("pagerank", "src_pagerank")
    )

    dst_ranks = (
        pagerank_nodes
        .withColumnRenamed("id", "dst_account")
        .withColumnRenamed("pagerank", "dst_pagerank")
    )

    pagerank_features = (
        df.select(
            "txn_id",
            F.col("nameOrig").alias("src_account"),
            F.col("nameDest").alias("dst_account"),
        )
        .join(src_ranks, on="src_account", how="left")
        .join(dst_ranks, on="dst_account", how="left")
        .fillna(0.0, subset=["src_pagerank", "dst_pagerank"])
        .select("txn_id", "src_pagerank", "dst_pagerank")
    )

    pagerank_feature_count = pagerank_features.count()

    print("[INFO] PageRank feature summary")
    print(f"       pagerank_node_count = {node_count}")
    print(f"       pagerank_txn_rows   = {pagerank_feature_count}")

    print(f"[INFO] Writing PageRank features to: {pagerank_features_path}")
    pagerank_features.write.mode("overwrite").parquet(pagerank_features_path)

    spark.stop()
    elapsed = perf_counter() - start
    print(f"[DONE] 05_extract_graph_features finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/config_dev.yaml")
    args = parser.parse_args()

    main(args.config)
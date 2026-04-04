import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.io_utils import load_config, resolve_path
from libs.spark_utils import build_spark_session


TABULAR_NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "deltaOrig",
    "deltaDest",
    "orig_zero_before_txn",
    "dest_zero_before_txn",
    "amount_to_oldbalanceOrg_ratio",
    "amount_to_oldbalanceDest_ratio",
]

DEGREE_FEATURES = [
    "src_total_out_degree",
    "dst_total_in_degree",
    "src_step_txn_count",
    "dst_step_txn_count",
    "src_prev_steps_out_count",
    "dst_prev_steps_in_count",
]

PAGERANK_FEATURES = [
    "src_pagerank",
    "dst_pagerank",
]


def add_split_column(df, train_max_step: int, val_max_step: int):
    return (
        df.withColumn(
            "split",
            F.when(F.col("step") <= train_max_step, F.lit("train"))
             .when(F.col("step") <= val_max_step, F.lit("val"))
             .otherwise(F.lit("test"))
        )
    )


def add_class_weight(df):
    counts = df.groupBy("isFraud").count().collect()
    count_map = {row["isFraud"]: row["count"] for row in counts}

    positive = count_map.get(1, 1)
    negative = count_map.get(0, 1)
    pos_weight = float(negative) / float(max(positive, 1))

    return df.withColumn(
        "class_weight",
        F.when(F.col("isFraud") == 1, F.lit(pos_weight)).otherwise(F.lit(1.0))
    ), pos_weight


def train_and_evaluate(train_df, eval_df, feature_cols, run_name: str):
    indexer = StringIndexer(
        inputCol="type",
        outputCol="type_index",
        handleInvalid="keep"
    )

    encoder = OneHotEncoder(
        inputCols=["type_index"],
        outputCols=["type_ohe"],
        handleInvalid="keep"
    )

    assembler = VectorAssembler(
        inputCols=feature_cols + ["type_ohe"],
        outputCol="features"
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="isFraud",
        weightCol="class_weight",
        maxIter=20,
        regParam=0.01,
        elasticNetParam=0.0
    )

    pipeline = Pipeline(stages=[indexer, encoder, assembler, lr])
    model = pipeline.fit(train_df)

    pred_df = model.transform(eval_df)

    pr_eval = BinaryClassificationEvaluator(
        labelCol="isFraud",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )

    roc_eval = BinaryClassificationEvaluator(
        labelCol="isFraud",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    auc_pr = pr_eval.evaluate(pred_df)
    auc_roc = roc_eval.evaluate(pred_df)

    pred_df = (
        pred_df
        .withColumn("probability_array", vector_to_array(F.col("probability")))
        .withColumn("prob_1", F.col("probability_array")[1])
        .withColumn(
            "pred_label",
            F.when(F.col("prob_1") >= 0.5, F.lit(1)).otherwise(F.lit(0))
        )
    )

    metrics_row = (
        pred_df.agg(
            F.sum(F.when((F.col("pred_label") == 1) & (F.col("isFraud") == 1), 1).otherwise(0)).alias("tp"),
            F.sum(F.when((F.col("pred_label") == 1) & (F.col("isFraud") == 0), 1).otherwise(0)).alias("fp"),
            F.sum(F.when((F.col("pred_label") == 0) & (F.col("isFraud") == 1), 1).otherwise(0)).alias("fn"),
            F.sum(F.when((F.col("pred_label") == 0) & (F.col("isFraud") == 0), 1).otherwise(0)).alias("tn"),
        )
        .collect()[0]
    )

    tp = float(metrics_row["tp"])
    fp = float(metrics_row["fp"])
    fn = float(metrics_row["fn"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "run_name": run_name,
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "f1_at_0_5": f1,
    }


def main(config_path: str) -> None:
    start = perf_counter()

    cfg = load_config(config_path)
    spark = build_spark_session(cfg)

    dataset_path = resolve_path(PROJECT_ROOT, cfg["paths"]["training_dataset_dev"])
    reports_path = resolve_path(PROJECT_ROOT, cfg["paths"]["reports"])

    train_max_step = int(cfg["split"]["train_max_step"])
    val_max_step = int(cfg["split"]["val_max_step"])

    print(f"[INFO] Reading training dataset from: {dataset_path}")
    df = spark.read.parquet(dataset_path)

    df = add_split_column(df, train_max_step, val_max_step)

    train_df = df.filter(F.col("split") == "train")
    val_df = df.filter(F.col("split") == "val")
    test_df = df.filter(F.col("split") == "test")

    train_df, pos_weight = add_class_weight(train_df)
    val_df = val_df.withColumn("class_weight", F.lit(1.0))
    test_df = test_df.withColumn("class_weight", F.lit(1.0))

    print("[INFO] Split summary")
    print(f"       train_rows = {train_df.count()}")
    print(f"       val_rows   = {val_df.count()}")
    print(f"       test_rows  = {test_df.count()}")
    print(f"       positive_class_weight = {pos_weight:.4f}")

    results = []

    results.append(
        train_and_evaluate(
            train_df=train_df,
            eval_df=test_df,
            feature_cols=TABULAR_NUMERIC_FEATURES,
            run_name="baseline_tabular"
        )
    )

    results.append(
        train_and_evaluate(
            train_df=train_df,
            eval_df=test_df,
            feature_cols=TABULAR_NUMERIC_FEATURES + DEGREE_FEATURES,
            run_name="baseline_tabular_plus_degree"
        )
    )

    results.append(
        train_and_evaluate(
            train_df=train_df,
            eval_df=test_df,
            feature_cols=TABULAR_NUMERIC_FEATURES + DEGREE_FEATURES + PAGERANK_FEATURES,
            run_name="baseline_tabular_plus_degree_plus_pagerank"
        )
    )

    print("[INFO] Evaluation results")
    for r in results:
        print(json.dumps(r, indent=2))

    metrics_path = Path(reports_path) / "day3_graph_enhanced_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Metrics written to: {metrics_path}")

    spark.stop()
    elapsed = perf_counter() - start
    print(f"[DONE] 07_train_model finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/config_dev.yaml")
    args = parser.parse_args()

    main(args.config)
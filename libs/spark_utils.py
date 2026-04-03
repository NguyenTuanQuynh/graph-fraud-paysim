from pyspark.sql import SparkSession


def build_spark_session(cfg: dict) -> SparkSession:
    spark_cfg = cfg.get("spark", {})

    builder = (
        SparkSession.builder
        .appName(cfg.get("app_name", "graph-fraud-paysim"))
        .master(cfg.get("master", "local[2]"))
        .config("spark.sql.shuffle.partitions", str(spark_cfg.get("shuffle_partitions", 4)))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", spark_cfg.get("driver_memory", "4g"))
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(spark_cfg.get("log_level", "WARN"))
    return spark
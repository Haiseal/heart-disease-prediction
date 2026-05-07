# pyspark_day2.py — chạy trên Hadoop terminal
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ── 1. Create SparkSession ──────────────────────────────────────
spark = SparkSession.builder \
    .appName("HeartCare_BRFSS") \
    .getOrCreate()

print("✅ Spark version:", spark.version)
spark.sparkContext.setLogLevel("ERROR")
# ── 2. Reading files from HDFS ──────────────────────────────────────

df = spark.read.csv(
    "hdfs://localhost:9000/heartcare/raw/brfss2022_clean.csv",
    header=True,
    inferSchema=True
)

# ── 3. Basic test ───────────────────────────────────────
print("📊 Shape:", df.count(), "rows x", len(df.columns), "cols")
df.printSchema()
df.show(5)

# ── 4. Basic stats ───────────────────────────────────────────
df.describe().show()

# ── 5. Check target distribution ─────────────────────────
print("🎯 Target distribution:")
df.groupBy("target").count().show()

# ── 6. Select features for ML ──────────────────────────────────
feature_cols = [c for c in df.columns if c != "target"]
print(f"✅ Features: {len(feature_cols)} cột")

spark.stop()

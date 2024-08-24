import gc

from pyspark.sql import SparkSession

from src.utils import DATA_FILEPATH, NO_DUP_FILEPATH, SPARK_MEMORY, date_col, id_cols

spark = (SparkSession.builder
         .config("spark.driver.memory", SPARK_MEMORY)
         .appName("SparkSQL").getOrCreate())


df = (spark.read.option("header", "true")
      .option("inferSchema", "true")
      .csv(DATA_FILEPATH))


df1 = df.dropDuplicates([date_col] + id_cols)

df1.write.csv(NO_DUP_FILEPATH, header=True)

gc.collect()

spark.stop()

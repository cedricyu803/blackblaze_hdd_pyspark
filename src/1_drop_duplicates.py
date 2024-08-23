import gc
import os

from pyspark.sql import SparkSession

RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
SPARK_MEMORY = os.getenv('SPARK_MEMORY', "12g")

DATA_DIR = './data'
DATA_FILEPATH = os.path.join(DATA_DIR, 'harddrive.csv')
os.path.exists(DATA_FILEPATH)
NO_DUP_FILEPATH = os.path.join(DATA_DIR, 'harddrive_nodup')


date_col = 'date'
target_label = 'failure'
id_cols = ['serial_number', 'model']


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

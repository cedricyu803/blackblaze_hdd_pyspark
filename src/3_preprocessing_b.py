import gc
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

na_frac_cutoff = 0.4


RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
SPARK_MEMORY = os.getenv('SPARK_MEMORY', "12g")

DATA_DIR = './data'
DATA_FILEPATH = os.path.join(DATA_DIR, 'harddrive_nodup')
CYCLE_ID_FILEPATH = os.path.join(DATA_DIR, 'cycle_id.csv')
CYCLE_ID_FAILURE_FILEPATH = os.path.join(DATA_DIR, 'cycle_id_failure.csv')
os.path.exists(DATA_FILEPATH)
PREPROCESSED_FILEPATH = os.path.join(DATA_DIR, 'harddrive_preprocessed')


def get_less_na_cols(df, na_frac_cutoff: float = 0.4):
    num_samples = df.count()
    na_frac = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c)
                         for c in df.columns]).cache()
    for field in na_frac.schema.fields:
        name = str(field.name)
        na_frac = na_frac.withColumn(name, F.col(name) / num_samples)
    cols_to_keep = [date_col, target_label] + id_cols
    for key, value in na_frac.first().asDict().items():
        if value > na_frac_cutoff:
            continue
        elif key not in cols_to_keep:
            cols_to_keep.append(key)
    return cols_to_keep


date_col = 'date'
target_label = 'failure'
id_cols = ['serial_number', 'model']


spark = (SparkSession.builder
         .config("spark.driver.memory", SPARK_MEMORY)
         .appName("SparkSQL").getOrCreate())


sort_cols = id_cols + [date_col]


# load NO_DUPLICATE dataframe
df = (spark.read.option("header", "true")
      .option("inferSchema", "true")
      .csv(DATA_FILEPATH))

# replace "NA" by None
df = (df
      .select([F.when(F.col(c) == "NA", None)
               .otherwise(F.col(c))
               .alias(c)
               for c in df.columns]))

gc.collect()


# drop columns with fraction of data with NULL larger than cutoff
cols_to_keep = get_less_na_cols(df, na_frac_cutoff=na_frac_cutoff)
cols_to_drop = [col for col in df.columns if col not in cols_to_keep]
df = df.select(cols_to_keep)


# assign cycle_ids: for each id_column value, two cycles are separated by
# more than 1 day
datediff_col = 'datediff'
cycle_id_col = "cycle_id"
datediff_by_id_cols = (df.select(*(id_cols + [date_col]))
                       .withColumn(
    date_col +
    "_lag_1", F.lag(date_col).over(
        Window.partitionBy(id_cols).orderBy(date_col))
)
    .withColumn(
    datediff_col,
        F.when(
            F.col(date_col + "_lag_1").isNotNull(),
            F.datediff(F.col(date_col), F.col(date_col + "_lag_1")),
        ).otherwise(F.lit(None)),
)
    .drop(date_col + "_lag_1")
    .sort(sort_cols)).cache()


df_cycle_id = (datediff_by_id_cols.fillna(1, subset=datediff_col)
               .dropDuplicates(subset=id_cols + [datediff_col])
               .sort(id_cols + [date_col])
               .withColumn(cycle_id_col,
                           F.row_number().over(Window().orderBy(F.lit('A')))))

df_cycle_id = (datediff_by_id_cols
               .fillna(1, subset=datediff_col)
               .join(df_cycle_id,
                     on=id_cols + [date_col, datediff_col],
                     how='left')
               .sort(sort_cols).cache())
df_cycle_id = (df_cycle_id
               .withColumn(cycle_id_col,
                           F.last(cycle_id_col, ignorenulls=True)
                           .over(Window.orderBy(id_cols + [date_col])))
               .drop(datediff_col))

df_cycle_id.toPandas().to_csv(CYCLE_ID_FILEPATH,
                              index=False)

gc.collect()

df_cycle_id_failure = (df_cycle_id.join(
    df.select(*(id_cols + [date_col, target_label])),
    on=id_cols + [date_col],
    how='left')
    .drop(*(id_cols + [date_col])))
df_cycle_id_failure = (df_cycle_id_failure
                       .where(df_cycle_id_failure[target_label] == 1)
                       .drop(target_label)
                       .dropDuplicates().sort(cycle_id_col))
df_cycle_id_failure = df_cycle_id.join(df_cycle_id_failure,
                                       on=cycle_id_col, how='inner')


df_cycle_id_failure.toPandas().to_csv(CYCLE_ID_FAILURE_FILEPATH,
                                      index=False)

gc.collect()

df.write.csv(PREPROCESSED_FILEPATH, header=True, mode='overwrite')

spark.stop()

print(f'Columns dropped: {cols_to_drop}')

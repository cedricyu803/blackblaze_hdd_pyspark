import gc
import json

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.utils import (
    CYCLE_ID_FAILURE_FILEPATH,
    CYCLE_ID_FILEPATH,
    LAGGED_COLS_FILEPATH,
    NO_DUP_FILEPATH,
    PREPROCESSED_COLUMNS_FILEPATH,
    PREPROCESSED_FILEPATH,
    RANDOM_SEED,
    SPARK_MEMORY,
    TRAINING_FILEPATH,
    cycle_id_col,
    date_col,
    get_const_columns,
    get_failed_cycles,
    get_less_na_cols,
    id_cols,
    lag_features,
    make_cycle_id,
    parse_na,
    sort_cols,
    target_label,
)

na_frac_cutoff = 0.4
num_lags = 3


spark = (SparkSession.builder
         .config("spark.driver.memory", SPARK_MEMORY)
         .appName("SparkSQL").getOrCreate())


# load NO_DUPLICATE dataframe
df = (spark.read.option("header", "true")
      .option("inferSchema", "true")
      .csv(NO_DUP_FILEPATH))

# replace "NA" by None
df = parse_na(df)

# drop columns with fraction of data with NULL larger than cutoff
cols_to_drop = get_less_na_cols(df, na_frac_cutoff=na_frac_cutoff)
df = df.drop(*cols_to_drop)

gc.collect()


# assign cycle_ids: for each id_column value, two cycles are separated by
# more than 1 day
df_cycle_id = make_cycle_id(df)
df_cycle_id.cache()
df_cycle_id.toPandas().to_csv(CYCLE_ID_FILEPATH,
                              index=False)

gc.collect()


# get cycles with failures
df_cycle_id_failure = get_failed_cycles(df=df, df_cycle_id=df_cycle_id)
df_cycle_id_failure.cache()
df_cycle_id_failure.toPandas().to_csv(CYCLE_ID_FAILURE_FILEPATH,
                                      index=False)

gc.collect()

df.write.csv(PREPROCESSED_FILEPATH, header=True, mode='overwrite')
with open(PREPROCESSED_COLUMNS_FILEPATH, 'w') as f:
    json.dump(df.columns, f)


# only keep failed cycles for training
df_training = df.join(
    df_cycle_id_failure,
    on=id_cols + [date_col], how='inner'
).sort(id_cols + [date_col, cycle_id_col])

# drop constant columns
const_cols = get_const_columns(df=df_training)
df_training = df_training.drop(*const_cols)

# lag features
df_training = lag_features(df_training, num_lags=num_lags)
df_training = df_training.dropna().sort(sort_cols)

df_training.toPandas().to_csv(TRAINING_FILEPATH, index=False)
with open(LAGGED_COLS_FILEPATH, 'w') as f:
    json.dump(df_training.columns, f)

spark.stop()

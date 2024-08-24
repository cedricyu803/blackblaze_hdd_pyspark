import gc
import json

from pyspark.ml import Pipeline
from pyspark.ml.feature import RobustScaler, VectorAssembler
from pyspark.sql import SparkSession

from src.utils import (
    NUM_FEATURE_COLUMNS_FILEPATH,
    RANDOM_SEED,
    SCALED_COLUMNS_FILEPATH,
    SCALER_PIPELINE_FILEPATH,
    SPARK_MEMORY,
    TEST_FILEPATH,
    TRAIN_FILEPATH,
    TRAINING_FILEPATH,
    VAL_FILEPATH,
    cycle_id_col,
    date_col,
    get_less_na_cols,
    id_cols,
    parse_na,
    scale_features,
    sort_cols,
    target_label,
)

test_frac = 0.1


def data_group_split(df, cycle_id_col=cycle_id_col,
                     random_seed=RANDOM_SEED):
    # train-validation-test split by cycle
    [df_train_cycle, df_val_cycle, df_test_cycle] = (
        df.select(cycle_id_col).dropDuplicates().sort(cycle_id_col)
        .randomSplit([1. - 2 * test_frac, test_frac, test_frac],
                     seed=random_seed))

    df_train = (df.join(df_train_cycle, on=cycle_id_col, how='inner')
                .drop(*(id_cols + [date_col, cycle_id_col])))
    df_val = (df.join(df_val_cycle, on=cycle_id_col, how='inner')
              .drop(*(id_cols + [date_col, cycle_id_col])))
    df_test = (df.join(df_test_cycle, on=cycle_id_col, how='inner')
               .drop(*(id_cols + [date_col, cycle_id_col])))
    return df_train, df_val, df_test


def fit_scaler(df_train, num_feature_cols):
    assembler = VectorAssembler(inputCols=num_feature_cols,
                                outputCol='features_temp')
    scaler = RobustScaler(inputCol='features_temp',
                          outputCol='features_scaled')
    pipeline = Pipeline(stages=[assembler, scaler])
    scalerModel = pipeline.fit(df_train)
    return scalerModel


spark = (SparkSession.builder
         .config("spark.driver.memory", SPARK_MEMORY)
         .appName("SparkSQL").getOrCreate())


# load df_training dataframe
df = (spark.read.option("header", "true")
      .option("inferSchema", "true")
      .csv(TRAINING_FILEPATH))
df = df.sort(id_cols + [date_col])

gc.collect()


# train-validation-test split by cycle
df_train, df_val, df_test = data_group_split(df)

# scale features
num_feature_cols = [col for col in df_train.columns
                    if col not in id_cols +
                    [cycle_id_col, date_col, target_label]]
scalerModel = fit_scaler(df_train=df_train,
                         num_feature_cols=num_feature_cols)
scalerModel.write().overwrite().save(SCALER_PIPELINE_FILEPATH)

df_train_scaled = scale_features(df=df_train,
                                 scalerModel=scalerModel,
                                 num_feature_cols=num_feature_cols)
df_val_scaled = scale_features(df=df_val,
                               scalerModel=scalerModel,
                               num_feature_cols=num_feature_cols)
df_test_scaled = scale_features(df=df_test,
                                scalerModel=scalerModel,
                                num_feature_cols=num_feature_cols)

# replace "NaN" and Infinity by None
df_train_scaled = parse_na(df_train_scaled)
# drop columns with fraction of data with NULL in the train set
cols_to_drop = get_less_na_cols(df_train_scaled, na_frac_cutoff=0.)
df_train_scaled = df_train_scaled.drop(*cols_to_drop)
df_val_scaled = df_val_scaled.drop(*cols_to_drop)
df_test_scaled = df_test_scaled.drop(*cols_to_drop)

df_train_scaled.toPandas().to_csv(TRAIN_FILEPATH, index=False)
df_val_scaled.toPandas().to_csv(VAL_FILEPATH, index=False)
df_test_scaled.toPandas().to_csv(TEST_FILEPATH, index=False)

with open(NUM_FEATURE_COLUMNS_FILEPATH, 'w') as f:
    json.dump(num_feature_cols, f)
with open(SCALED_COLUMNS_FILEPATH, 'w') as f:
    json.dump(df_train_scaled.columns, f)


spark.stop()

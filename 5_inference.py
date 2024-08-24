import gc
import json

from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import SparkSession

from src.utils import (
    FEATURE_COLUMNS_FILEPATH,
    LAGGED_COLS_FILEPATH,
    ML_MODEL_FILEPATH,
    NUM_FEATURE_COLUMNS_FILEPATH,
    PREDICTIONS_FILEPATH,
    PREPROCESSED_COLUMNS_FILEPATH,
    PREPROCESSED_FILEPATH,
    SCALED_COLUMNS_FILEPATH,
    SCALER_PIPELINE_FILEPATH,
    SPARK_MEMORY,
    cycle_id_col,
    date_col,
    id_cols,
    lag_features,
    make_cycle_id,
    model_transform_features,
    scale_features,
    sort_cols,
    target_label,
)

# same as training pipeline
num_lags = 3

inference_filepath = PREPROCESSED_FILEPATH
nrows = 100

spark = (SparkSession.builder
         .config("spark.driver.memory", SPARK_MEMORY)
         .appName("SparkSQL").getOrCreate())

# load metadata from training pipeline
with open(PREPROCESSED_COLUMNS_FILEPATH, 'r') as f:
    preprocessed_cols = json.load(f)
with open(LAGGED_COLS_FILEPATH, 'r') as f:
    lagged_cols = json.load(f)
with open(NUM_FEATURE_COLUMNS_FILEPATH, 'r') as f:
    num_feature_cols = json.load(f)
with open(SCALED_COLUMNS_FILEPATH, 'r') as f:
    scaled_cols = json.load(f)
with open(FEATURE_COLUMNS_FILEPATH, 'r') as f:
    feature_cols = json.load(f)
for list_ in [preprocessed_cols, lagged_cols,
              num_feature_cols, scaled_cols, feature_cols]:
    if target_label in list_:
        list_.remove(target_label)


# load training data dataframe
df_inf = (spark.read.option("header", "true")
          .option("inferSchema", "true")
          .csv(inference_filepath))
df_inf = df_inf.sort(sort_cols).limit(nrows)

# select columns following training pipeline
df_inf = df_inf.select(*preprocessed_cols)

gc.collect()

# make cycle_id
df_cycle_id = make_cycle_id(df_inf)
df_inf = df_inf.join(
    df_cycle_id,
    on=id_cols + [date_col], how='inner'
).sort(id_cols + [date_col, cycle_id_col])

# lag features
df_inf = lag_features(df_inf, num_lags=num_lags)
df_inf = df_inf.dropna().sort(sort_cols)
df_inf = df_inf.select(*lagged_cols)

# scale features
scalerModel = PipelineModel.load(SCALER_PIPELINE_FILEPATH)
df_inf_scaled = scale_features(df=df_inf,
                               scalerModel=scalerModel,
                               num_feature_cols=num_feature_cols,
                               inference=True)
df_inf_scaled = df_inf_scaled.select(*(id_cols + [date_col] + scaled_cols))

# model inference
model = GBTClassificationModel.load(ML_MODEL_FILEPATH)
df_inf_scaled, assembler_inf = model_transform_features(
    df=df_inf_scaled,
    feature_cols=feature_cols,
    inference=True)
pred_inf = model.transform(df_inf_scaled)

pred_inf.toPandas().to_csv(PREDICTIONS_FILEPATH, index=False)

spark.stop()

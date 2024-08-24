import os

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.sql.window import Window

RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
SPARK_MEMORY = os.getenv('SPARK_MEMORY', "12g")


DATA_DIR = './data'
DATA_FILEPATH = os.path.join(DATA_DIR, 'harddrive.csv')
NO_DUP_FILEPATH = os.path.join(DATA_DIR, 'harddrive_nodup')
PREPROCESSED_FILEPATH = os.path.join(DATA_DIR, 'harddrive_preprocessed')
CYCLE_ID_FILEPATH = os.path.join(DATA_DIR, 'cycle_id.csv')
CYCLE_ID_FAILURE_FILEPATH = os.path.join(DATA_DIR, 'cycle_id_failure.csv')
TRAINING_FILEPATH = os.path.join(DATA_DIR, 'harddrive_training.csv')
TRAIN_FILEPATH = os.path.join(DATA_DIR, 'df_train.csv')
VAL_FILEPATH = os.path.join(DATA_DIR, 'df_val.csv')
TEST_FILEPATH = os.path.join(DATA_DIR, 'df_test.csv')

MODELS_DIR = './models'
os.makedirs(MODELS_DIR, exist_ok=True)
SCALER_PIPELINE_FILEPATH = os.path.join(MODELS_DIR, 'scaler')
ML_MODEL_FILEPATH = os.path.join(MODELS_DIR, 'model')
ML_EVAL_SCORES_FILEPATH = os.path.join(MODELS_DIR, 'model_scores.json')
PREPROCESSED_COLUMNS_FILEPATH = os.path.join(MODELS_DIR,
                                             'preprocessed_columns.json')
LAGGED_COLS_FILEPATH = os.path.join(MODELS_DIR, 'lagged_columns.json')
NUM_FEATURE_COLUMNS_FILEPATH = os.path.join(MODELS_DIR,
                                            'num_feature_columns.json')
SCALED_COLUMNS_FILEPATH = os.path.join(MODELS_DIR, 'scaled_columns.json')
FEATURE_COLUMNS_FILEPATH = os.path.join(MODELS_DIR, 'feature_columns.json')

INFERENCE_DIR = './predictions'
os.makedirs(INFERENCE_DIR, exist_ok=True)
PREDICTIONS_FILEPATH = os.path.join(INFERENCE_DIR, 'predictions.csv')

date_col = 'date'
target_label = 'failure'
id_cols = ['serial_number', 'model']
cycle_id_col = 'cycle_id'

sort_cols = id_cols + [date_col]


def parse_na(df):
    na_list = ["NA", 'na', 'NaN', 'nan', 'Infinity']
    df = (df
          .select([F.when(F.col(c).isin(na_list), None).otherwise(F.col(c))
                   .alias(c) for c in df.columns]))
    return df


def get_less_na_cols(df, na_frac_cutoff: float = 0.4):
    num_samples = df.count()
    na_frac = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c)
                         for c in df.columns]).cache()
    for field in na_frac.schema.fields:
        name = str(field.name)
        na_frac = na_frac.withColumn(name, F.col(name) / num_samples)
    cols_to_drop = []
    for key, value in na_frac.first().asDict().items():
        if (value > na_frac_cutoff and
                key not in [date_col, target_label] + id_cols):
            cols_to_drop.append(key)
    return cols_to_drop


def make_cycle_id(df, cycle_id_col=cycle_id_col):
    # assign cycle_ids: for each id_column value, two cycles are separated by
    # more than 1 day
    datediff_col = 'datediff'
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
        .sort(sort_cols))

    df_cycle_id = (datediff_by_id_cols.fillna(1, subset=datediff_col)
                   .dropDuplicates(subset=id_cols + [datediff_col])
                   .sort(id_cols + [date_col])
                   .withColumn(
                       cycle_id_col,
                       F.row_number().over(Window().orderBy(F.lit('A')))))

    df_cycle_id = (datediff_by_id_cols
                   .fillna(1, subset=datediff_col)
                   .join(df_cycle_id,
                         on=id_cols + [date_col, datediff_col],
                         how='left')
                   .sort(sort_cols))
    df_cycle_id = (df_cycle_id
                   .withColumn(cycle_id_col,
                               F.last(cycle_id_col, ignorenulls=True)
                               .over(Window.orderBy(id_cols + [date_col])))
                   .drop(datediff_col))
    return df_cycle_id


def get_failed_cycles(df, df_cycle_id):
    # get cycles with failures
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

    return df_cycle_id_failure


def get_const_columns(df):
    # drop constant columns
    col_value_counts = df.agg(*(F.countDistinct(F.col(c)).alias(c)
                                for c in df.columns))
    df_col_value_counts = col_value_counts.toPandas()
    const_cols = list(
        df_col_value_counts[df_col_value_counts < 2].T.dropna().index)
    const_cols = [col for col in const_cols
                  if col not in id_cols +
                  [date_col, target_label, cycle_id_col]]
    return const_cols


def lag_features(df, num_lags: int = 3,
                 no_lag_cols=id_cols + [date_col, target_label, cycle_id_col]
                 ):
    # lag features
    no_lag_cols = id_cols + [date_col, target_label, cycle_id_col]
    window = Window.partitionBy(cycle_id_col).orderBy(date_col)
    for col in df.columns:
        if col in no_lag_cols:
            continue
        for n in range(num_lags):
            df = df.withColumn(
                col + f"_lag_{n+1}", F.lag(col, offset=n + 1).over(window))

    return df


def scale_features(df, scalerModel, num_feature_cols,
                   inference: bool = False):
    if not inference:
        df_scaled = (scalerModel.transform(df)
                     .select(*[target_label, 'features_scaled'])
                     .withColumn("features_scaled_array",
                                 vector_to_array("features_scaled"))
                     .select([target_label] + [
                         F.col("features_scaled_array")[i].alias(col)
                         for i, col in enumerate(num_feature_cols)]))
    else:
        df_scaled = (scalerModel.transform(df)
                     .withColumn("features_scaled_array",
                                 vector_to_array("features_scaled"))
                     .select(id_cols + [date_col] + [
                         F.col("features_scaled_array")[i].alias(col)
                         for i, col in enumerate(num_feature_cols)]))

    return df_scaled


def model_transform_features(df, feature_cols,
                             inference: bool = False):
    assembler = VectorAssembler(inputCols=feature_cols,
                                outputCol='features')
    df = assembler.transform(df)
    if not inference:
        df = df.select([target_label, 'features'])
    else:
        df = df.select(id_cols + [date_col] + ['features'])
    return df, assembler

import gc
import json

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

from src.utils import (
    FEATURE_COLUMNS_FILEPATH,
    ML_EVAL_SCORES_FILEPATH,
    ML_MODEL_FILEPATH,
    RANDOM_SEED,
    SPARK_MEMORY,
    TEST_FILEPATH,
    TRAIN_FILEPATH,
    VAL_FILEPATH,
    cycle_id_col,
    date_col,
    id_cols,
    model_transform_features,
    target_label,
)

metricName = 'f1'


spark = (SparkSession.builder
         .config("spark.driver.memory", SPARK_MEMORY)
         .appName("SparkSQL").getOrCreate())


# load training data dataframe
df_train = (spark.read.option("header", "true")
            .option("inferSchema", "true")
            .csv(TRAIN_FILEPATH))
df_val = (spark.read.option("header", "true")
          .option("inferSchema", "true")
          .csv(VAL_FILEPATH))
df_test = (spark.read.option("header", "true")
           .option("inferSchema", "true")
           .csv(TEST_FILEPATH))


gc.collect()

# transform features into vector
feature_cols = [col for col in df_train.columns
                if col not in id_cols + [date_col, cycle_id_col, target_label]]


df_train, assembler_train = model_transform_features(
    df=df_train,
    feature_cols=feature_cols)
df_val, assembler_val = model_transform_features(
    df=df_val,
    feature_cols=feature_cols)
df_test, assembler_test = model_transform_features(
    df=df_test,
    feature_cols=feature_cols)


# fit ML model
model = GBTClassifier(labelCol=target_label, featuresCol='features',
                      seed=RANDOM_SEED)
model = model.fit(df_train)

# predict
pred_train = model.transform(df_train)
pred_val = model.transform(df_val)
pred_test = model.transform(df_test)

# save ML model
model.write().overwrite().save(ML_MODEL_FILEPATH)

# evaluate
scores = {'metricName': metricName}
evaluator = MulticlassClassificationEvaluator(
    labelCol=target_label,
    predictionCol='prediction', metricName=metricName)
scores['train'] = evaluator.evaluate(pred_train)
scores['val'] = evaluator.evaluate(pred_val)
scores['test'] = evaluator.evaluate(pred_test)

with open(ML_EVAL_SCORES_FILEPATH, 'w') as f:
    json.dump(scores, f)
with open(FEATURE_COLUMNS_FILEPATH, 'w') as f:
    json.dump(feature_cols, f)

spark.stop()

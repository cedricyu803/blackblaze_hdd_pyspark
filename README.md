Updated August 2024

# Machine learning with PySpark: Blackblaze hard drive failure prediction

We provide an end-to-end PySpark example with the 'Blackblaze hard drive failure prediction' example, in which we predict from the S.M.A.R.T. telemetry from a hard drive whether it has failed. The goal of this exercise is to showcase the use of **PySpark** to perform data science tasks on a real dataset: from EDA, date cleaning and preprocessing, to feature engineering and scaling, ML model training, to inference.

## In this repo
This repo contains:
- `./data`: the raw dataset is to be put here (see below). The preprocessed and engineered datasets from the training pipeline are also saved here
- `./models`: the fitted scaler and ML model are saved here. We also output the preprocessed and engineered metada (column names) here in `json`, to be used by the inference pipeline. The evaluation scores of the model are also saved here in `json`
- `./notebooks`: Jupyter notebook for performing EDA and developing the ML pipelines
- `./predictions`: predictions from the inference pipeline is saved here
- `./src`:
  - `utils.py` : utility functions used by our PySpark scripts
    - We also define global variables (e.g. output filepaths) here (should be put say in a `yml` config file but we keep it simple)
- PySpark scripts in the root folder:
   1. `1_drop_duplicates.py`  (training)
   2. `2_preprocessing.py`  (training)
   3. `3_feature_engineering.py`  (training)
   4. `4_model_training.py`  (training)
   5. `5_inference.py`  (inference)
- `requirements.txt`: Python (3.10) requirements for running the PySpark tasks


## The dataset
The Blackblaze hard drive dataset hosted on [Kaggle](https://www.kaggle.com/datasets/backblaze/hard-drive-test-data/data) is used. This dataset contains from the S.M.A.R.T. telemetry--- raw and normalised--- from hard drives the first two quarters in 2016. The 'failure' column indicates failure: it is 0 when the hard drive is running, and is 1 when it fails upon which it is taken out of service. The dataset comes with the id columns `serial_number` and `model` of each hard drive, and a `date` column.

## Instructions for running the ML pipeline on PySpark

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/backblaze/hard-drive-test-data/data) and put it in `./data`
2. Run `pip install -r requirements.txt` on a Python 3.10 environment
3. Configure the PySpark memory limit with the environment variable `SPARK_MEMORY`; default is `12g`
4. Run the training pipeline by running following PySpark tasks with the command: `spark-submit --executor-memory 12g <script name>` in the following order:
   1. `1_drop_duplicates.py`
   2. `2_preprocessing.py`
   3. `3_feature_engineering.py`
   4. `4_model_training.py`
   - See the next section for description 
5. Test run the inference pipeline by running `spark-submit --executor-memory 12g 5_inference.py`. This by default runs inference on a test dataset consisting of the first 100 rows of the preprocessed dataset (from `2_preprocessing.py`)

## The machine learning problem and solution

Since our main goal is to demonstrate the use of PySpark on machine learning, we only consider a simple machine learning problem and develop a minimal working solution.

### Our (simplistic) machine learning problem
From the labelled dataset, we consider a simple *binary classification* problem classifying the state of a hard drive into normal (0) or failure (1).

Admittedly, this problem is not very actionable: ideally we would like to be able to predict a failure in advance, so that the human can act on it e.g. by taking the hard drive out of service to prevent loss. A more actionable alternative ML problem can be the *remaining useful life (RUL)* prediction.

### Our machine learning solution: training pipeline
Our training pipeline consists of 4 steps:
   1. `1_drop_duplicates.py`
      1.  Ingests raw labelled dataset
      2.  Drops duplicated rows
      3.  Writes to file
      - We do this as a standalone step since `dropDuplicates` is computationally expensive
   2. `2_preprocessing.py`
      1. Defines parameters (should have put it in a `yml` config file but let's keep it simple)
      ```
      na_frac_cutoff = 0.4
      num_lags = 3
      ```
      2. Loads the labelled dataset with no duplicated rows
      3. Parses `NULL` values: replace `NA` by `None`
      4. Drops columns with fraction of `NULL` values larger than `na_frac_cutoff`
      5. Assigns `cycle_id` to each id (`serial_number` and `model`) value, two cycles are separated by more than 1 day
      6. Keeps cycles in the dataset that eventually failed--- they are the only ones used for training
      7. Drop constant columns
      8. Generate lag features according to `num_lags`
      9. Writes training data and column names to files
   3. `3_feature_engineering.py`
      1. Defines parameters (should have put it in a `yml` config file but let's keep it simple)
      ```
      test_frac = 0.1  # fraction of validation and test sets in train-val-test split
      ```
      2. Loads training data from Step 2
      3. Train-validation-test splits by cycle, according to `test_frac`
      4. Scales features with `RobustScaler` and saves fitted scaler to file
      5. Parses  `NULL` values (`NaN` and `Infinity`) and drop columns with `NULL` values from the train set-- they appear as many of the columns have very small values
      6. Writes scaled training datasets and column names to files
   4. `4_model_training.py`
      1. Defines parameters (should have put it in a `yml` config file but let's keep it simple)
      ```
      metricName = 'f1'
      ```
      2. Loads scaled training datasets from Step 3
      3. Fits a `GBTClassifier` with train set and saves fitted model to file
      4. Evaluates the fitted model on train, validation, and test sets
      5. Writes evaluation scores to `json` file

### Our machine learning solution: inference pipeline
The inference pipeline corresponding to the training pipeline is `5_inference.py`.
1. Defines parameters (should have put it in a `yml` config file but let's keep it simple)
```
num_lags = 3  # same as training pipeline

# here we use a test inference dataset consisting of
# the first 100 rows of the preprocessed dataset (from `2_preprocessing.py`)
inference_filepath = PREPROCESSED_FILEPATH
nrows = 100
```
2. Loads metadata (column names in various steps) from training pipeline
3. Loads inference dataset
4. Selects columns according to the preprocessed training data (from `2_preprocessing.py`)
5. Makes cycle_id
6. Generates lag features
7. Scales features with the fitted scaler from the training pipeline
8. Makes predictions with the fitted model from the training pipeline
9. Writes predictions to `csv` file


Acknowledgement: Some of the steps are inspired by [this Medium series](https://medium.com/geekculture/a-complete-solution-to-the-backbaze-com-kaggle-problem-cf1fab1af529).


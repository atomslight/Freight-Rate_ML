# Freight Rate Machine Learning End to End Project

## **Result** dir Contain all result at one place

## **Report** dir Contains how to run and setup the project doc and Report on whole project findings and whole report from start to end

## **Checkpoint Trained State Models** contains all the trained weights and model files
If you dont have uv then install it

pip install uv

Note: If you're on Windows

If uv isn't recognized after installation, run it via Python:

py -m uv --version

or ensure your Python Scripts directory is on your PATH.



Install all the dependancies

Create a directory of your project
and go in that directory
then
uv init

Install all required dependancies in system

uv add jupyterlab pandas numpy matplotlib scikit-learn xgboost lightgbm joblib




# Start the Environment

Launch JupyterLab using:

bash
uv run jupyter lab


---

# Stage 1: Train-Test Model Training

Open the `main train test.ipynb` notebook in JupyterLab.

After completing the final step, the following trained checkpoint files will be generated:

- `imputer.pkl`
- `onehot_encoder.pkl`
- `best_xgboost_model.pkl`

> Note: The `best_xgboost_model.pkl` model has already been trained on the train-test dataset and can be reused for prediction.

Finally, load `best_xgboost_model.pkl` and run the prediction step. This will generate:

- `xgboost_predictions_LoadedMODEL.csv`

Also, run the metrics evaluation code to generate:

- `metrics.csv`

These are the final test results obtained from the train-test dataset.

---

# Stage 2: Validation Data Prediction

Open the `Validation unknown.ipynb` notebook in JupyterLab.

At the end of the notebook, load the trained model:

- `best_xgboost_model.pkl`

Run prediction on the validation dataset. This will generate:

- `validation_predictions.csv`

---

# Stage 3: December Chart Prediction

> Note: To make this stage work, I had to modify the `score.py` script because it expected exactly **12,000 rows**. This was no longer possible since some rows were removed during data cleaning due to **negative values in the `weight` column**, which were treated as invalid outliers.

I also created the required missing columns and ensured that `December_input.csv` had the same structure, formatting, and feature columns as both `train-test.csv` and `validation.csv`.

Finally, load the following preprocessing artifacts:

- `imputer.pkl`
- `onehot_encoder.pkl`

Use them to:

1. Impute missing values.
2. Apply the same preprocessing transformations used during training.
3. Generate the final predictions for the December input dataset.


## Stage 1: Train-Test Pipeline

**Notebook:** `main train test.ipynb`

### Data Exploration (EDA)

-   Loaded the `train-test-exp.csv` dataset.
-   Identified patterns and potential issues in each attribute using
    visualizations.
-   Converted the `date` column into machine-readable features:
    -   Month
    -   Day
    -   Weekday
    -   Week of Year
    -   Day of Year
-   Dropped the original `date` column after feature extraction.
-   Performed correlation analysis and found that **distance** had the
    strongest relationship with the target variable.
-   Created the `distance_cat` feature to serve as the basis for
    `StratifiedShuffleSplit`.
-   Split the dataset into **80% training** and **20% testing** sets.

### Data Cleaning & Preparation

The test set was kept aside to avoid data leakage.

The following preprocessing steps were performed on the training data:

-   Removed rows containing invalid (negative) weight values, as they
    were considered outliers.
-   Dropped categorical attributes before numerical imputation.
-   Preserved copies of `equipment` and `load_id` for later use.
-   Applied **SimpleImputer** to the numerical training data and used
    the fitted imputer to transform both the training and test datasets.
-   Applied **One-Hot Encoding** to the previously saved `equipment`
    column.

> **Note:** The `pickup` and `delivery` city names were intentionally
> removed because they frequently change. Their corresponding latitude
> and longitude values already capture the geographical information,
> making them more suitable features.

The fitted preprocessing artifacts were saved for reuse:

-   `imputer.pkl`
-   `onehot_encoder.pkl`

At this stage, all EDA, data cleaning, and preprocessing steps were
completed.

------------------------------------------------------------------------

## Model Selection

Since this is a **non-linear supervised regression problem**, the
following models were evaluated:

  Model                CV RMSE   CV MAE   CV MAPE    CV R² Assessment
  ------------------ --------- -------- --------- -------- -------------
  🥇 XGBoost            596.65   113.18     5.44%   0.8384 Excellent
  🥈 LightGBM           598.55   121.18     6.43%   0.8374 Good
  🥉 Random Forest      600.47   122.42     6.36%   0.8363 Overfitting

------------------------------------------------------------------------

## Final Model

**Model:** XGBoost Regressor (optimized using `RandomizedSearchCV`)

### Best Hyperparameters

``` python
{
    "subsample": 1.0,
    "reg_lambda": 5,
    "reg_alpha": 0,
    "n_estimators": 300,
    "min_child_weight": 5,
    "max_depth": 3,
    "learning_rate": 0.05,
    "gamma": 0,
    "colsample_bytree": 1.0
}
```

### Final Test Performance

  Metric          Value
  -------- ------------
  RMSE       **546.62**
  MAE        **102.21**
  MAPE        **5.16%**
  R²         **0.8616**

### Interpretation

-   **RMSE = 546.62** --- Predictions deviate by approximately **546
    units** on average, with larger errors penalized more heavily.
-   **MAE = 102.21** --- The average prediction error is approximately
    **102 units**.
-   **MAPE = 5.16%** --- Predictions differ from the actual values by
    only **5.16%** on average, which is considered excellent.
-   **R² = 0.8616** --- The model explains **86.16%** of the variance in
    freight rates.

### Overall Assessment

-   ✅ Data preprocessing
-   ✅ Missing value imputation
-   ✅ One-Hot Encoding
-   ✅ Train/Test Split
-   ✅ Cross-validation
-   ✅ Multiple baseline models
-   ✅ Hyperparameter tuning using `RandomizedSearchCV`
-   ✅ Model comparison
-   ✅ Final evaluation on an untouched test set

The trained model and prediction outputs were saved in the **Results**
folder, including:

-   `xgboost_predictions_LoadedMODEL.csv`
-   Test metrics

------------------------------------------------------------------------

## Stage 2: Validation Pipeline

**Notebook:** `Validation unknown.ipynb`

The validation dataset follows the same preprocessing pipeline:

-   Remove rows containing invalid weight values.
-   Transform the `date` column into:
    -   Month
    -   Day
    -   Weekday
    -   Week of Year
    -   Day of Year
-   Load the previously saved preprocessing artifacts:
    -   `imputer.pkl`
    -   `onehot_encoder.pkl`
-   Apply the same preprocessing transformations.
-   Load the trained model:

`best_xgboost_model.pkl`

-   Generate predictions for the validation dataset.
-   Export the final `validation_predictions.csv`.

------------------------------------------------------------------------

## Stage 3: December Dataset

**Notebook:** `december_Model_code.ipynb`

The December dataset follows the same preprocessing pipeline used for
the validation dataset.

Steps:

-   Prepare the dataset.
-   Apply the saved preprocessing artifacts.
-   Load `best_xgboost_model.pkl`.
-   Generate predictions.
-   Run the modified `score.py` script to generate the final evaluation
    graph.

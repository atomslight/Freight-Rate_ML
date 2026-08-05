# Machine Learning Assessment Submission

This repository contains the complete solution for the Machine Learning assessment, including data exploration, preprocessing, model training, evaluation, and prediction.

---

# Repository Structure

| Folder / File | Description |
|---------------|-------------|
| `main train test.ipynb` | Complete training pipeline including EDA, preprocessing, feature engineering, model training, evaluation, and hyperparameter tuning. |
| `Validation unknown.ipynb` | Generates predictions for the validation dataset using the trained model. |
| `december_Model_code.ipynb` | Generates predictions for the December dataset and prepares it for scoring. |
| `Checkpoint Trained States/` | Contains the trained preprocessing artifacts and final model (`best_xgboost_model.pkl`, `imputer.pkl`, `onehot_encoder.pkl`). |
| `Results/` | Contains all generated outputs including train-test results, validation predictions, December predictions, evaluation metrics, and the final score graph (`candidate_december.png`). |
| `Report/` | Contains the complete project report, environment setup instructions, and execution guide. |
| `pyproject.toml` & `uv.lock` | Project dependencies and reproducible environment configuration. |

---

# Assessment Coverage

- ✅ Exploratory Data Analysis (EDA)
- ✅ Data Cleaning
- ✅ Feature Engineering
- ✅ Train/Test Split
- ✅ Missing Value Imputation
- ✅ One-Hot Encoding
- ✅ Model Training
- ✅ Cross Validation
- ✅ Hyperparameter Tuning
- ✅ Model Comparison
- ✅ Final Model Evaluation
- ✅ Validation Prediction
- ✅ December Prediction
- ✅ Model Persistence

---

# Project Workflow

```text
train-test-exp.csv
        │
        ▼
main train test.ipynb
        │
        ├── EDA
        ├── Data Cleaning
        ├── Feature Engineering
        ├── Train/Test Split
        ├── Model Training
        ├── Hyperparameter Tuning
        ├── Model Evaluation
        └── Save Model
                │
                ▼
Checkpoint Trained States/
        ├── best_xgboost_model.pkl
        ├── imputer.pkl
        └── onehot_encoder.pkl
                │
        ┌───────┴────────┐
        ▼                ▼
Validation       December Prediction
unknown.ipynb    december_Model_code.ipynb
        │                │
        ▼                ▼
validation_      December Predictions
predictions.csv         │
                         ▼
                     score.py
                         │
                         ▼
               candidate_december.png
```

---

# Outputs

## Checkpoint Trained States

- `best_xgboost_model.pkl`
- `imputer.pkl`
- `onehot_encoder.pkl`

## Validation

- `validation_predictions.csv`

## Train-Test

- Test predictions
- Test metrics

## December

- December predictions
- `candidate_december.png`

All generated outputs are available inside the **`Results/`** directory.

---

# Documentation

The **`Report/`** directory contains:

- Complete project report
- Environment setup instructions
- Execution guide
- Project methodology and explanation

---

# Environment

Project dependencies are managed using **uv**.

Dependency files:

- `pyproject.toml`
- `uv.lock`

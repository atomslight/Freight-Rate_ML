# Machine Learning Assessment Submission

This repository contains the solution for the Machine Learning assessment.
The code covers Data Exploration, Cleaning, Feature Engineering, Modeling, Evaluation, and final Predictions.

## Assessment Requirement Mapping

| Assessment Requirement | Status | Repository Location | Explanation |
|---|---|---|---|
| Training using `data/train_test.csv` (or `train-test-exp.csv`) | Done | [`main train test.ipynb`](main%20train%20test.ipynb) | The primary notebook where the model is trained on the train-test dataset. |
| Data exploration (EDA) | Done | [`main train test.ipynb`](main%20train%20test.ipynb) | Exploratory data analysis, including understanding data distributions and handling missing values. |
| Data cleaning | Done | [`main train test.ipynb`](main%20train%20test.ipynb) | Removal of invalid rows (e.g. negative weight) and filling missing values using SimpleImputer. |
| Feature engineering | Done | [`main train test.ipynb`](main%20train%20test.ipynb) | Includes date transformations into numerical columns and One-Hot Encoding for categorical features. |
| Train/validation split strategy | Done | [`main train test.ipynb`](main%20train%20test.ipynb) | Implementation of StratifiedShuffleSplit based on distance categories to maintain balanced data splits. |
| Validation methodology | Done | [`main train test.ipynb`](main%20train%20test.ipynb) | Utilizes cross-validation and RandomizedSearchCV for evaluating models securely without overfitting. |
| Model selection | Done | [`main train test.ipynb`](main%20train%20test.ipynb) | Compared Random Forest, XGBoost, and LightGBM models. XGBoost was chosen as the final best model. |
| Hyperparameter tuning | Done | [`main train test.ipynb`](main%20train%20test.ipynb) | Applied RandomizedSearchCV on the chosen models to find optimal hyperparameters. |
| Prediction on `validation.csv` | Done | [`validation unknown.ipynb`](validation%20unknown.ipynb) | Loads the saved XGBoost model and executes inference on the unseen validation dataset. |
| Generation of `validation_predictions.csv` | Done | [`validation_predictions.csv`](validation_predictions.csv) | The generated CSV mapping `load_id` to `predicted_rate` for the validation dataset. |
| Prediction on `december_chart_inputs.csv` | Done | [`december_Model_code.ipynb`](december_Model_code.ipynb) | Applies preprocessing and model prediction to the modified December dataset. |
| Running `score.py` | Done | [`score.py`](score.py) | Python script that takes predictions as inputs to compute metrics and plot charts. |
| Generation of `candidate_december.png` | Done | [`Results/scorer_results/candidate_december.png`](Results/scorer_results/candidate_december.png) | The final plotted visualization displaying December predictions. |
| Model persistence | Done | [`best_xgboost_model.pkl`](best_xgboost_model.pkl) | The serialized XGBoost model, saved alongside its preprocessors (`imputer.pkl`, `onehot_encoder.pkl`). |
| Report generation | Done | [`Report/Freight Rate Prediction Assessment Report.docx`](Report/Freight%20Rate%20Prediction%20Assessment%20Report.docx) | Comprehensive analytical report detailing decisions, experiments, and model performance. |

---

## Repository Navigation

**Where is the EDA?**
[`main train test.ipynb`](main%20train%20test.ipynb)
*Purpose:* Primary notebook used to train the model, containing in-depth data distribution visualizations and EDA logic.

**Where is preprocessing?**
[`main train test.ipynb`](main%20train%20test.ipynb)
*Purpose:* Handles the data pipeline logic, mapping raw data through the `imputer.pkl` and `onehot_encoder.pkl` artifacts.

**Where are feature engineering steps?**
[`main train test.ipynb`](main%20train%20test.ipynb)
*Purpose:* Contains code to convert raw date/categorical attributes into meaningful machine-readable variables.

**Where is train/test split?**
[`main train test.ipynb`](main%20train%20test.ipynb)
*Purpose:* Executes an 80-20 Stratified Shuffle Split to create the training and test arrays.

**Where is model training?**
[`main train test.ipynb`](main%20train%20test.ipynb)
*Purpose:* Trains base level and complex models against the preprocessed training datasets.

**Where is cross validation?**
[`main train test.ipynb`](main%20train%20test.ipynb)
*Purpose:* Uses `RandomizedSearchCV` cross-validation loops to ensure models aren't overfitting.

**Where is hyperparameter tuning?**
[`main train test.ipynb`](main%20train%20test.ipynb)
*Purpose:* Explores the hyperparameter space for XGBoost, Random Forest, and LightGBM models.

**Where is model comparison?**
[`main train test.ipynb`](main%20train%20test.ipynb)
*Purpose:* Showcases differences in precision between tested models and their ensemble combinations.

**Where is final evaluation?**
[`main train test.ipynb`](main%20train%20test.ipynb)
*Purpose:* Calculates and prints final metrics representing model performance.

**Where is inference?**
[`validation unknown.ipynb`](validation%20unknown.ipynb) and [`december_Model_code.ipynb`](december_Model_code.ipynb)
*Purpose:* Notebooks dedicated specifically to executing inference tasks on unseen validation and december datasets.

**Where are saved models?**
[`best_xgboost_model.pkl`](best_xgboost_model.pkl), [`imputer.pkl`](imputer.pkl), and [`onehot_encoder.pkl`](onehot_encoder.pkl)
*Purpose:* The pickled joblib files capturing the entire trained pipeline required to run predictions.

**Where are predictions?**
[`Results/`](Results/) directory and [`validation_predictions.csv`](validation_predictions.csv)
*Purpose:* The folder where generated predictions on test sets, validation sets, and December inputs are exported.

**Where is the report?**
[`Report/Freight Rate Prediction Assessment Report.docx`](Report/Freight%20Rate%20Prediction%20Assessment%20Report.docx)
*Purpose:* A detailed MS Word document explaining the entire end-to-end data science process.

**Where is the generated December chart?**
[`Results/scorer_results/candidate_december.png`](Results/scorer_results/candidate_december.png)
*Purpose:* A visual line chart created by the `score.py` validator script.

**Where is the validation_predictions.csv?**
[`validation_predictions.csv`](validation_predictions.csv)
*Purpose:* The required target inference CSV mapping ID entries to final predicted prices.

**Where is the final notebook?**
[`main train test.ipynb`](main%20train%20test.ipynb)
*Purpose:* Acts as the comprehensive unified workspace covering the majority of the training lifecycle.

**Where are dependencies?**
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock)
*Purpose:* Define exact package lists and hashes to maintain a completely reproducible execution environment.

**Where is the environment setup?**
[`Report/Running the project.txt`](Report/Running%20the%20project.txt)
*Purpose:* A text guide providing standard setup instructions for bootstrapping the `uv` virtual environment.

**Where is the execution command?**
[`score.py`](score.py) and [`main.py`](main.py)
*Purpose:* Run `uv run python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs_mod.csv` to execute final evaluation validation.

---

## Repository Flow

Dataset (`train-test-exp.csv`)
↓
EDA Notebook (`main train test.ipynb`)
↓
Data Cleaning (`main train test.ipynb`)
↓
Feature Engineering (`main train test.ipynb`)
↓
Train/Test Split (`main train test.ipynb`)
↓
Missing Value Handling (`main train test.ipynb`)
↓
Encoding (`main train test.ipynb`)
↓
Model Training (`main train test.ipynb`)
↓
Hyperparameter Search (`main train test.ipynb`)
↓
Model Comparison (`main train test.ipynb`)
↓
Final Model Selection (`main train test.ipynb`)
↓
Validation (`validation unknown.ipynb`)
↓
Prediction on `validation.csv` (`validation unknown.ipynb`)
↓
`validation_predictions.csv` (`validation_predictions.csv`)
↓
Prediction on `december_chart_inputs.csv` (`december_Model_code.ipynb`)
↓
`score.py` (`score.py`)
↓
`candidate_december.png` (`Results/scorer_results/candidate_december.png`)

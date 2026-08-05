Dataset loaded train-test-exp
Pattern and problem identification using graphs for each attr


Transforming data into machine understandable form and breaking date into month,day,weekday,week of year,day of year format

Drop original date attr.

Understanding highest correlation and got to know that distance is strongest factor

Creation of distance cat attr that will form as a base index for stratifiedshufflesplit

Data splitted into train test form 80-20% 

now keep test set aside and do data cleaning and data preparation using EDA

Invalid weight rows removed as they were outliers

dropped all categorical attr as we are heading in Data Imputation and filling of missing data

Also copy these dropped equipment and load_id (identifier) category as this will be later on used!

so we use Simple imputer on numerical data train set fitted and then transformed onto train and test data

now using One Hot Encoding on that earlier copied equipment attr

Note: We intentionally dropped pickup and delivery location name as they are continue changing but their replacement lat and long handle that pattern so its fine and good practice in feature engineering

also save the checkpoint .pkl fitted (trained) imputer and one hot encoder (Best practice for code reusability)

finally all the data preparation and EDA steps are done now we headed towards model selection

Since we understood that this was a non linear temporal supervised regression task we selected 3 models for testing
🎉 *These are excellent final results.**

🥇 XGBoost	596.65	113.18	5.44%	0.8384	Excellent
🥈 LightGBM	598.55	121.18	6.43%	0.8374	Good
🥉 Random Forest	600.47	122.42	6.36%	0.8363	Overfitting

## Interpretation

### RMSE = 546.62

On average, your predictions are about **₹/$546** (depending on the currency of `posted_rate`) away from the true freight rate, with larger errors penalized more.

### MAE = 102.21

On average, the prediction differs by only about **102 units** from the true freight rate.

This is a very good practical error.

### MAPE = 5.16%

This is one of the strongest metrics.

It means:

> **On average, your predictions are only about 5.16% away from the actual freight rate.**

In many regression applications, a MAPE below **10%** is considered very good. Around **5%** is excellent.

### R² = 0.8616

Your model explains **86.16% of the variance** in freight rates.

That's a strong result for a real-world tabular regression problem.

# Final model summary

**Model:** XGBoost Regressor (optimized with RandomizedSearchCV)

**Best Hyperparameters**

#python
{
    'subsample': 1.0,
    'reg_lambda': 5,
    'reg_alpha': 0,
    'n_estimators': 300,
    'min_child_weight': 5,
    'max_depth': 3,
    'learning_rate': 0.05,
    'gamma': 0,
    'colsample_bytree': 1.0
}
```

### Final Test Performance

| Metric   |      Value |
| -------- | ---------: |
| **RMSE** | **546.62** |
| **MAE**  | **102.21** |
| **MAPE** |  **5.16%** |
| **R²**   | **0.8616** |

---

## Overall assessment

You've built a strong end-to-end ML solution:

* ✅ Data preprocessing
* ✅ Missing value imputation
* ✅ One-hot encoding
* ✅ Train/test split
* ✅ Cross-validation
* ✅ Multiple baseline models
* ✅ Hyperparameter tuning with `RandomizedSearchCV`
* ✅ Comparison across Random Forest, XGBoost, and LightGBM
* ✅ Final evaluation on an untouched test set

This is a solid machine learning workflow and gives you a well-justified final model.

Finally Predictions and metrics are given that I saved in Results folder
Test xgboost_predictions_LoadedMODEL.csv (Model loading)

This was the Process of Train-Test dataset Prediction train-test-exp.csv

Stage 2: Validation Pipeline
Notebook code file Complete: "validation unknown.ipynb"

Similarly prepare data 
remove rows of weights invalid

transform date by splitting it into month,day,weekday,week of year,day of year format.

load the previous saved imputer.pkl, onehot_encoder.pkl
and perform similar steps of data imputation by transforming this pretrained states


and finally predict the predicted_rate on validations.csv dataset after data preparation and other steps using 

Model File that was trained on train-test.csv dataset:
best_xgboost_model.pkl

Stage 3:December Dataset 
Notebook code file Complete: "december_Model_code.ipynb"

Similarly prepare the dataset as you did with validation stage 2:

and do prediction
with the same 
best_xgboost_model.pkl model file

Finally using score.py(Modified) we generate the graph

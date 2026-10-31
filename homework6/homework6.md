# Homework 6 - Machine Learning Zoomcamp 2026

This homework involves building and tuning regression models to predict car fuel efficiency (column fuel_efficiency_mpg). We'll compare decision trees, random forests, and XGBoost, then tune hyperparameters.

> **Note**: This homework uses the pinned 2026 car fuel-efficiency release in the course repository. The dataset is downloaded from:
> <https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv>

## Setup

```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xbp

df = pd.read_csv('car_fuel_efficiency_2026.csv').fillna(0)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
target = 'fuel_efficiency_mpg'
y_train, y_val, y_test = df_train[target].values, df_val[target].values, df_test[target].values
df_train = df_train.drop(columns=[target]); df_val = df_val.drop(columns=[target]); df_test = df_test.drop(columns=[target])
dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(df_train.to_dict(orient='records'))
X_val = dv.transform(df_val.to_dict(orient='records'))
X_test = dv.transform(df_test.to_dict(orient='records'))
feature_names = dv.get_feature_names_out()
```

## Question 1 - Decision Tree with max_depth=1

Train a decision tree regressor with max_depth=1. Using DictVectorizer, map the one-hot split feature back to its original column name.

**Which original feature is used for splitting the data?**

- 'vehicle_weight'
- 'model_year'
- 'origin'
- 'fuel_type'

**Answer**: 'origin'

**Explanation**: The root split of a max_depth=1 decision tree uses the origin feature (one-hot encoded as origin=USA, origin=Europe, origin=Asia after DictVectorizer). This feature captures the geographical origin of the car (USA, Europe, Asia), which is a strong predictor of fuel efficiency.

---

## Question 2 - Random Forest with n_estimators=10

Train a random forest regressor with n_estimators=10, random_state=1, n_jobs=-1.

**What's the RMSE of this model on the validation data?**

- 0.1837
- 1.837
- 18.37
- 183.7

**Answer**: 1.837

**Explanation**: With 10 estimators and random_state=1, the random forest achieves an RMSE of approximately 1.837 on the validation set. The n_jobs=-1 parameter uses all available CPU cores for parallel training.

---

## Question 3 - Experiment with n_estimators

Try n_estimators values: [10, 50, 100, 150], with random_state=1. Evaluate on the validation dataset.

**Which value gives the lowest validation RMSE?** (Consider 3 decimal places)

- 10
- 50
- 100
- 150

**Answer**: 50

**Explanation**: Evaluating random forests with different numbers of estimators shows that n_estimators=50 achieves the lowest validation RMSE (with 3 decimal places precision). Increasing beyond 50 provides diminishing returns.

---

## Question 4 - Select the best max_depth

Try max_depth values: [10, 15, 20, 25]. For each max_depth, try n_estimators from [10, 50, 100, 150]. Fix random_state=1. Calculate the mean RMSE for each max_depth.

**What's the best max_depth, using the mean RMSE?**

- 10
- 15
- 20
- 25

**Answer**: 20

**Explanation**: Performing a grid search over max_depth in {10, 15, 20, 25} with n_estimators in {10, 50, 100, 150}, the mean validation RMSE is lowest for max_depth=20. Deeper trees (25) overfit, while shallower trees (10, 15) underfit the data.

---

## Question 5 - Feature Importance

Extract feature importance from a random forest with n_estimators=10, max_depth=20, random_state=1, n_jobs=-1.

**What's the most important feature among these four?**

- vehicle_weight
- horsepower
- acceleration
- engine_displacement

**Answer**: vehicle_weight

**Explanation**: The random forest's feature_importances_ attribute shows that vehicle_weight has the highest importance among the four candidate features. This makes physical sense: heavier vehicles generally consume more fuel per mile, so weight is a strong predictor of fuel efficiency.

---

## Question 6 - XGBoost with eta tuning

Train an XGBoost model with 100 rounds, comparing eta=0.3 vs eta=0.1.

**Which eta leads to the best RMSE score on the validation dataset?**

- 0.3
- 0.1
- Both give equal value

**Answer**: 0.1

**Explanation**: Reducing the learning rate (eta) from 0.3 to 0.1 improves the XGBoost model's performance on the validation set. Although a smaller eta requires more boosting rounds to converge, it typically leads to better generalization and lower test error. With 100 rounds and max_depth=6, eta=0.1 achieves the best RMSE.

---

## Learning in Public

Share what you learned from this homework in a short post, chart, or notebook excerpt. Use #mlzoomcamp, tag Alexey Grigorev or DataTalksClub, and submit the post link with your homework.

**Example post for LinkedIn**:

```text
🚀 Completed Module 6 of Machine Learning Zoomcamp 2026!

I compared decision trees, random forests, and XGBoost for car fuel-efficiency
regression, then tuned tree depth, forest size, and learning rate.

My repo: <LINK>

#mlzoomcamp @Alexey Grigorev
```

**Example post for X**:

```text
Compared tree ensembles in Module 6 of #mlzoomcamp.

🌳 Decision tree
🌲 Random forest
⚡ XGBoost
📉 Tuned the validation RMSE

My repo: <LINK>
```

---

## Submit the results

Submit your results here: <https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw06>

*The numerical options are calculated from the pinned 2026 release. If your answer doesn't match any option exactly, select the closest one. If it's exactly in between two options, select the higher value.*

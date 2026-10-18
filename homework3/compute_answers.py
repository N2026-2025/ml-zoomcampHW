"""Compute Homework 3 answers (lead scoring 2026 release).
Writes homework3/answers.json with all Q1-Q6 results.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
CSV = HERE / "course_lead_scoring_2026.csv"
OUT = HERE / "answers.json"

df = pd.read_csv(CSV)

# ---- Data preparation: categorical -> 'NA', numerical -> 0.0
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].fillna("NA")
    else:
        df[c] = df[c].fillna(0.0)

results = {}
results["shape"] = list(df.shape)
results["missing_after"] = {c: int(df[c].isnull().sum()) for c in df.columns}

# ---- Q1: mode of industry
q1 = df["industry"].mode()[0]
results["q1_mode_industry"] = str(q1)

# ---- Q2: correlation matrix of numerical features
num_cols = list(df.select_dtypes(include=[np.number]).columns.drop("converted"))
results["numeric_cols"] = num_cols
corr = df[num_cols].corr()
results["q2_corr"] = {f"{k[0]} x {k[1]}": round(float(v), 4) for k, v in corr.abs().unstack().items() if k[0] != k[1]}
pairs = {
    "interaction_count and lead_score": ("interaction_count", "lead_score"),
    "number_of_courses_viewed and lead_score": ("number_of_courses_viewed", "lead_score"),
    "number_of_courses_viewed and interaction_count": ("number_of_courses_viewed", "interaction_count"),
    "annual_income and interaction_count": ("annual_income", "interaction_count"),
}
q2_scores = {name: round(float(corr.loc[a, b]), 4) for name, (a, b) in pairs.items()}
results["q2_scores"] = q2_scores
results["q2_best"] = max(q2_scores, key=q2_scores.get)

# ---- Split (exact calls from the homework)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
results["split_sizes"] = {
    "train": len(df_train), "val": len(df_val), "test": len(df_test),
}
y_train = df_train["converted"].values
y_val = df_val["converted"].values

cat_cols = ["lead_source", "industry", "employment_status", "location"]
num_all = ["annual_income", "number_of_courses_viewed", "interaction_count", "lead_score"]

# ---- Q3: mutual information (train only), rounded to 2 decimals
mi = {}
for col in cat_cols:
    s = mutual_info_classif(
        df_train[[col]].apply(lambda x: x.astype("category").cat.codes),
        y_train, discrete_features=True, random_state=42,
    )[0]
    mi[col] = round(float(s), 2)
results["q3_mi"] = mi
results["q3_best"] = max(mi, key=mi.get)

# ---- Q4: logistic regression with one-hot encoding
def make_X(dfin, cols):
    X = pd.get_dummies(dfin[cols], drop_first=False, dtype=float)
    return X

all_cols = cat_cols + num_all
X_train = make_X(df_train, all_cols)
X_val = make_X(df_val, all_cols).reindex(columns=X_train.columns, fill_value=0.0)

model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_val, model.predict(X_val))
results["q4_accuracy"] = float(acc)
results["q4_accuracy_rounded2"] = round(float(acc), 2)

# ---- Q5: feature elimination (group-wise: each original feature)
base_acc = acc
diffs = {}
for feat in all_cols:
    keep = [c for c in all_cols if c != feat]
    Xt = make_X(df_train, keep)
    Xv = make_X(df_val, keep).reindex(columns=Xt.columns, fill_value=0.0)
    m = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
    m.fit(Xt, y_train)
    a = accuracy_score(y_val, m.predict(Xv))
    diffs[feat] = float(base_acc - a)
results["q5_base_accuracy"] = float(base_acc)
results["q5_diffs"] = {k: round(float(v), 4) for k, v in diffs.items()}
results["q5_diffs_full"] = {k: float(v) for k, v in diffs.items()}
subset = {k: v for k, v in diffs.items()
          if k in ("lead_source", "number_of_courses_viewed", "interaction_count")}
results["q5_subset"] = {k: round(float(v), 4) for k, v in subset.items()}
results["q5_smallest"] = min(subset, key=lambda k: abs(subset[k]))

# ---- Q6: regularized logistic regression over C values
q6 = {}
q6_full = {}
for C in [0.000001, 0.00001, 0.0001, 0.001]:
    m = LogisticRegression(solver="liblinear", C=C, max_iter=1000, random_state=42)
    m.fit(X_train, y_train)
    a = accuracy_score(y_val, m.predict(X_val))
    q6_full[str(C)] = round(float(a), 4)
    q6[str(C)] = round(float(a), 3)
results["q6_scores"] = q6
results["q6_full"] = q6_full
best = max(q6.values())
cands = [c for c, v in q6.items() if v == best]
results["q6_best"] = min(cands, key=float)

OUT.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
print("written:", OUT)

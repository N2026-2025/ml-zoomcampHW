import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.10"}

C = []

def M(src):
    C.append(nbf.v4.new_markdown_cell(src))

def J(src):
    C.append(nbf.v4.new_code_cell(src))

M("# Homework 4 - Model Evaluation and Tuning (ML Zoomcamp 2026)\n\n**Dataset:** `course_lead_scoring_2026.csv` - pinned 2026 release (`cohorts/2026/data/`). The notebook first tries the local CSV copy, otherwise it downloads the official homework URL.\n\n**Goal:** evaluate a logistic-regression classifier (`converted` target) with ROC AUC, precision, recall and F1, then tune it with 5-fold cross-validation.\n\nQuestions **Q1-Q6** + Answers Summary + self-verification cell.\n\nForm: <https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw04>")

M("## Setup\n\nImportamos las librerias necesarias. Los comentarios del codigo estan en espanol; los encabezados, en ingles.")

J("# Importamos pandas/numpy para datos, matplotlib para graficas y sklearn para el modelo\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\nimport matplotlib\nmatplotlib.use(\"Agg\")\nimport matplotlib.pyplot as plt\nfrom IPython.display import Image\nfrom sklearn.feature_extraction import DictVectorizer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import roc_auc_score\nfrom sklearn.model_selection import KFold, train_test_split")

M("## Load the dataset\n\nLeemos el CSV local (`homework4/course_lead_scoring_2026.csv`). Si no existe, se descarga la URL oficial del homework.")

J("# Cargamos la copia local; si falta, usamos la URL oficial fijada 2026\nLOCAL_CSV = Path(\"course_lead_scoring_2026.csv\")\nURL = \"https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv\"\ndf = pd.read_csv(LOCAL_CSV if LOCAL_CSV.exists() else URL)\nprint(df.shape)\ndf.head()")

M("## Data preparation\n\nRevisamos los nulos: las categoricas se rellenan con `'NA'` y las numericas con `0.0`, tal como pide el enunciado.")

J("# Revisamos nulos y rellenamos segun el tipo de columna\nprint(df.isnull().sum())\ncat_cols = df.select_dtypes(include=\"object\").columns.tolist()\nnum_cols = [c for c in df.select_dtypes(exclude=\"object\").columns if c != \"converted\"]\nfor c in cat_cols:\n    df[c] = df[c].fillna(\"NA\")\nfor c in num_cols:\n    df[c] = df[c].fillna(0.0)\nprint(\"nulls after fill:\", int(df.isnull().sum().sum()))")

M("## Train / validation / test split\n\nUsamos las llamadas exactas del enunciado (`random_state=1`): 3000 train / 1000 val / 1000 test. La `y` es la columna `converted`.")

J("# Partimos con las llamadas exactas del enunciado\ndf_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)\ndf_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)\nprint(len(df_train), len(df_val), len(df_test))\ny_train = df_train[\"converted\"].values\ny_val = df_val[\"converted\"].values")

M("## Q1: ROC AUC feature importance\n\nUsamos cada variable numerica como score y calculamos su AUC en train. Si el AUC sale menor que 0.5, invertimos la variable con `-` (la correlacion negativa se vuelve positiva). Gana la de mayor AUC.")

J("# AUC por variable numerica; invertimos con `-` si sale < 0.5\nq1_cols = [\"lead_score\", \"number_of_courses_viewed\", \"interaction_count\", \"annual_income\"]\nq1 = {}\nfor c in q1_cols:\n    s = df_train[c].values\n    auc = roc_auc_score(y_train, s)\n    if auc < 0.5:\n        auc = roc_auc_score(y_train, -s)\n    q1[c] = round(float(auc), 4)\n    print(c, q1[c])\nbest_q1 = max(q1, key=q1.get)\nprint(\"Best:\", best_q1)")

M("## Q2: Training the model\n\nOne-hot encoding con `DictVectorizer` y regresion logistica (`solver='liblinear', C=1.0, max_iter=1000`). Medimos el AUC en validacion.\n\nNota de compatibilidad: con las versiones nuevas de scikit-learn/scipy, `DictVectorizer` genera matrices dispersas con indices int64 que `liblinear` rechaza; por eso convertimos a denso con `.toarray()` (el dataset es pequeno y el resultado no cambia).")

J("# One-hot con DictVectorizer + regresion logistica; AUC en validacion\nfeatures = [c for c in df.columns if c != \"converted\"]\ndv = DictVectorizer()\nX_train = dv.fit_transform(df_train[features].to_dict(orient=\"records\"))\nX_val = dv.transform(df_val[features].to_dict(orient=\"records\"))\nX_train, X_val = X_train.toarray(), X_val.toarray()\nmodel = LogisticRegression(solver=\"liblinear\", C=1.0, max_iter=1000)\nmodel.fit(X_train, y_train)\ny_pred = model.predict_proba(X_val)[:, 1]\nauc_val = round(float(roc_auc_score(y_val, y_pred)), 3)\nprint(\"Validation AUC:\", auc_val)")

M("## Q3: Precision and Recall\n\nEvaluamos el modelo en validacion para cada umbral de 0.0 a 1.0 (paso 0.01), calculamos precision y recall, y graficamos ambas curvas. El cruce (ignorando umbrales donde ambas metricas son cero) es el umbral con menor diferencia absoluta.")

J("# Barremos umbrales, calculamos precision/recall y graficamos\nthresholds = np.arange(0.0, 1.001, 0.01)\nrows = []\nfor t in thresholds:\n    pred = y_pred >= t\n    tp = int((pred & (y_val == 1)).sum())\n    fp = int((pred & (y_val == 0)).sum())\n    fn = int(((~pred) & (y_val == 1)).sum())\n    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0\n    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0\n    rows.append((round(float(t), 2), p, r))\npr = pd.DataFrame(rows, columns=[\"threshold\", \"precision\", \"recall\"])\nplt.figure()\nplt.plot(pr.threshold, pr.precision, label=\"precision\")\nplt.plot(pr.threshold, pr.recall, label=\"recall\")\nplt.xlabel(\"threshold\")\nplt.ylabel(\"score\")\nplt.legend()\nplt.grid(True)\nplt.savefig(\"q3_precision_recall.png\")\nplt.close()\nnz = pr[~((pr.precision == 0) & (pr.recall == 0))].copy()\nnz[\"gap\"] = (nz.precision - nz.recall).abs()\nbest_t = float(nz.sort_values(\"gap\").iloc[0].threshold)\nprint(\"Intersection threshold:\", best_t)\nImage(\"q3_precision_recall.png\")")

M("## Q4: F1 score\n\nCombinamos precision (P) y recall (R) con F1 = 2*P*R/(P+R) para cada umbral y graficamos la curva. El maximo indica el mejor equilibrio.")

J("# F1 para cada umbral y su grafica\nf1_rows = []\nfor t, p, r in rows:\n    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0\n    f1_rows.append((t, f1))\nf1df = pd.DataFrame(f1_rows, columns=[\"threshold\", \"f1\"])\nplt.figure()\nplt.plot(f1df.threshold, f1df.f1)\nplt.xlabel(\"threshold\")\nplt.ylabel(\"F1\")\nplt.grid(True)\nplt.savefig(\"q4_f1.png\")\nplt.close()\nbest_f1_t = float(f1df.sort_values(\"f1\", ascending=False).iloc[0].threshold)\nprint(\"Best F1 threshold:\", best_f1_t)\nImage(\"q4_f1.png\")")
M("## Q5: 5-Fold CV\n\nEvaluamos el modelo con `KFold(n_splits=5, shuffle=True, random_state=1)` sobre `df_full_train` y medimos la desviacion estandar de los AUC entre folds.")

J("# 5-fold CV sobre df_full_train con los mismos parametros del modelo\ndv5 = DictVectorizer()\nX_full = dv5.fit_transform(df_full_train[features].to_dict(orient=\"records\")).toarray()\ny_full = df_full_train[\"converted\"].values\nkf = KFold(n_splits=5, shuffle=True, random_state=1)\nscores = []\nfor tr, va in kf.split(X_full):\n    m = LogisticRegression(solver=\"liblinear\", C=1.0, max_iter=1000)\n    m.fit(X_full[tr], y_full[tr])\n    scores.append(roc_auc_score(y_full[va], m.predict_proba(X_full[va])[:, 1]))\nprint([round(float(s), 4) for s in scores])\nprint(\"std:\", round(float(np.std(scores)), 4))")

M("## Q6: Hyperparameter Tuning\n\nRepetimos el 5-fold CV para C en [0.000001, 0.001, 1] y comparamos media y std del AUC (redondeadas a 3 decimales). Desempates: menor std y luego menor C.")

J("# Barrido de C con el mismo KFold; desempatamos por menor std y menor C\nresults = []\nfor Cval in [0.000001, 0.001, 1]:\n    ss = []\n    for tr, va in kf.split(X_full):\n        m = LogisticRegression(solver=\"liblinear\", C=Cval, max_iter=1000)\n        m.fit(X_full[tr], y_full[tr])\n        ss.append(roc_auc_score(y_full[va], m.predict_proba(X_full[va])[:, 1]))\n    results.append((Cval, round(float(np.mean(ss)), 3), round(float(np.std(ss)), 3)))\n    print(Cval, results[-1][1], results[-1][2])\nbest_C = min(results, key=lambda r: (-r[1], r[2], r[0]))\nprint(\"Best C:\", best_C[0])")

M("## Answers Summary\n\nRespuestas para el formulario (calculadas por este notebook):\n\n| Question | Answer |\n|---|---|\n| Q1 - best ROC AUC feature | `lead_score` (0.7882) |\n| Q2 - validation AUC | **0.732** |\n| Q3 - precision/recall intersection | **0.63** |\n| Q4 - max F1 threshold | **0.41** |\n| Q5 - CV std | **0.007** |\n| Q6 - best C | **0.001** (mean 0.749) |")

J("# Autocomprobacion: todo debe coincidir con las respuestas esperadas\nassert (len(df_train), len(df_val), len(df_test)) == (3000, 1000, 1000)\nassert best_q1 == \"lead_score\" and q1[\"lead_score\"] == 0.7882\nassert auc_val == 0.732\nassert best_t == 0.63\nassert best_f1_t == 0.41\nassert round(float(np.std(scores)), 3) == 0.007\nassert best_C[0] == 0.001\nprint(\"ALL CHECKS PASSED\")")

M("## Notes\n\nTodo usa `random_state=1`, asi que los resultados son identicos en cada ejecucion. Para reproducir: abre el notebook en VS Code/Jupyter con `homework4/` como carpeta de trabajo y haz **Restart & Run All**.")

nb.cells = C
OUT = Path(__file__).resolve().parent / "notebook.ipynb"
nbf.write(nb, OUT)
print("wrote", OUT, "cells:", len(C))

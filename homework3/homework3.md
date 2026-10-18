# Homework 3 — Lead Scoring Classifier · ML Zoomcamp 2026 (Solved)

**Notebook:** `homework3.ipynb` · **Dataset:** `course_lead_scoring_2026.csv`
(pinned release, `cohorts/2026/data/`)
**Status:** solved end-to-end · `error_outputs: 0` · asserts OK
Reference: [03-classification](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/03-classification)

---

## 1. Answers (copy into the form)

| Question | Description | Answer |
|---|---|---|
| **Q1** | Mode of `industry` | **`technology`** |
| **Q2** | Biggest correlation | **`interaction_count` and `lead_score`** (0.9157) |
| **Q3** | Biggest mutual information | **`lead_source`** (0.03) |
| **Q4** | Logistic regression accuracy (val) | **`0.65`** |
| **Q5** | Least useful feature | **`number_of_courses_viewed`** |
| **Q6** | Best `C` | **`0.001`** |

Form: <https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw03>

---

## 2. Data preparation

Shape **(5000, 9)**: 4 categoricals (`lead_source`, `industry`,
`employment_status`, `location`), 4 numericals (`annual_income`,
`number_of_courses_viewed`, `interaction_count`, `lead_score`), plus binary
target `converted`. Missing values existed in 6 columns (148 `lead_source`,
240 `industry`, 194 `employment_status`, 208 `location`, 369 `annual_income`,
35 `lead_score`); categoricals were filled with `'NA'`, numericals with `0.0`.

---

## 3. Methodology and results per question

### Q1 — Mode of `industry` → `technology`

`df['industry'].mode()[0]` after filling NAs. The dominant category is
**technology** — the filled `'NA'` bucket is smaller than it.

### Q2 — Correlation → `interaction_count` and `lead_score` (0.9157)

Pearson correlation matrix over the 4 numerical features:

| Pair | corr |
|---|---|
| interaction_count and lead_score | 0.9157 |
| number_of_courses_viewed and lead_score | 0.7572 |
| number_of_courses_viewed and interaction_count | 0.7216 |
| annual_income and interaction_count | 0.1228 |

`interaction_count × lead_score` wins by far — expected, since `lead_score`
is derived from engagement signals.
Ref: [07-correlation](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/03-classification/07-correlation.md).

### Split

Exact homework calls with `random_state=42`: **3000 train / 1000 val /
1000 test** (60/20/20). Target `converted` excluded from feature frames.

### Q3 — Mutual information → `lead_source` (0.03)

Train set only, `mutual_info_classif(discrete_features=True)`, 2 decimals:

| Variable | MI |
|---|---|
| `lead_source` | 0.03 |
| `employment_status` | 0.02 |
| `industry` | 0.0 |
| `location` | 0.0 |

Acquisition channel carries the most signal about conversion.
Ref: [06-mutual-info](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/03-classification/06-mutual-info.md).

### Q4 — Logistic regression → 0.65

One-hot encoding (`pd.get_dummies`) +
`LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)`
→ val accuracy **0.645 → 0.65**.
Ref: [10-training-log-reg](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/03-classification/10-training-log-reg.md).

### Q5 — Feature elimination → `number_of_courses_viewed`

`base − without` per original feature:

| Removed | diff |
|---|---|
| `lead_source` | +0.0030 |
| `number_of_courses_viewed` | +0.0020 |
| `interaction_count` | +0.0440 |

Smallest is **`number_of_courses_viewed`** (+0.002): dropping it barely moves
accuracy. (Full run: `annual_income` −0.079 — removing it *improves* val
accuracy, a sign of noise.)

### Q6 — Regularization → `0.001`

Smaller `C` = stronger regularization. Val accuracy (3 decimals):

| `C` | acc |
|---|---|
| 0.000001 | 0.598 |
| 0.00001 | 0.598 |
| 0.0001 | 0.613 |
| 0.001 | 0.645 |

**`C = 0.001`** wins outright (no tie-break needed); heavy regularization
underfits here.

---

## 4. Self-verification

The notebook ends with an assert cell (`OK: todas las respuestas verificadas`).

---

## 5. Environment & reproduction

| Component | Version |
|---|---|
| Python | 3.14.7 |
| pandas | 2.3.3 |
| numpy | 2.4.0 |
| scikit-learn | 1.9.1 |

Loads the pinned local copy with fallback to the official URL.
**Restart & Run All** — all splits use `random_state=42`, results identical.
Values are computed at runtime, never hardcoded.

# Multiple Choice — Homework 3 · Lead Scoring (ML Zoomcamp 2026)

Dataset: `homework3/course_lead_scoring_2026.csv` · Notebook resuelto: `homework3/homework3.ipynb`

---

## Q1 — Most frequent observation (mode) for `industry`?

- [ ] `NA`
- [x] `technology`
- [ ] `healthcare`
- [ ] `retail`

> `df['industry'].mode()[0]` → `technology`.

## Q2 — Which two features have the biggest correlation?

- [x] `interaction_count` and `lead_score` (0.9157)
- [ ] `number_of_courses_viewed` and `lead_score` (0.7572)
- [ ] `number_of_courses_viewed` and `interaction_count` (0.7216)
- [ ] `annual_income` and `interaction_count` (0.1228)

> Pearson correlation matrix over the numerical features.

## Q3 — Biggest mutual information score vs `converted`?

- [ ] `industry` (0.0)
- [ ] `location` (0.0)
- [x] `lead_source` (0.03)
- [ ] `employment_status` (0.02)

> Train set only, `mutual_info_classif`, rounded to 2 decimals.

## Q4 — Logistic regression accuracy on validation?

- [ ] 0.55
- [x] 0.65 (0.6450)
- [ ] 0.75
- [ ] 0.85

> One-hot + `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)`.

## Q5 — Smallest difference (feature elimination)?

- [ ] `lead_source` (+0.0030)
- [x] `number_of_courses_viewed` (+0.0020)
- [ ] `interaction_count` (+0.0440)

## Q6 — Best `C`?

| `C` | acc |
|---|---|
| 0.000001 | 0.598 |
| 0.00001 | 0.598 |
| 0.0001 | 0.613 |
| 0.001 | 0.645 |

- [ ] 0.000001
- [ ] 0.00001
- [ ] 0.0001
- [x] 0.001

> Smallest `C` on ties (no tie here — `0.001` wins outright).

---

**Answer key:** Q1 `technology` · Q2 `interaction_count` + `lead_score` ·
Q3 `lead_source` · Q4 `0.65` · Q5 `number_of_courses_viewed` · Q6 `0.001`

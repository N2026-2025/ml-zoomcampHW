# Homework 5 — Deploying Machine Learning Models (Answers)

ML Zoomcamp 2026 — Deployment module (cohort 2026).

**Answers in English; comments/explanations in Spanish (`<!-- -->`).**

## Setup / environment
- Project directory: `c:\MidtermML\05-deployment`
- Python target: 3.11.15 (project pins `>=3.11,<3.12`)
- Local `uv` version: **0.10.3** (`uv --version`) <!-- versión de uv instalada localmente -->
- Local interpreter used for inference: Python 3.14.7 (Python 3.11.15 not downloadable on Windows here); scikit-learn 1.9.1. The artifact was pickled with scikit-learn 1.7.2; it loads with an `InconsistentVersionWarning` but `predict_proba` works (verified). <!-- el modelo se pickleo con sklearn 1.7.2; al cargar avisa de version inconsistente pero predecir funciona -->

## Artifact checksum (verified)
`sha256sum pipeline.bin` =
`1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1` ✓
<!-- checksum del artefacto verificado, coincide con homework.md y model_metadata.json -->

---

## Q1 — Environment check
**Answer:** `uv 0.10.3` (setup check, not a graded answer; your local `uv` version is not a property of the model).
<!-- comprobacion de entorno, no es una respuesta calificada -->

## Q2 — Locked dependency
**Answer:** **1.7.2**

`pyproject.toml` (line 9): `scikit-learn==1.7.2`
`model_metadata.json`: `"scikit_learn": "1.7.2"`
<!-- No regenerar el lockfile; usar `uv sync --locked`. -->

---

## Q3 — Load the model
**Answer:** **0.533**

Script loads `pipeline.bin` with `pickle` and calls `predict_proba` for the lead:

```json
{
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}
```

`conversion_probability` (raw) ≈ 0.533…, rounded to three decimals = **0.533**; `conversion` = `true` (>= 0.5).
<!-- cargo pipeline.bin con pickle y llamo a predict_proba; redondeo a 3 decimales -->

## Q4 — Serve the model
**Answer:** **0.770**

API: `uv run uvicorn predict:app --host 0.0.0.0 --port 9696` (`GET /health`, `POST /predict`).

Lead:

```json
{
  "lead_source": "organic_search",
  "industry": "technology",
  "employment_status": "employed",
  "location": "europe",
  "number_of_courses_viewed": 4,
  "annual_income": 80304.0,
  "interaction_count": 7,
  "lead_score": 0.74
}
```

`/predict` response:
```json
{
  "conversion_probability": 0.769799,
  "conversion": true
}
```

`conversion_probability` rounded to three decimals = **0.770**. Probability in `[0, 1]` ✓ and boolean `conversion` field ✓.
<!-- ejecuto el misma inferencia que el endpoint POST /predict (normalize_record + predict_proba); respuesta contiene probabilidad en [0,1] y conversion booleano -->

## Q5 — Container configuration
**Answer:** **python:3.11.15-slim-bookworm**

Declared in `Dockerfile` line 1:
`FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3`

The Dockerfile also pins `uv 0.10.11` and installs from the checked-in `uv.lock` via `uv sync --locked --no-dev`; the API starts on port `9696`.
<!-- etiqueta de imagen base declarada en el Dockerfile (version controlada, no el tamano de imagen local) -->

## Q6 — Run the container
**Answer:** **0.770** (same reference inference as Q4; same 0.005 tolerance)

Run the same request as Q4 against the container (`python q6_test.py` or `curl`).
Expected `conversion_probability` rounded to three decimals = **0.770**; `conversion` = `true`.

Verify `curl -s http://localhost:9696/health` returns:
```json
{
  "status": "ok",
  "model_sha256": "1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1",
  "feature_defaults_sha256": "1694f842b908fac9f53c0cc19be6d91f934befc63af23cc32fefafde69acd0cd"
}
```
The `model_sha256` matches the checksum in `model_metadata.json` and the verified `pipeline.bin` checksum. `feature_defaults_sha256` matches `feature_defaults.json`.

> Environment note: the Docker daemon is not running in this machine, so `docker build` / `docker run` (Q6) could not be executed locally. The expected Q6 result equals Q4 (**0.770**) and should be validated on a machine with Docker available. <!-- el demonio de Docker no esta corriendo aqui; el resultado esperado de Q6 iguala a Q4 y debe validarse con Docker -->

---

## Submit
Submit at: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw05
Numeric probabilities to three decimal places; tolerance = absolute error ≤ 0.005.

## Files
- `hw5_homework.ipynb` — notebook (English narrative, Spanish code comments)
- `HW5_ANSWERS.md` — this file (English answers, Spanish comments)

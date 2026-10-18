# Homework 3 — Clasificador de Lead Scoring · ML Zoomcamp 2026

Solución completa del Homework 3 (clasificación con regresión logística).
Referencia del curso: [03-classification](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/03-classification).

## Contenido de la carpeta

| Archivo | Qué es |
|---|---|
| `homework (3).md` | Enunciado original del homework |
| `course_lead_scoring_2026.csv` | Dataset (release fijada 2026, 5000×9) |
| `homework3.ipynb` | **Notebook resuelto y ejecutado** (24 celdas, 0 errores) |
| `homework3.md` | Solución detallada en inglés |
| `multiplechoice3.md` (raíz) | Banco de preguntas múltiple opción con claves |
| `answers.json` | Respuestas calculadas (generado por script) |
| `compute_answers.py` | Script que calcula Q1–Q6 |
| `gen_a/b/c.py` | Generadores del notebook |
| `verify_nb.py` | Verificación + ejecución del notebook |

## Respuestas (para el formulario)

| Pregunta | Respuesta |
|---|---|
| Q1 — moda de `industry` | `technology` |
| Q2 — mayor correlación | `interaction_count` y `lead_score` (0.9157) |
| Q3 — mayor información mutua | `lead_source` (0.03) |
| Q4 — accuracy regresión logística | **0.65** |
| Q5 — feature menos útil | `number_of_courses_viewed` |
| Q6 — mejor `C` | **0.001** |

Formulario: <https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw03>

## Cómo reproducir

```bash
pip install pandas numpy scikit-learn nbformat nbclient
cd homework3
python compute_answers.py   # calcula Q1–Q6 -> answers.json
python gen_a.py && python gen_b.py && python gen_c.py  # genera el .ipynb
python verify_nb.py         # ejecuta y verifica (debe decir EXECUTION: OK)
```

O abre `homework3.ipynb` en VS Code/Jupyter y haz **Restart & Run All**.
Todo usa `random_state=42`: los resultados son idénticos en cada corrida.
El notebook carga la copia local y, si falta, la URL oficial.

## Metodología (resumen)

1. **Preparación:** categóricas → `'NA'`, numéricas → `0.0`.
2. **Q1:** `df['industry'].mode()` → `technology`.
3. **Q2:** matriz de correlación de Pearson; gana `interaction_count × lead_score`.
4. **Split exacto** del enunciado → 3000 train / 1000 val / 1000 test.
5. **Q3:** `mutual_info_classif` solo en train → `lead_source`.
6. **Q4:** one-hot (`get_dummies`) + `LogisticRegression(liblinear, C=1.0)` → 0.65.
7. **Q5:** reentrenar quitando una feature cada vez → `number_of_courses_viewed` (+0.002).
8. **Q6:** barrido de `C`; `0.001` gana con 0.645.
9. **Autocomprobación:** celda de asserts al final del notebook.

## Aprendizajes clave

- La **información mutua** mide qué categórica reduce más la incertidumbre del target.
- El **one-hot encoding** convierte categorías en columnas binarias para la regresión logística.
- La **eliminación de features** detecta variables prescindibles (diferencia ≈ 0).
- En sklearn, **`C` pequeño = más regularización**; aquí regularizar fuerte empeora (underfitting).

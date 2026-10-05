# Homework 2 — Machine Learning for Regression · ML Zoomcamp 2026

**Notebook:** `homework2.ipynb` · **Dataset:** `car_fuel_efficiency_2026.csv` (release *pinned* del repo del curso, `cohorts/2026/data/`)
**Estado:** ejecutado de principio a fin · `error_outputs: 0` · `execution_counts` 1–15 · asserts OK

---

## 1. Respuestas (para copiar en el formulario)

| Pregunta | Descripción | Respuesta |
|---|---|---|
| **Q1** | Columna con valores faltantes | **`horsepower`** |
| **Q2** | Mediana de `horsepower` | **`254`** |
| **Q3** | Mejor RMSE: 0 vs media | **`With mean`** |
| **Q4** | Mejor `r` | **`0`** |
| **Q5** | `np.std` de los 10 RMSE de validación | **`0.029`** |
| **Q6** | RMSE en test (seed 9, train+val, `r=0.001`) | **`2.236`** |

Formulario: <https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw02>

---

## 2. EDA — ¿tiene cola larga `fuel_efficiency_mpg`?

**No.** En esta release la distribución es prácticamente simétrica:

| Estadístico | Valor |
|---|---|
| Skewness del target crudo | **0.0808** |
| Media | 29.998 |
| Mediana | 30.000 |
| Q50 / Q90 / Q99 / máx | 30.0 / 33.8 / 36.9 / 41.2 |

Un skew de **0.08** está muy lejos del umbral de 1 que indicaría cola larga, y la media es prácticamente igual a la mediana. Además el máximo es 41.2 MPG: **no hay valores extremos** que tiren de la distribución.

> **Consecuencia metodológica (importante):** en la lecture de *car-price* el target `msrp` **sí** tenía cola larga, y por eso se aplicaba `np.log1p`. Aquí **no corresponde aplicarlo**. Lo verifiqué empíricamente: con `log1p` los RMSE quedan en escala logarítmica (Q6 ≈ 0.072) y **no coinciden con ninguna opción**; con el target **crudo** en MPG los valores **sí coinciden** con las opciones oficiales. Por eso el notebook predice sobre los **MPG crudos** y todos los RMSE están en unidades de MPG.

---

## 3. Metodología y resultados por pregunta

### Preparación

Columnas usadas: `engine_displacement`, `horsepower`, `vehicle_weight`, `model_year` (features) y `fuel_efficiency_mpg` (target). Los nombres de columna en esta release ya vienen en minúsculas y con `_`, así que la normalización `str.lower().str.replace(' ', '_')` de la lecture no hacía falta.

El split es **exactamente** el de la lecture (60/20/20), con `np.random.seed(...)` para reproducibilidad y `n_train` como remanente para no perder filas por redondeo:

```python
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

np.random.seed(seed)
idx = np.arange(n)
np.random.shuffle(idx)
```

Con seed 42: **train 6000 / val 2000 / test 2000**.

El entrenamiento usa la **ecuación normal** de la lecture (con columna de unos para el bias), y la regularización (ridge) suma `r` a la diagonal de la matriz de Gram:

```python
def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg
    w = np.linalg.inv(XTX).dot(X.T).dot(y)
    return w[0], w[1:]

def rmse(y, y_pred):
    return float(np.sqrt(np.mean((y - y_pred) ** 2)))
```

### Q1 — Columna con valores faltantes → `horsepower`

| Columna | NAs |
|---|---|
| `engine_displacement` | 0 |
| **`horsepower`** | **877** |
| `vehicle_weight` | 0 |
| `model_year` | 0 |

Es la única de las cuatro features con faltantes (8.77% del dataset).

### Q2 — Mediana de `horsepower` → **254**

`df['horsepower'].median()` = **254.0** (la columna tiene NAs, que pandas excluye de la mediana).

### Q3 — ¿0 o media? → **`With mean`**

Modelos **sin regularización**, entrenados con seed 42 y evaluados en validación. La media se calculó **solo con el train** (`254.46338337605272`) para no filtrar información del validation.

| Imputación | RMSE (val) | `round(score, 3)` |
|---|---|---|
| Con 0 | 2.205287 | 2.205 |
| **Con media** | **2.201822** | **2.202** |

**Gana la media** (2.2018 < 2.2053), aunque la diferencia es pequeña. Tiene sentido: con 0 el término `0·w` simplemente se anula y el modelo *ignora* esa feature, mientras que la media inyecta la información estadística del resto de la flota.

### Q4 — Mejor `r` → **0**

NAs rellenados con 0, ridge sobre la lista pedida, RMSE en validación con 4 decimales:

| `r` | RMSE (val) | `round(score, 4)` |
|---|---|---|
| **0** | **2.205293** | **2.2053** |
| 0.01 | 2.205828 | 2.2058 |
| 0.1 | 2.224143 | 2.2241 |
| 1 | 2.349229 | 2.3492 |
| 5 | 2.409380 | 2.4094 |
| 10 | 2.419470 | 2.4195 |
| 100 | 2.429202 | 2.4292 |

El mejor es **`r = 0`**, es decir **sin regularización**: el RMSE crece de forma monótona con `r`. No hay empate, así que la regla de "menor `r`" no llega a aplicarse. La razón es que estas cuatro features no están colineadas ni duplicadas, así que la inversa de la matriz de Gram está bien condicionada y no hay pesos descontrolados que regularizar.

### Q5 — `np.std` de los RMSE → **0.029**

10 seeds (`0`–`9`), imputación con 0, sin regularización, RMSE en validación de cada split:

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| RMSE | 2.2392 | 2.2021 | 2.1634 | 2.2044 | 2.2019 | 2.2458 | 2.2697 | 2.1908 | 2.2166 | 2.2070 |

`np.std(scores)` = 0.028781 → **`0.029`**. La desviación es baja (~1.3% del RMSE medio), así que el modelo es **estable** frente a la elección del split.

### Q6 — RMSE en test → **2.236**

Split con **seed 9**, se **combinan train + validation** (8000 filas), NAs con 0, ridge con `r = 0.001`, y evaluación en el test (2000 filas, intacto hasta el final):

```
Test RMSE = 2.235828  ->  round(..., 3) = 2.236
```

La lógica: una vez elegido `r` con el validation, ese set ya cumplió su función — se recicla dentro del training para aprovechar las 2000 filas extra, y el test queda reservado como evaluación final honesta.

---

## 4. Autocomprobación

El notebook incluye una celda de asserts que valida cada respuesta contra las listas de opciones exactas del homework, además de comprobar la consistencia interna (p. ej. que el `r` elegido sea realmente el mínimo, que haya 10 seeds, que `answer_q5` sea el redondeo real del std, y que el RMSE de Q6 sea el redondeo real del cálculo). Resultado de la corrida:

```
OK: all answers verified
```

---

## 5. Entorno y reproducción

| Componente | Versión |
|---|---|
| Python | 3.14.7 |
| pandas | 2.3.3 |
| numpy | 2.4.0 |
| nbformat / nbclient | 5.11.1 / 0.11.0 |

El dataset se carga desde la copia *pinned* local (`cohorts/2026/data/car_fuel_efficiency_2026.csv`) si existe, con fallback a la URL oficial del homework — igual que en `homework1.ipynb`.

Para reproducir: **Restart & Run All** en `homework2.ipynb` (los asserts vuelven a pasar y los resultados son idénticos, porque todos los split usan `np.random.seed`).

**Nota:** los valores no están hardcodeados — se calculan en la corrida, y la tabla de la sección 1 refleja lo que el notebook imprimió realmente.
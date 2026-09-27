
Todo verificado. Resumen final:

## Tarea completada — `homework1.ipynb` revisado, corregido y ejecutado

**Archivo:** `c:\mlzoomcamp\machine-learning-zoomcamp\01-intro\homework1\homework1.ipynb` (reescrito: 26 celdas — 14 markdown + 12 código — **ejecutado de principio a fin**, 0 errores, `nbformat` válido, kernelspec `.venv (3.14.7)`).

| Problema                                                                                                                                                                  | Corrección                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Q7:** `XTX = X.dot(X.T)` → matriz 7×7 de rango ≤ 2, `det = 0`, **no invertible** (y la nota de "determinante 0" era síntoma del error, no la causa) | `XTX = X.T.dot(X)` → 2×2, `det = 9.45e12` → `w = inv(XTX) @ X.T @ y`           |
| **Q7:** el filtro `df[df.origin=="Asia"]` no se encadenaba a la selección de columnas (usaba el df completo)                                                     | `asia = df[df.origin=="Asia"]` → `asia[[...]].head(7).to_numpy()`                  |
| **Q4:** solo imprimía `isnull().sum()` sin contar columnas                                                                                                       | se cuentan las columnas con NA (y se muestran:`horsepower` 877, `acceleration` 264) |
| **Q6:** sin conclusión                                                                                                                                             | se calcula y se concluye**"Yes, it decreased"** (254.0 → 252.0)                  |
| **Q5:** no filtraba por Asia (coincidía por azar)                                                                                                                  | ahora filtra`origin == "Asia"` antes de maximizar                                     |
| Encabezado`### Q2 Fuel types` (numeración incorrecta)                                                                                                                  | renumerado Q1–Q7 + tabla resumen final                                                 |

### Respuestas (todas calculadas en el notebook y verificadas por ejecución)

- **Q1** → pandas **3.0.6**
- **Q2** → **10000** registros
- **Q3** → **3** tipos de combustible (Diesel, Gasoline, Hybrid)
- **Q4** → **2** columnas con valores faltantes
- **Q5** → **41.2**
- **Q6** → **Yes, it decreased** (mediana 254.0 → 252.0 al rellenar con el mode 252.0)
- **Q7** → `sum(w) = 0.36919696904925486` → **0.369**

### Validación realizada

1. Ejecuté el notebook completo con `nbclient` usando el kernel del `.venv` (kernelspec temporal con ruta absoluta al intérprete, no depende de `python` en PATH).
2. Cada respuesta se imprime con la marca `Respuesta Qn:` y hay una **celda de autocomprobación con asserts** (`OK: todas las respuestas verificadas`).
3. Log de verificación: `EXECUTION: OK` · `NBFORMAT VALIDATE: OK` · **8/8 PASS → RESULT: ALL OK** (en `C:\Users\nesto\AppData\Local\Temp\hw1_verify.txt`).
4. Inspección del archivo: `execution_counts` 1–12, `error_outputs: 0`, la fórmula corregida en código y el error antiguo solo mencionado en el markdown explicativo ("por qué `X.dot(X.T)` estaba mal").

**Notas:** los datos se cargan desde la copia *pinned* local (`../../cohorts/2026/data/car_fuel_efficiency_2026.csv`, offline y determinista) con fallback a la URL oficial del homework; Q6 trabaja sobre una copia (`hp_filled`) para no mutar el dataset crudo. Para reproducir: *Restart & Run All* con el kernel `.venv (3.14.7)` — los asserts vuelven a pasar.

# Resumen del avance: Modelo Baseline (TF-IDF + Regresión Logística)

## 1. Objetivo y enfoque
Se desarrolló el primer modelo de referencia (*baseline*) para clasificar el sentimiento de comentarios en español en tres categorías: **Negativo (0)**, **Neutral (1)** y **Positivo (2)**. El propósito de este modelo es establecer una base rápida, interpretable y reproducible contra la cual comparar arquitecturas más complejas (como modelos *Transformer*).

---

## 2. Datos y partición
Se trabajó sobre el conjunto de datos previamente limpio (`208,446` comentarios válidos), manteniendo la partición estratificada para conservar la proporción original de clases:

* **Entrenamiento (`train`):** `166,756` comentarios (80%)
* **Prueba (`test`):** `41,690` comentarios (20%)

---

## 3. Pipeline del modelo
* **Vectorización:** `TfidfVectorizer` ajustado únicamente sobre los datos de entrenamiento (`fit_transform`) para evitar fuga de datos (*data leakage*), aplicando `transform` sobre el conjunto de prueba.
* **Clasificador:** `LogisticRegression` multiclase entrenado sobre las representaciones TF-IDF.
* **Notebook de experimentación:** `notebooks/02_baseline_tfidf_logreg.ipynb`
* **Artefactos guardados (`models/`):**
  * `models/tfidf_vectorizer.joblib`
  * `models/logreg_baseline.joblib`

---

## 4. Resultados en prueba (`test`)
Al evaluar el modelo con los `41,690` comentarios del conjunto de prueba, se obtuvieron las siguientes métricas:

| Clase / Métrica | Precision | Recall | F1-Score | Muestras |
| :--- | :---: | :---: | :---: | :---: |
| **Negativo (0)** | 0.79 | 0.66 | 0.72 | 9,485 |
| **Neutral (1)** | 0.74 | 0.85 | 0.79 | 13,638 |
| **Positivo (2)** | 0.84 | 0.83 | 0.84 | 18,567 |
| **Accuracy** | — | — | **0.80 (0.7964)** | **41,690** |
| **Macro Avg (Macro F1)** | **0.79** | **0.78** | **0.78 (0.7825)** | **41,690** |

### Lectura rápida de resultados
* **Desempeño general:** El modelo alcanza casi un **80% de exactitud global** (`0.7964`) y un **Macro F1 de `0.7825`**, con tiempos de inferencia prácticamente instantáneos (`0.0706 s` para todo el conjunto de prueba).
* **Fortaleza principal:** Buena capacidad para identificar comentarios **positivos** (`F1 = 0.84`) y alta cobertura en **neutrales** (`Recall = 0.85`).
* **Principal oportunidad de mejora:** La clase **Negativo (0)** presenta el *recall* más bajo (`0.66`), lo que indica que las quejas o comentarios negativos se están confundiendo con neutrales o positivos (posiblemente por ironía, negaciones compuestas o contexto que un modelo de bolsa de palabras no captura).
# Proyecto Integrador MLOps - Monitoreo de Modelo de Crédito

Este proyecto nace de un caso práctico de una empresa financiera.

Has iniciado tu labor en el equipo de Datos y Analítica como Científico de Datos Junior Advanced. Tu primera asignación es desarrollar un modelo predictivo con información histórica de créditos para anticipar el comportamiento de nuevos usuarios.

La empresa trabaja con un esquema de proyectos muy estructurado y no puede modificar la organización de carpetas, porque el despliegue a producción depende de pipelines automáticos en Jenkins. Por eso este proyecto debe mantenerse dentro de la estructura de carpetas existente.

También se espera que el desarrollo esté disponible en un repositorio público de GitHub.

## Caso de negocio

Una entidad de crédito quiere predecir si un cliente pagará su préstamo a tiempo (`Pago_atiempo`). El modelo se entrena con datos históricos y luego se monitorea en tiempo real para detectar si las condiciones de los clientes cambian con el tiempo.

El principal riesgo es el **data drift**: cuando las características de las nuevas solicitudes ya no se parecen a las usadas para entrenar el modelo, el resultado puede degradar.

## Qué hace el proyecto

- Carga el dataset `Base_de_datos.csv`.
- Realiza un análisis exploratorio de datos con `comprension_eda.ipynb`.
- Limpia los datos y crea nuevas variables con `ft_engineering.py`.
- Entrena y evalúa el modelo en `model_training_evaluation.py`.
- Despliega / prepara el modelo con `model_deploy.py`.
- Monitorea el comportamiento actual con `model_monitoring.py`.
- En el monitoreo:
  - divide los datos según `fecha_prestamo` en:
    - `reference_df`: datos históricos de referencia
    - `current_df`: datos actuales recientes
  - calcula métricas de drift entre histórico y actual
  - aplica el modelo entrenado sobre los datos actuales
  - muestra la probabilidad de pago (`prob_pago`) y la predicción binaria (`pred_label`)
  - presenta todo en una aplicación Streamlit.

## Flujo completo del proyecto

El proyecto está organizado como un flujo de MLOps sencillo pero completo:

1. `Cargar_datos.ipynb`: lectura inicial del dataset.
2. `comprension_eda.ipynb`: análisis exploratorio, revisión de nulos, distribuciones y relaciones con el target.
3. `ft_engineering.py`: limpieza, creación de variables derivadas y construcción del `preprocessor`.
4. `model_training_evaluation.py`: entrenamiento, tuning, comparación de modelos y exportación de métricas.
5. `model_deploy.py`: carga del mejor modelo y exposición de endpoints para predicción batch.
6. `model_monitoring.py`: dashboard Streamlit para revisar drift y comportamiento del modelo sobre datos actuales.

## Decisiones de preparación de datos

La calidad del modelo depende más de las variables usadas que del algoritmo por sí solo. En este proyecto se tomaron decisiones importantes en la etapa de feature engineering:

- Se eliminó `puntaje` porque era una variable **post-decision**. Ese score estaba calculado con información demasiado cercana o posterior al resultado de pago, por lo que introducía **fuga de información** (`data leakage`).
- Mantener `puntaje` hacía que el modelo quedara artificialmente sesgado y demasiado dependiente de una variable que prácticamente anticipaba el target.
- Por esa razón, en [mlops_pipeline/src/ft_engineering.py](mlops_pipeline/src/ft_engineering.py#L47) se descarta explícitamente junto con otras columnas irrelevantes o con potencial de fuga.
- También se limpiaron categorías corruptas en `tendencia_ingresos`, se trataron ceros inválidos en variables de saldo y se creó la bandera `tiene_datacredito`.
- Se generaron dos variables derivadas: `ratio_cuota_salario` y `ratio_deuda_ingresos`.

Estas decisiones no solo mejoran performance: también hacen que el modelo sea más consistente y más defendible desde el punto de vista metodológico.

## Resultado del entrenamiento

Según [data/model_summary.csv](data/model_summary.csv), el mejor modelo actual del proyecto es:

- `Random Forest (opt)`
- `ROC-AUC = 0.6706`
- `Accuracy = 0.7455`

Ese modelo es el que se usa como base para el despliegue en la API.

## Estructura del proyecto

```
mlops_pipeline/
└── src/
    ├── Cargar_datos.ipynb           # Carga del dataset
    ├── comprension_eda.ipynb        # Análisis exploratorio de datos
    ├── ft_engineering.py            # Feature engineering y transformación
    ├── model_training_evaluation.py # Entrenamiento y evaluación del modelo
    ├── model_deploy.py              # Despliegue básico del modelo
    └── model_monitoring.py          # Monitoreo de drift y pronósticos actuales
Base_de_datos.csv                    # Dataset principal del proyecto
requirements.txt                     # Dependencias del proyecto
```

## Configuración del entorno

Ejecutar estos comandos desde la carpeta raíz del proyecto:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Despliegue del modelo con FastAPI

El archivo `mlops_pipeline/src/model_deploy.py` expone el modelo en una API REST.

Internamente, este módulo hace cuatro cosas:

1. Reentrena y serializa el mejor modelo si todavía no existe un artefacto desplegable.
2. Carga el artefacto `data/best_model_artifact.pkl` cuando la API inicia.
3. Recibe registros por JSON o CSV para inferencia batch.
4. Aplica limpieza, creación de features, transformación del preprocesador y luego predicción.

Endpoints principales:

- `GET /health`: estado del servicio y nombre del modelo cargado.
- `POST /predict`: predicción batch por JSON.
- `POST /predict-csv`: predicción batch cargando un CSV.

Ejecutar API en local:

```bash
uvicorn mlops_pipeline.src.model_deploy:app --host 0.0.0.0 --port 8000 --reload
```

Al iniciar, la API valida si existe el artefacto `data/best_model_artifact.pkl`.
Si no existe, lo entrena y lo crea automáticamente usando el mejor modelo de `data/model_summary.csv`.

### Qué ocurre cuando se llama a `/predict` o `/predict-csv`

- La API recibe registros nuevos.
- Esos registros pasan por `clean_data()` y `create_features()` de [mlops_pipeline/src/ft_engineering.py](mlops_pipeline/src/ft_engineering.py).
- Luego se aplica `preprocessor.transform(...)` usando el preprocesador ya ajustado en entrenamiento.
- Finalmente el modelo devuelve:
  - la clase predicha (`predictions`)
  - la probabilidad de pago (`probabilities`)

Importante: en inferencia no se vuelve a hacer `train_test_split`, no se reentrena el modelo y no se vuelve a ajustar el preprocesador. Solo se reutiliza lo aprendido en entrenamiento.

Ejemplo de request a `/predict`:

```json
{
  "records": [
    {
      "salario_cliente": 1500000,
      "cuota_pactada": 250000,
      "total_otros_prestamos": 400000,
      "tipo_laboral": "EMPLEADO",
      "tendencia_ingresos": "Estable"
    }
  ]
}
```

## App de visualización con Streamlit

La app de Streamlit del proyecto ya está implementada en `mlops_pipeline/src/model_monitoring.py`.

Ejecutar dashboard en local:

```bash
streamlit run mlops_pipeline/src/model_monitoring.py
```

La app permite:

- Monitorear drift entre histórico y actual.
- Ver alertas automáticas de riesgo (Low/Medium/High).
- Explorar distribuciones y evolución temporal del PSI.
- Revisar muestra con pronósticos del modelo.

## Docker

Se incluye `Dockerfile` y `.dockerignore` en la raíz del proyecto.

El objetivo de Docker en este avance es empaquetar la API con:

- el código fuente
- las dependencias de `requirements.txt`
- el servidor Uvicorn
- una forma reproducible de ejecución en cualquier entorno


Construir imagen:

```bash
docker build -t credito-ml-api .
```

Ejecutar contenedor:

```bash
docker run -p 8000:8000 credito-ml-api
```

Luego la documentación interactiva queda disponible en:

- http://localhost:8000/docs


Después de abrirse Streamlit, el dashboard mostrará:

- controles para ajustar la porción histórica y el tamaño de la muestra actual
- alertas generales de drift
- tabla de métricas por variable
- comparación de distribuciones histórico vs actual
- evolución temporal del PSI
- tabla de pronóstico actual del modelo

## Principales hallazgos del ejercicio

1. **Monitoreo de drift**: el modelo no se evalúa solo por su performance histórica, sino también por cómo cambian las variables de entrada en el presente.
2. **División por fecha**: el proyecto usa `fecha_prestamo` para separar el dataset en `referencia` y `actual`, lo que permite comparar periodos.
3. **Control de leakage**: la variable `puntaje` fue eliminada porque sesgaba completamente el modelo y aportaba información posterior o demasiado cercana al target.
4. **Métricas usadas**:
   - `KS` para detectar cambios en variables numéricas.
   - `PSI` para medir cuánto cambia la distribución de una variable.
   - `JS` para comparar distribuciones más suavemente.
   - `Chi-cuadrado` para validar cambios en variables categóricas.
5. **Predicciones actuales**: la app muestra `prob_pago` y `pred_label` para entender si el modelo está fallando hoy.
6. **Diagnóstico de errores**: si la tabla de pronóstico muestra muchas filas donde `pred_label` no coincide con `Pago_atiempo`, el modelo puede estar degradado.

## Recomendaciones generales

- Si el dashboard detecta muchas variables con `High` de drift, revisar la calidad y la fuente de los datos actuales.
- Si el modelo falla en la muestra actual, evaluar retraining con datos más recientes.
- Usar la tabla de pronósticos para identificar ejemplos concretos donde el modelo se equivoca.

## Nota sobre el dataset

El archivo principal es `Base_de_datos.csv`. El proyecto usa este archivo como origen para preparar los datos y construir el monitoreo.

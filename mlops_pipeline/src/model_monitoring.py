# src/model_monitoring.py

# librerías básicas
import os
import sys
import numpy as np
import pandas as pd
# aquí importamos la librería para crear la aplicación 'Streamlit'
import streamlit as st
st.set_page_config(page_title="Monitoreo del Modelo", layout="wide")
# aquí importamos librerías de visualización
import matplotlib.pyplot as plt
import seaborn as sns
# librerías estadísticas para calcular las métricas de drift
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp
# librerías de sklearn
from sklearn.linear_model import LogisticRegression


# importamos funciones de preprocesamiento del proyecto
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from ft_engineering import build_preprocessing_pipeline, clean_data, create_features

##########################################
# 1. Configuración
##########################################

# ruta al dataset original
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, "Base_de_datos.csv")
# nombre de la variable objetivo
TARGET = "Pago_atiempo"
RANDOM_STATE = 42

##########################################
# 2. Cargar el dataset y prepararlo
##########################################

@st.cache_data
def load_data():
    # 2.1 cargamos el dataset con la fecha del préstamo
    df = pd.read_csv(DATASET_PATH, parse_dates=["fecha_prestamo"])
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"], errors="coerce")

    # 2.2 limpiamos y creamos los features con las funciones del proyecto
    df_clean = clean_data(df.copy())
    df_clean = create_features(df_clean)

    # la función clean_data elimina fecha_prestamo; la reincorporamos para el split temporal
    if "fecha_prestamo" not in df_clean.columns:
        df_clean["fecha_prestamo"] = df["fecha_prestamo"].values

    # 2.3 retornamos el dataset completo ordenado por fecha
    return df_clean.sort_values("fecha_prestamo").reset_index(drop=True)


# llamamos a la función y guardamos el dataset completo
df = load_data()

##########################################
# 3. Funciones para calcular métricas de drift
##########################################

# helper: normaliza columnas categóricas para que no fallen las métricas
def normalize_categorical(series):
    if isinstance(series.dtype, pd.CategoricalDtype):
        # añadimos la categoría antes de usarla en fillna
        series = series.cat.add_categories(["[missing]"])
    return series.fillna("[missing]").astype(object)


# PSI (Population Stability Index) para variables numéricas
# mide cuánto cambió la distribución de una variable entre dos períodos
def calculate_psi(expected, actual, bins=10):
    expected = expected.dropna().astype(float)
    actual = actual.dropna().astype(float)
    if expected.empty or actual.empty:
        return np.nan
    expected_counts, bin_edges = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    exp_pct = np.where(expected_counts == 0, 1e-8, expected_counts / len(expected))
    act_pct = np.where(actual_counts == 0, 1e-8, actual_counts / len(actual))
    return float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))


# PSI para variables categóricas
def calculate_psi_categorical(expected, actual):
    exp_counts = normalize_categorical(expected).value_counts(normalize=True)
    act_counts = normalize_categorical(actual).value_counts(normalize=True)
    all_cats = exp_counts.index.union(act_counts.index)
    exp_pct = exp_counts.reindex(all_cats, fill_value=1e-8)
    act_pct = act_counts.reindex(all_cats, fill_value=1e-8)
    return float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))


# KS (Kolmogorov-Smirnov) para variables numéricas
# prueba estadística que compara las distribuciones de dos muestras
def calculate_ks(expected, actual):
    result = ks_2samp(expected.dropna(), actual.dropna())
    return float(result.statistic), float(result.pvalue)


# Jensen-Shannon para variables numéricas
# mide la similitud entre dos distribuciones (0 = iguales, 1 = muy diferentes)
def calculate_js(expected, actual, bins=10):
    expected = expected.dropna().astype(float)
    actual = actual.dropna().astype(float)
    if expected.empty or actual.empty:
        return np.nan
    exp_hist, bin_edges = np.histogram(expected, bins=bins, density=True)
    act_hist, _ = np.histogram(actual, bins=bin_edges, density=True)
    exp_prob = np.where(exp_hist == 0, 1e-8, exp_hist / exp_hist.sum())
    act_prob = np.where(act_hist == 0, 1e-8, act_hist / act_hist.sum())
    return float(jensenshannon(exp_prob, act_prob, base=2) ** 2)


# Jensen-Shannon para variables categóricas
def calculate_js_categorical(expected, actual):
    exp_counts = normalize_categorical(expected).value_counts(normalize=True)
    act_counts = normalize_categorical(actual).value_counts(normalize=True)
    all_cats = exp_counts.index.union(act_counts.index)
    p = exp_counts.reindex(all_cats, fill_value=1e-8)
    q = act_counts.reindex(all_cats, fill_value=1e-8)
    return float(jensenshannon(p, q, base=2) ** 2)


# Chi-cuadrado para variables categóricas
# prueba si las frecuencias de las categorías cambiaron significativamente
def calculate_chi2(expected, actual):
    exp_counts = normalize_categorical(expected).value_counts().astype(float)
    act_counts = normalize_categorical(actual).value_counts().astype(float)
    all_cats = exp_counts.index.union(act_counts.index)
    exp_vals = exp_counts.reindex(all_cats, fill_value=1e-8)
    act_vals = act_counts.reindex(all_cats, fill_value=1e-8)
    if len(all_cats) < 2:
        return np.nan, np.nan
    chi2_stat, p_value, _, _ = chi2_contingency([exp_vals.values, act_vals.values])
    return float(chi2_stat), float(p_value)


# clasifica el nivel de riesgo de drift: Low / Medium / High
def risk_level(feature_type, psi, js, ks_pvalue=np.nan, chi2_pvalue=np.nan):
    if feature_type == "numeric":
        ks_flag = (not np.isnan(ks_pvalue)) and ks_pvalue < 0.05
        if psi >= 0.25 or js >= 0.2 or ks_flag:
            return "High"
        if psi >= 0.1 or js >= 0.1:
            return "Medium"
        return "Low"
    # categórica
    chi2_flag = (not np.isnan(chi2_pvalue)) and chi2_pvalue < 0.05
    if chi2_flag or js >= 0.2:
        return "High"
    if js >= 0.1:
        return "Medium"
    return "Low"


# calcula todas las métricas de drift para cada variable del dataset
def compute_drift_metrics(reference, current):
    # variables numéricas (excluimos el target)
    numeric_cols = (
        reference.select_dtypes(include=[np.number])
        .columns.drop(TARGET, errors="ignore")
        .tolist()
    )
    # variables categóricas (excluimos el target y la fecha)
    categorical_cols = [
        col
        for col in reference.select_dtypes(include=["category", "object"]).columns
        if col not in (TARGET, "fecha_prestamo")
    ]
    rows = []

    for col in numeric_cols:
        ks_stat, ks_pval = calculate_ks(reference[col], current[col])
        psi = calculate_psi(reference[col], current[col])
        js = calculate_js(reference[col], current[col])
        rows.append({
            "Variable": col,
            "Tipo": "Numérica",
            "KS stat": round(ks_stat, 4),
            "KS p-value": round(ks_pval, 4),
            "PSI": round(psi, 4) if not np.isnan(psi) else np.nan,
            "JS": round(js, 4) if not np.isnan(js) else np.nan,
            "Chi2 stat": np.nan,
            "Chi2 p-value": np.nan,
            "Riesgo": risk_level("numeric", psi, js, ks_pvalue=ks_pval),
        })

    for col in categorical_cols:
        chi2_stat, chi2_pval = calculate_chi2(reference[col], current[col])
        psi = calculate_psi_categorical(reference[col], current[col])
        js = calculate_js_categorical(reference[col], current[col])
        rows.append({
            "Variable": col,
            "Tipo": "Categórica",
            "KS stat": np.nan,
            "KS p-value": np.nan,
            "PSI": round(psi, 4),
            "JS": round(js, 4),
            "Chi2 stat": round(chi2_stat, 4) if not np.isnan(chi2_stat) else np.nan,
            "Chi2 p-value": round(chi2_pval, 4) if not np.isnan(chi2_pval) else np.nan,
            "Riesgo": risk_level("categorical", psi, js, chi2_pvalue=chi2_pval),
        })

    return pd.DataFrame(rows)

##########################################
# 4. Modelo de pronóstico
##########################################

# entrenamos un modelo de regresión logística SOLO con los datos históricos (referencia)
# así podemos ver si el modelo sigue funcionando bien sobre los datos actuales
def build_monitoring_model(reference):
    X = reference.drop(columns=[TARGET, "fecha_prestamo"], errors="ignore")
    y = reference[TARGET]
    preprocessor, *_ = build_preprocessing_pipeline(X)
    X_proc = preprocessor.fit_transform(X)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_proc, y)
    return model, preprocessor


# aplicamos el modelo entrenado sobre los datos actuales para ver cómo se comporta
def predict_current_data(current, model, preprocessor):
    X = current.drop(columns=[TARGET, "fecha_prestamo"], errors="ignore")
    X_proc = preprocessor.transform(X)
    current = current.copy()
    current["prob_pago"] = model.predict_proba(X_proc)[:, 1]
    current["pred_label"] = model.predict(X_proc)
    return current

##########################################
# 5. Interfaz de la aplicación Streamlit
##########################################

st.title("📊 Monitor de Data Drift — Modelo de Crédito")
st.markdown(
    "Compara las distribuciones históricas vs actuales de cada variable, "
    "calcula métricas de drift y entrega alertas automáticas sobre el estado del modelo."
)

# ── controles del sidebar ──
st.sidebar.header("⚙️ Configuración")
# el slider permite cambiar qué porción del dataset se usa como histórico
ref_ratio = st.sidebar.slider("Porción histórica (referencia)", 0.3, 0.9, 0.7, step=0.05)
sample_size = st.sidebar.slider("Registros actuales a mostrar", 50, 1000, 200, step=50)
all_features = sorted([c for c in df.columns if c not in (TARGET, "fecha_prestamo")])
selected_feature = st.sidebar.selectbox("Variable a explorar", all_features)
freq_label = st.sidebar.selectbox("Frecuencia temporal", ["Mensual (M)", "Semanal (W)", "Trimestral (Q)"])
drift_freq = freq_label.split("(")[1].rstrip(")")

# dividimos el dataset en referencia (histórico) y actual según el slider
cutoff_date = df["fecha_prestamo"].quantile(ref_ratio)
reference_df = df[df["fecha_prestamo"] <= cutoff_date].copy()
current_df   = df[df["fecha_prestamo"] > cutoff_date].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Fecha de corte:** {cutoff_date.strftime('%Y-%m-%d')}")
st.sidebar.markdown(f"**Histórico:** {len(reference_df):,} registros")
st.sidebar.markdown(f"**Actual:** {len(current_df):,} registros")

if current_df.empty:
    st.warning("No hay datos actuales con la configuración actual. Reduce la porción histórica.")
    st.stop()

# entrenamos el modelo y generamos pronósticos sobre los datos actuales
model, preprocessor = build_monitoring_model(reference_df)
current_predicted = predict_current_data(current_df.head(sample_size), model, preprocessor)

# calculamos las métricas de drift para todas las variables
metrics_df = compute_drift_metrics(reference_df, current_df)

# determinamos el nivel de alerta general según los resultados
high   = int((metrics_df["Riesgo"] == "High").sum())
medium = int((metrics_df["Riesgo"] == "Medium").sum())
if high > 0:
    alert_title  = "🔴 Riesgo alto detectado"
    alert_detail = f"{high} variable(s) con drift crítico"
elif medium > 0:
    alert_title  = "🟡 Riesgo moderado"
    alert_detail = f"{medium} variable(s) con drift relevante"
else:
    alert_title  = "🟢 Sin señales de drift crítico"
    alert_detail = "Todas las variables se mantienen estables"

####################################
# 5.1 Sección: Estado general del drift
####################################
st.subheader("🚦 Estado general del drift")
st.markdown(f"### {alert_title}")
st.caption(alert_detail)

# mostramos la recomendación según el nivel de riesgo detectado
if (metrics_df["Riesgo"] == "High").any():
    high_vars = metrics_df[metrics_df["Riesgo"] == "High"]["Variable"].tolist()
    st.error(
        f"**Acción recomendada: retraining del modelo.**\n\n"
        f"Variables con drift crítico: {', '.join(high_vars)}.\n\n"
        "Si múltiples variables presentan drift alto, el modelo puede estar degradado."
    )
elif (metrics_df["Riesgo"] == "Medium").any():
    med_vars = metrics_df[metrics_df["Riesgo"] == "Medium"]["Variable"].tolist()
    st.warning(
        f"**Monitoreo reforzado recomendado.**\n\n"
        f"Variables en seguimiento: {', '.join(med_vars)}.\n\n"
        "Controlar en los próximos ciclos antes de decidir retraining."
    )
else:
    st.success(
        "**El modelo puede mantenerse activo.** "
        "No se detectan cambios significativos en las distribuciones de entrada."
    )

####################################
# 5.2 Sección: Tabla de métricas por variable
####################################
st.markdown("---")
st.subheader("📋 Tabla de métricas por variable")

# coloreamos la columna Riesgo según el nivel
def color_risk(val):
    if val == "High":
        return "background-color: #f8d7da; color: #721c24;"
    if val == "Medium":
        return "background-color: #fff3cd; color: #856404;"
    return "background-color: #d4edda; color: #155724;"

st.dataframe(
    metrics_df.style.applymap(color_risk, subset=["Riesgo"]),
    use_container_width=True,
)

####################################
# 5.3 Sección: Distribuciones histórico vs actual
####################################
st.markdown("---")
st.subheader(f"📈 Comparación de distribución — {selected_feature}")
col1, col2 = st.columns(2)

# gráfico de distribución según el tipo de variable
if selected_feature in reference_df.select_dtypes(include=[np.number]).columns:
    # variable numérica: histograma superpuesto
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(reference_df[selected_feature].dropna(), color="#2c7fb8", label="Histórico",
                 stat="density", bins=18, alpha=0.55, ax=ax)
    sns.histplot(current_df[selected_feature].dropna(), color="#dd571c", label="Actual",
                 stat="density", bins=18, alpha=0.55, ax=ax)
    ax.set_title(f"Distribución histórica vs actual — {selected_feature}")
    ax.set_ylabel("Densidad")
    ax.legend()
    plt.tight_layout()
    col1.pyplot(fig)
else:
    # variable categórica: barra comparativa
    ref_counts = normalize_categorical(reference_df[selected_feature]).value_counts(normalize=True)
    cur_counts = normalize_categorical(current_df[selected_feature]).value_counts(normalize=True)
    dist = pd.concat([ref_counts, cur_counts], axis=1, keys=["Histórico", "Actual"]).fillna(0)
    col1.bar_chart(dist.sort_values("Histórico", ascending=False))

# gráfico de evolución del PSI en el tiempo
df_trend = current_df.copy()
df_trend = df_trend.set_index("fecha_prestamo").sort_index()
is_numeric = selected_feature in reference_df.select_dtypes(include=[np.number]).columns
trend_rows = []
for period, group in df_trend.groupby(pd.Grouper(freq=drift_freq)):
    if group.shape[0] < 10:
        continue
    psi_val = (
        calculate_psi(reference_df[selected_feature], group[selected_feature])
        if is_numeric
        else calculate_psi_categorical(reference_df[selected_feature], group[selected_feature])
    )
    trend_rows.append({"Período": period.strftime("%Y-%m"), "PSI": round(psi_val, 4), "N": len(group)})
trend_df = pd.DataFrame(trend_rows)

if trend_df.empty:
    col2.info("No hay suficientes períodos con datos para mostrar la evolución temporal.")
else:
    col2.markdown("**Evolución del PSI en el tiempo**")
    col2.line_chart(trend_df.set_index("Período")["PSI"])
    col2.dataframe(trend_df, use_container_width=True)

####################################
# 5.4 Sección: Tabla de pronósticos del modelo
####################################
st.markdown("---")
st.subheader("🤖 Muestra actual con pronósticos del modelo")
st.caption("El modelo fue entrenado solo sobre datos históricos. Los pronósticos sobre datos actuales permiten detectar degradación.")

# mostramos las columnas clave: fecha, target real, probabilidad, predicción y la variable seleccionada
display_cols = ["fecha_prestamo", TARGET, "prob_pago", "pred_label", selected_feature]
to_show = [c for c in display_cols if c in current_predicted.columns]
st.dataframe(
    current_predicted[to_show]
    .sort_values(by="prob_pago", ascending=False)
    .reset_index(drop=True),
    use_container_width=True,
)

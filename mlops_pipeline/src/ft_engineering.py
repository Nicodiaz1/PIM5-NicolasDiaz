"""
ft_engineering.py
=================
Feature Engineering pipeline del proyecto de crédito.

Genera conjuntos de entrenamiento y evaluación listos para modelamiento.
Salidas:
    - data/train_features.csv
    - data/test_features.csv
    - data/train_labels.csv
    - data/test_labels.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "Base_de_datos.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "Pago_atiempo"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ──────────────────────────────────────────────
# 1. CARGA
# ──────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load_data] Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df


# ──────────────────────────────────────────────
# 2. LIMPIEZA Y AJUSTE DE TIPOS
# ──────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Eliminar columnas irrelevantes, con fuga o muy correlacionadas con el target
    cols_to_drop = [
        "fecha_prestamo", "tipo_credito", "creditos_sectorFinanciero",
        "promedio_ingresos_datacredito", "saldo_mora_codeudor", "saldo_mora",
        "creditos_sectorCooperativo", "creditos_sectorReal",
        "puntaje",  # post-decision: score calculado tras el pago.
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    print(f"[clean_data] Columnas eliminadas: {cols_to_drop}")

    # Normalizar tipo_laboral como variable categórica
    df["tipo_laboral"] = df["tipo_laboral"].astype(str).str.strip().str.upper()
    df["tipo_laboral"] = df["tipo_laboral"].astype("category")

    # ── Tendencia ingresos: limpiar valores corruptos, mantener solo categorías válidas ──
    # El CSV tiene ~22 filas con números cargados por error (ej: '8315', '1000000').
    # Se convierten a NaN para que el pipeline impute con la moda.
    if "tendencia_ingresos" in df.columns:
        categorias_validas = {"Creciente", "Decreciente", "Estable"}
        corruptos = (~df["tendencia_ingresos"].isin(categorias_validas) & df["tendencia_ingresos"].notna()).sum()
        df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
            df["tendencia_ingresos"].isin(categorias_validas), other=np.nan
        )
        nan_total = df["tendencia_ingresos"].isna().sum()
        print(f"[clean_data] tendencia_ingresos: {corruptos} corruptos -> NaN | NaN totales: {nan_total} (se imputarán con moda)")

    # ── Huella consulta: 0 = válido (sin consultas al buró), NO se convierte a NaN ──
    if "huella_consulta" in df.columns:
        print(f"[clean_data] huella_consulta: {df['huella_consulta'].isna().sum()} NaN | 0 es válido (0 consultas al buró)")

    # ── Tratar ceros como datos faltantes en saldos ──────────────────────────
    # Un saldo de 0 en el momento del préstamo indica dato ausente (no saldo real)
    # Se reemplaza por NaN para que el pipeline los impute con la mediana
    for col in ["saldo_total", "saldo_principal"]:
        if col in df.columns:
            zeros_found = (df[col] == 0).sum()
            df[col] = df[col].replace(0, np.nan)
            print(f"[clean_data] {col}: {zeros_found} ceros → NaN (se imputarán con mediana)")

    # ── Puntaje Datacredito: ceros y negativos indican sin historial ─────────
    # Se crea una bandera binaria y luego se reemplaza el valor inválido por NaN
    if "puntaje_datacredito" in df.columns:
        df["tiene_datacredito"] = (df["puntaje_datacredito"] > 0).astype(int)
        invalidos = (df["puntaje_datacredito"] <= 0).sum()
        df["puntaje_datacredito"] = df["puntaje_datacredito"].where(df["puntaje_datacredito"] > 0, np.nan)
        print(f"[clean_data] puntaje_datacredito: {invalidos} valores ≤0 → NaN | nueva col: 'tiene_datacredito'")

    print(f"[clean_data] Nulos por columna:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


# ──────────────────────────────────────────────
# 3. INGENIERÍA DE ATRIBUTOS DERIVADOS
# ──────────────────────────────────────────────
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea atributos derivados identificados en el EDA:
    - ratio_cuota_salario     : carga mensual de cuota relativa al salario
    - ratio_deuda_ingresos    : endeudamiento relativo al salario
    """
    eps = 1e-6  # evitar división por cero
    df["ratio_cuota_salario"] = df["cuota_pactada"] / (df["salario_cliente"] + eps)
    df["ratio_deuda_ingresos"] = df["total_otros_prestamos"] / (df["salario_cliente"] + eps)
    
    print("[create_features] Atributos derivados creados: ratio_cuota_salario, ratio_deuda_ingresos")
    return df


# ──────────────────────────────────────────────
# 4. SPLIT y PIPELINE DE TRANSFORMACIÓN
# ──────────────────────────────────────────────
def build_preprocessing_pipeline(df: pd.DataFrame):
    # Seleccionar variables numéricas, categóricas nominales y ordinales
    num_features = df.select_dtypes(include=[np.number]).columns.drop("Pago_atiempo", errors="ignore").tolist()
    # Ajusta estos nombres según las columnas que existen tras la limpieza
    cat_nominal_features = [col for col in ["tipo_laboral"] if col in df.columns]
    cat_ordinal_features = [col for col in ["tendencia_ingresos"] if col in df.columns]

    # Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    cat_ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # Orden lógico: perspectiva negativa → neutra → positiva
        ("ordinal", OrdinalEncoder(
            categories=[["Decreciente", "Estable", "Creciente"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat_nominal", cat_nominal_pipeline, cat_nominal_features),
        ("cat_ordinal", cat_ordinal_pipeline, cat_ordinal_features),
    ])
    return preprocessor, num_features, cat_nominal_features, cat_ordinal_features


def run_feature_engineering():
    # --- Carga y limpieza ---
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = create_features(df)

    # --- Separar X e y ---
    X = df.drop(columns=[TARGET])
    y = df[TARGET]


    # --- Split train / test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[split] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # --- Construcción del pipeline ---
    preprocessor, num_features, cat_nominal_features, cat_ordinal_features = build_preprocessing_pipeline(X_train)

    # Ajustar SOLO sobre entrenamiento (evitar data leakage)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Recuperar nombres de columnas después del encoding
    num_cols = num_features
    cat_nominal_cols = []
    if cat_nominal_features:
        cat_nominal_cols = preprocessor.named_transformers_["cat_nominal"].named_steps["onehot"].get_feature_names_out(cat_nominal_features).tolist()
    cat_ordinal_cols = cat_ordinal_features
    all_cols = num_cols + cat_nominal_cols + cat_ordinal_cols

    df_train_feat = pd.DataFrame(X_train_proc, columns=all_cols)
    df_test_feat = pd.DataFrame(X_test_proc, columns=all_cols)

    # --- Guardar resultados ---
    df_train_feat.to_csv(os.path.join(OUTPUT_DIR, "train_features.csv"), index=False)
    df_test_feat.to_csv(os.path.join(OUTPUT_DIR, "test_features.csv"), index=False)
    pd.DataFrame(y_train).reset_index(drop=True).to_csv(os.path.join(OUTPUT_DIR, "train_labels.csv"), index=False)
    pd.DataFrame(y_test).reset_index(drop=True).to_csv(os.path.join(OUTPUT_DIR, "test_labels.csv"), index=False)

    print(f"\n[output] Archivos guardados en: {OUTPUT_DIR}")
    print("  - train_features.csv")
    print("  - test_features.csv")
    print("  - train_labels.csv")
    print("  - test_labels.csv")

    return df_train_feat, df_test_feat, y_train.reset_index(drop=True), y_test.reset_index(drop=True), preprocessor


if __name__ == "__main__":
    run_feature_engineering()


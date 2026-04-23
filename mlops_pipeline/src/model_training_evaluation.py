"""
model_training_evaluation.py
============================
Entrenamiento y evaluación de modelos supervisados de clasificación.

Modelos entrenados:
    - Regresión Logística
    - Árbol de Decisión
    - Random Forest
    - XGBoost

Funciones reutilizables:
    - build_model(name, estimator, X_train, y_train, X_test, y_test)
    - summarize_classification(y_test, y_pred, model_name)

Salidas:
    - data/model_summary.csv
    - data/roc_curves.png
    - data/metrics_comparison.png
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, GridSearchCV

# Agregar raíz al path para importar ft_engineering
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ft_engineering import run_feature_engineering

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# ──────────────────────────────────────────────
# GRID SEARCH PARA RF Y XGB
# ──────────────────────────────────────────────
def grid_search_rf_xgb(X_train, y_train):
    """
    Realiza GridSearchCV para RandomForest y XGBoost.
    Retorna los objetos GridSearchCV ya ajustados.
    """
    print("\n[GridSearch] Buscando mejores hiperparámetros para Random Forest...")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced']
    }
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='roc_auc', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print(f"Mejores parámetros RF: {rf_grid.best_params_}")

    print("\n[GridSearch] Buscando mejores hiperparámetros para XGBoost...")
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
    }
    xgb = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
    xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring='roc_auc', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    print(f"Mejores parámetros XGB: {xgb_grid.best_params_}")

    return rf_grid, xgb_grid

# ──────────────────────────────────────────────
# FUNCIONES REUTILIZABLES
# ──────────────────────────────────────────────
def summarize_classification(y_true, y_pred, y_prob, model_name: str) -> dict:
    """
    Genera un resumen de métricas de clasificación para un modelo dado.
    Imprime el reporte completo y retorna un dict con las métricas principales.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n{'='*50}")
    print(f"  Modelo: {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Reporte detallado:")
    print(classification_report(y_true, y_pred, target_names=["No paga", "Paga"]))
    print(f"  Matriz de confusión:\n{confusion_matrix(y_true, y_pred)}")

    return {
        "Modelo": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
    }


def build_model(name: str, estimator, X_train, y_train, X_test, y_test) -> dict:
    """
    Entrena un estimador sklearn/xgboost, evalúa en test y retorna métricas.

    Parámetros:
        name      : nombre descriptivo del modelo
        estimator : instancia del modelo (sin entrenar)
        X_train, y_train : datos de entrenamiento
        X_test,  y_test  : datos de evaluación
    Retorna:
        dict con métricas y el estimador entrenado
    """
    print(f"\n[build_model] Entrenando: {name} ...")
    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_test)
    y_prob = estimator.predict_proba(X_test)[:, 1]

    metrics = summarize_classification(y_test, y_pred, y_prob, name)
    metrics["estimator"] = estimator
    metrics["y_prob"] = y_prob
    return metrics


# ──────────────────────────────────────────────
# GRÁFICOS COMPARATIVOS
# ──────────────────────────────────────────────
def plot_roc_curves(results: list, y_test, output_path: str):
    """Grafica las curvas ROC de todos los modelos en un solo gráfico."""
    plt.figure(figsize=(8, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        plt.plot(fpr, tpr, label=f"{r['Modelo']} (AUC={r['ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curvas ROC – Comparación de Modelos")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[plot] ROC curves guardado en: {output_path}")


def plot_metrics_comparison(summary_df: pd.DataFrame, output_path: str):
    """Gráfico de barras comparando métricas principales por modelo."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(summary_df))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, summary_df[metric], width, label=metric)

    ax.set_xlabel("Modelo")
    ax.set_ylabel("Score")
    ax.set_title("Comparación de Métricas por Modelo")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(summary_df["Modelo"], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[plot] Metrics comparison guardado en: {output_path}")


# ──────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────
def run_training():
    # --- 1. Obtener datos procesados del feature engineering ---
    print("=" * 60)
    print("PASO 1: Feature Engineering")
    print("=" * 60)
    X_train, X_test, y_train, y_test, _ = run_feature_engineering()

    # --- Chequeo de balance y valores únicos ---
    print("\n[Chequeo] Proporción de clases en y_train:")
    print(y_train.value_counts(normalize=True))
    print("\n[Chequeo] Proporción de clases en y_test:")
    print(y_test.value_counts(normalize=True))
    print("\n[Chequeo] Valores únicos en y_train:", y_train.unique())
    print("[Chequeo] Valores únicos en y_test:", y_test.unique())

    # --- Grid Search para optimización de hiperparámetros ---
    rf_grid, xgb_grid = grid_search_rf_xgb(X_train, y_train)

    # --- 2. Definir modelos a entrenar (usando los mejores estimadores para RF y XGB) ---
    models = [
        (
            "Regresión Logística",
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
        ),
        (
            "Árbol de Decisión",
            DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE, class_weight="balanced"),
        ),
        (
            "Random Forest (opt)",
            rf_grid.best_estimator_,
        ),
        (
            "XGBoost (opt)",
            XGBClassifier(
                **xgb_grid.best_params_,
                eval_metric='logloss',
                random_state=RANDOM_STATE,
                scale_pos_weight=float((y_train == 0).sum()) / float((y_train == 1).sum()),
            ),
        ),
    ]

    # --- 3. Entrenar y evaluar cada modelo ---
    print("\n" + "=" * 60)
    print("PASO 2: Entrenamiento y Evaluación")
    print("=" * 60)
    results = []
    for name, estimator in models:
        result = build_model(name, estimator, X_train, y_train, X_test, y_test)
        results.append(result)

    # --- Cross-validation para cada modelo ---
    print("\n[Cross-validation] Evaluando estabilidad de modelos en X_train...")
    for name, estimator in models:
        scores = cross_val_score(estimator, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"Modelo: {name}")
        print(f"  ROC-AUC por fold: {scores}")
        print(f"  Media ROC-AUC: {scores.mean():.4f}")
        print(f"  Desviación estándar ROC-AUC: {scores.std():.4f}\n")
    # --- 4. Tabla resumen ---
    summary_cols = ["Modelo", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    summary_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in results])
    summary_df = summary_df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("RESUMEN COMPARATIVO DE MODELOS")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(OUTPUT_DIR, "model_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[output] Tabla resumen guardada en: {summary_path}")

    # --- 5. Gráficos comparativos ---
    print("\n" + "=" * 60)
    print("PASO 3: Generando Gráficos Comparativos")
    print("=" * 60)
    plot_roc_curves(results, y_test, os.path.join(OUTPUT_DIR, "roc_curves.png"))
    plot_metrics_comparison(summary_df, os.path.join(OUTPUT_DIR, "metrics_comparison.png"))

    # --- 6. Mejor modelo ---
    best = summary_df.iloc[0]
    print(f"\nMejor modelo: {best['Modelo']} (ROC-AUC = {best['ROC-AUC']})")

    return summary_df, results


if __name__ == "__main__":
    run_training()


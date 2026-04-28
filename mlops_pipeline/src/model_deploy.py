"""
model_deploy.py
===============
Despliegue del mejor modelo como API con FastAPI.

Funciones principales:
1. Entrenar y guardar el mejor modelo en un artefacto serializado.
2. Cargar el artefacto para inferencia.
3. Exponer endpoints para predicción batch por JSON y CSV.
"""

import io
import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Agregar raíz al path para importar ft_engineering
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
	sys.path.insert(0, SRC_DIR)

from ft_engineering import (
	DATA_PATH,
	RANDOM_STATE,
	TARGET,
	clean_data,
	create_features,
	run_feature_engineering,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACT_PATH = os.path.join(DATA_DIR, "best_model_artifact.pkl")
MODEL_SUMMARY_PATH = os.path.join(DATA_DIR, "model_summary.csv")


class PredictRequest(BaseModel):
	records: list[dict[str, Any]] = Field(
		...,
		min_length=1,
		description="Lista de registros para inferencia batch",
	)


app = FastAPI(
	title="Credit Model API",
	description="API para predicción batch del modelo de pago a tiempo",
	version="1.0.0",
)


def load_raw_training_columns() -> list[str]:
	"""Obtiene las columnas de entrada esperadas en formato crudo (antes del pipeline)."""
	df = pd.read_csv(DATA_PATH)
	df = clean_data(df)
	df = create_features(df)
	feature_cols = [c for c in df.columns if c != TARGET]
	return feature_cols


def get_best_model_name_from_summary() -> str:
	"""Lee model_summary.csv y retorna el nombre del mejor modelo por ROC-AUC."""
	if not os.path.exists(MODEL_SUMMARY_PATH):
		return "Random Forest (opt)"

	summary = pd.read_csv(MODEL_SUMMARY_PATH)
	if "Modelo" not in summary.columns:
		return "Random Forest (opt)"

	return str(summary.iloc[0]["Modelo"])


def build_estimator(model_name: str, y_train: pd.Series):
	"""Instancia un estimador compatible con el nombre del modelo."""
	if "Random Forest" in model_name:
		return RandomForestClassifier(
			n_estimators=200,
			max_depth=8,
			min_samples_split=2,
			class_weight="balanced",
			random_state=RANDOM_STATE,
		)

	if "XGBoost" in model_name:
		pos = float((y_train == 1).sum())
		neg = float((y_train == 0).sum())
		scale_pos_weight = (neg / pos) if pos > 0 else 1.0
		return XGBClassifier(
			n_estimators=200,
			max_depth=5,
			learning_rate=0.1,
			eval_metric="logloss",
			random_state=RANDOM_STATE,
			scale_pos_weight=scale_pos_weight,
		)

	if "Regresión Logística" in model_name:
		return LogisticRegression(
			max_iter=1000,
			random_state=RANDOM_STATE,
			class_weight="balanced",
		)

	return DecisionTreeClassifier(
		max_depth=6,
		random_state=RANDOM_STATE,
		class_weight="balanced",
	)


def train_and_save_artifact(force_retrain: bool = False) -> dict[str, Any]:
	"""Entrena el mejor modelo y guarda un artefacto para inferencia."""
	if os.path.exists(ARTIFACT_PATH) and not force_retrain:
		return load_artifact()

	X_train_proc, _, y_train, _, preprocessor = run_feature_engineering()
	best_model_name = get_best_model_name_from_summary()
	estimator = build_estimator(best_model_name, y_train)
	estimator.fit(X_train_proc, y_train)

	artifact = {
		"model_name": best_model_name,
		"model": estimator,
		"preprocessor": preprocessor,
		"raw_feature_columns": load_raw_training_columns(),
		"target": TARGET,
	}

	os.makedirs(DATA_DIR, exist_ok=True)
	with open(ARTIFACT_PATH, "wb") as f:
		pickle.dump(artifact, f)

	return artifact


def load_artifact() -> dict[str, Any]:
	"""Carga el artefacto serializado del modelo."""
	if not os.path.exists(ARTIFACT_PATH):
		raise FileNotFoundError(
			f"No existe artefacto en {ARTIFACT_PATH}. Ejecuta train_and_save_artifact()."
		)

	with open(ARTIFACT_PATH, "rb") as f:
		artifact = pickle.load(f)
	return artifact


def _prepare_raw_input(df_raw: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
	"""
	Normaliza el input crudo para que sea compatible con el preprocesamiento entrenado.
	Completa columnas faltantes con NaN y elimina columnas extra.
	"""
	df = df_raw.copy()
	missing_cols = [c for c in expected_columns if c not in df.columns]
	for col in missing_cols:
		df[col] = np.nan

	df = df[expected_columns]
	df = clean_data(df)
	df = create_features(df)

	# Tras clean/create pueden haberse creado o eliminado columnas.
	# Reindex asegura orden y completa faltantes con NaN.
	df = df.reindex(columns=expected_columns, fill_value=np.nan)
	return df


def predict_batch(df_records: pd.DataFrame) -> dict[str, Any]:
	"""Realiza predicción batch para un DataFrame de registros de entrada."""
	artifact = load_artifact()
	expected_columns = artifact["raw_feature_columns"]
	preprocessor = artifact["preprocessor"]
	model = artifact["model"]

	df_in = _prepare_raw_input(df_records, expected_columns)
	X_proc = preprocessor.transform(df_in)

	pred_label = model.predict(X_proc)
	pred_prob = model.predict_proba(X_proc)[:, 1]

	return {
		"model_name": artifact["model_name"],
		"total_records": int(len(df_in)),
		"predictions": [int(v) for v in pred_label],
		"probabilities": [float(round(v, 6)) for v in pred_prob],
	}


@app.on_event("startup")
def bootstrap_model() -> None:
	"""Garantiza que exista un artefacto al iniciar la API."""
	try:
		load_artifact()
	except FileNotFoundError:
		train_and_save_artifact(force_retrain=False)


@app.get("/")
def root() -> dict[str, str]:
	return {
		"message": "Credit Model API activa",
		"docs": "/docs",
	}


@app.get("/health")
def health() -> dict[str, Any]:
	artifact = load_artifact()
	return {
		"status": "ok",
		"model_name": artifact["model_name"],
		"artifact_path": ARTIFACT_PATH,
	}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
	try:
		df = pd.DataFrame(request.records)
		if df.empty:
			raise HTTPException(status_code=400, detail="records no puede estar vacío")
		return predict_batch(df)
	except HTTPException:
		raise
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"Error en predicción: {exc}") from exc


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)) -> dict[str, Any]:
	try:
		content = await file.read()
		df = pd.read_csv(io.BytesIO(content))
		if df.empty:
			raise HTTPException(status_code=400, detail="CSV sin registros")
		return predict_batch(df)
	except HTTPException:
		raise
	except Exception as exc:
		raise HTTPException(status_code=400, detail=f"CSV inválido: {exc}") from exc


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000, reload=False)

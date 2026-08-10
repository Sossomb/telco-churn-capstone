"""
Tests du pipeline de churn — Mission 5.
Lancer avec : pytest tests/ -v (depuis la racine du projet)
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "pipeline.joblib"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Telco-Customer-Churn.csv"

EXPECTED_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


@pytest.fixture(scope="module")
def pipeline():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def sample_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    X = df.drop(columns=["customerID", "Churn"]).head(20)
    y = (df["Churn"] == "Yes").astype(int).head(20)
    return X, y


def test_output_shape(pipeline, sample_data):
    """La sortie a la bonne forme : une prédiction par ligne d'entrée."""
    X, _ = sample_data
    preds = pipeline.predict(X)
    assert preds.shape == (len(X),)


def test_probabilities_in_valid_range(pipeline, sample_data):
    """Les probabilités prédites sont bien dans [0, 1]."""
    X, _ = sample_data
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_handles_missing_values(pipeline, sample_data):
    """Le pipeline gère les valeurs manquantes sans planter (ex. TotalCharges vide)."""
    X, _ = sample_data
    X_missing = X.copy()
    X_missing.loc[X_missing.index[0], "TotalCharges"] = np.nan
    X_missing.loc[X_missing.index[1], "tenure"] = 0
    preds = pipeline.predict(X_missing)
    assert preds.shape == (len(X_missing),)
    assert not np.any(np.isnan(preds))


def test_expected_features_present(sample_data):
    """Toutes les features attendues par le pipeline sont présentes dans les données."""
    X, _ = sample_data
    for col in EXPECTED_FEATURES:
        assert col in X.columns, f"Feature manquante : {col}"


def test_performance_on_reference_set(pipeline, sample_data):
    """La performance sur un mini-jeu de référence reste dans une plage raisonnable."""
    X, y = sample_data
    proba = pipeline.predict_proba(X)[:, 1]
    preds = (proba >= 0.06).astype(int)  # seuil métier de la Mission 4
    accuracy = (preds == y.values).mean()
    assert accuracy >= 0.3  # seuil large : juste pour détecter une régression grave


def test_reload_gives_identical_predictions(sample_data):
    """Recharger le pipeline depuis le disque donne des prédictions strictement identiques."""
    X, _ = sample_data
    pipeline_1 = joblib.load(MODEL_PATH)
    pipeline_2 = joblib.load(MODEL_PATH)
    proba_1 = pipeline_1.predict_proba(X)
    proba_2 = pipeline_2.predict_proba(X)
    assert np.array_equal(proba_1, proba_2)
"""
Pipeline de préparation des données — Telco Customer Churn.
Réutilisé par toutes les missions (3, 4, 5) pour garantir un traitement
strictement identique et éviter toute fuite de données.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


class TotalChargesImputer(BaseEstimator, TransformerMixin):
    """Règle métier : 0€ pour un client nouveau (tenure=0), sinon
    tenure x MonthlyCharges. Ligne par ligne -> pas de risque de fuite."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        missing = X["TotalCharges"].isna()
        is_new = missing & (X["tenure"] == 0)
        is_existing = missing & (X["tenure"] > 0)
        X.loc[is_new, "TotalCharges"] = 0
        X.loc[is_existing, "TotalCharges"] = (
            X.loc[is_existing, "tenure"] * X.loc[is_existing, "MonthlyCharges"]
        )
        return X


def build_preprocessor():
    """ColumnTransformer : imputation médiane+scaling (numérique),
    imputation mode+OneHot (catégoriel). Rien n'est appris ici, juste
    la recette — l'apprentissage se fait au .fit() sur le train."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUM_COLS),
        ("cat", categorical_pipeline, CAT_COLS),
    ])


def build_pipeline(model):
    """Assemble le pipeline complet : règle métier -> ColumnTransformer -> modèle."""
    return Pipeline([
        ("total_charges_fix", TotalChargesImputer()),
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])


def load_data(path="../data/Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    X = df.drop(columns=["customerID", "Churn"])
    y = (df["Churn"] == "Yes").astype(int)
    return X, y
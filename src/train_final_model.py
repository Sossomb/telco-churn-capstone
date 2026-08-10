"""
Entraîne et sérialise le pipeline final (issu du tuning Optuna, Mission 4).
Hyperparamètres figés pour la reproductibilité — pas besoin de relancer
Optuna à chaque déploiement.
"""
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import recall_score, precision_score, confusion_matrix

from preprocessing import build_pipeline, load_data

RANDOM_STATE = 42
BEST_PARAMS = dict(
    C=2.6780773845790105,
    penalty="l2",
    class_weight="balanced",
    max_iter=2944,
    tol=0.000607544432471206,
    solver="saga",
    random_state=RANDOM_STATE,
)
DECISION_THRESHOLD = 0.06
FN_COST = 890
FP_COST = 50

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def main():
    X, y = load_data(path=str(DATA_PATH))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1. Évaluation honnête sur le test (avant le refit final)
    eval_pipeline = CalibratedClassifierCV(
        estimator=build_pipeline(LogisticRegression(**BEST_PARAMS)),
        method="sigmoid", cv=cv,
    )
    eval_pipeline.fit(X_train, y_train)
    y_proba_test = eval_pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    metrics = {
        "recall_test": round(recall_score(y_test, y_pred_test), 3),
        "precision_test": round(precision_score(y_test, y_pred_test), 3),
        "cost_test_eur": int(FN_COST * fn + FP_COST * fp),
        "confusion_matrix_test": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }
    print("Performance sur le test (avant refit final) :", metrics)

    # 2. Refit final sur TOUTES les données disponibles (train + test)
    final_pipeline = CalibratedClassifierCV(
        estimator=build_pipeline(LogisticRegression(**BEST_PARAMS)),
        method="sigmoid", cv=cv,
    )
    final_pipeline.fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final_pipeline, MODEL_DIR / "pipeline.joblib")

    metadata = {
        "model_version": "1.0.0",
        "decision_threshold": DECISION_THRESHOLD,
        "expected_features": list(X.columns),
        "best_hyperparameters": {k: v for k, v in BEST_PARAMS.items()},
        "validation_performance": metrics,
        "trained_on_n_samples": len(X),
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    print("Pipeline sauvegardé dans model/pipeline.joblib")
    print("Métadonnées sauvegardées dans model/model_metadata.json")


if __name__ == "__main__":
    main()
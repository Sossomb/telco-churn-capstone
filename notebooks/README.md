# Prédiction du churn client — Telco Customer Churn

Projet final (capstone) — Module 7, Supervised Learning, Master IA.
Prédit le risque de résiliation d'un client télécom pour prioriser les
campagnes de rétention.

## Problème métier

Un opérateur télécom veut identifier, parmi ses clients actifs, ceux
présentant un risque élevé de résiliation (churn), afin que le service
marketing puisse leur adresser en priorité une offre de rétention. Un
faux négatif (client qui part sans avoir été ciblé) coûte estimé à ~890€
(perte de revenu récurrent sur 1 an) ; un faux positif (offre envoyée à
tort) coûte ~50€ (remise + contact). Ce ratio de coûts (~18:1) oriente
tout le projet vers la maximisation du **rappel**, avec un seuil de
décision optimisé plutôt que le défaut de 0,5 (voir `rapport_capstone.md`,
Mission 0 et Mission 4).

## Données

Dataset **Telco Customer Churn** (IBM Sample / Kaggle `blastchar/telco-customer-churn`)
— 7043 clients, 20 features (démographie, services souscrits, type de
contrat, charges) + la cible `Churn` (Yes/No). Taux de churn : 26,5%.
Voir `data/download_data.sh` pour le script de téléchargement (nécessite
un compte Kaggle + token API).

## Modèle

Régression logistique, hyperparamètres optimisés par Optuna (60 essais,
6 hyperparamètres), calibrée (Platt scaling), avec un seuil de décision
métier de **0,06** (au lieu de 0,5) pour minimiser le coût métier total.

**Performance finale (jeu de test, jamais vu pendant l'entraînement) :**

| Métrique | Valeur |
|---|---|
| Rappel | 0,979 |
| Précision | 0,366 |
| Coût métier total | 38 770 € (vs 152 200 € au seuil 0,5 — réduction de 74,5%) |

Détail complet de la démarche (EDA, pipeline, comparaison de modèles,
tuning, SHAP, seuil) dans `rapport_capstone.md` et les notebooks
`notebooks/01_eda.ipynb` à `notebooks/04_tuning_calibration_shap.ipynb`.

## Structure du dépôt

\`\`\`
data/          Dataset + script de téléchargement
notebooks/     Notebooks d'analyse (EDA -> tuning/calibration/SHAP)
src/           Code réutilisable (pipeline de préparation, entraînement final)
model/         Pipeline sérialisé (pipeline.joblib) + métadonnées
tests/         Tests pytest
api/           API FastAPI
requirements.txt
\`\`\`

## Installation

\`\`\`bash
python -m venv venv
venv\\Scripts\\Activate.ps1        # Windows
pip install -r requirements.txt
\`\`\`

## Entraîner le modèle final

\`\`\`bash
python src/train_final_model.py
\`\`\`
Sauvegarde `model/pipeline.joblib` et `model/model_metadata.json`.

## Lancer les tests

\`\`\`bash
pytest tests/ -v
\`\`\`

## Lancer l'API

\`\`\`bash
uvicorn api.main:app --reload
\`\`\`

L'API écoute sur `http://127.0.0.1:8000`.

### Endpoints

- **`GET /health`** — statut de l'API + version du modèle.
- **`GET /model-info`** — features attendues, métadonnées, performance de validation.
- **`POST /predict`** — prédiction pour un client (voir `api/sample_request.json`).
  Réponse : `{"churn_prediction": 0|1, "churn_probability": float, "decision_threshold": float}`

## Documentation complémentaire

- `rapport_capstone.md` — rapport complet (6 missions + questions de réflexion).
- `MODEL_CARD.md` — model card (données d'entraînement, performance par sous-groupe, limites).
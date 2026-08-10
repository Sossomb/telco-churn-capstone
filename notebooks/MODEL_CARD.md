# Model Card — Prédiction du churn client (Telco)

Inspirée de Mitchell et al., *"Model Cards for Model Reporting"* (FAccT 2019).

## Détails du modèle

- **Type** : régression logistique (scikit-learn), calibrée (`CalibratedClassifierCV`, Platt scaling).
- **Version** : 1.0.0
- **Hyperparamètres** : optimisés par Optuna (60 essais, 6 hyperparamètres). `C=2,68`, `penalty=l2`, `class_weight=balanced`, `max_iter=2944`, `tol=0,0006`.
- **Seuil de décision** : 0,06 (optimisé sur les coûts métier, pas le défaut 0,5).
- **Développeuse** : Khady, Master IA, Module 7 (capstone).
- **Date** : 2026.

## Utilisation prévue

- **Usage principal** : scorer les clients actifs d'un opérateur télécom
  pour prioriser une campagne de rétention (ciblage marketing).
- **Utilisateurs prévus** : service marketing / rétention client.
- **Hors périmètre** : ce modèle ne doit pas être utilisé pour des
  décisions individuelles à fort impact (ex. refus de service), ni pour
  un usage hors du secteur télécom sans réentraînement.

## Données d'entraînement

- **Source** : Telco Customer Churn (IBM Sample / Kaggle `blastchar/telco-customer-churn`).
- **Volume** : 7043 clients, 20 features + cible `Churn`, taux de churn 26,5%.
- **Split** : 80% train (5634) / 20% test (1409), stratifié. Modèle final
  réentraîné sur 100% des données (7043) après évaluation sur le test.
- **Fenêtre temporelle** : non documentée dans le dataset source (limite connue).

## Performance

**Sur le jeu de test (1409 clients, avant réentraînement final) :**

| Métrique | Valeur |
|---|---|
| Rappel | 0,979 |
| Précision | 0,366 |
| Coût métier estimé | 38 770 € (vs 152 200 € au seuil par défaut 0,5) |

**Performance par sous-groupe (rappel, jeu de test) :**

| Sous-groupe | Rappel | n |
|---|---|---|
| Genre — Homme | 0,989 | 722 |
| Genre — Femme | 0,969 | 687 |
| Senior | 0,990 | 222 |
| Non-senior | 0,975 | 1187 |

Écart maximal observé entre sous-groupes : 2,1 points. Performance
homogène sur ces deux critères démographiques disponibles.

## Limites connues

- **Précision faible (36,6%)** : conséquence assumée du seuil très bas
  (0,06), qui génère beaucoup de fausses alertes (633/1409) — à mettre
  en balance avec la capacité opérationnelle réelle des équipes.
- **Churn "inattendu" mal détecté** : ~25% des churners en contrat long
  sont ratés (Mission 3, analyse d'erreurs).
- **Interprétabilité SHAP à nuancer** : colinéarité entre `tenure`,
  `MonthlyCharges`, `TotalCharges` (Mission 4).
- **Fenêtre de définition du churn non documentée** dans le dataset source.
- **Stationnarité non garantie** : à surveiller en production.
- **Analyse de fairness limitée** : seules 2 variables démographiques testées.

## Recommandations d'usage

- Réentraîner périodiquement (voir plan de monitoring).
- Ajuster le seuil selon la capacité opérationnelle réelle de l'équipe.
- Utiliser SHAP comme aide à la décision, pas comme justification automatique unique.
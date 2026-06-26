🏠 Prédiction de Prix Immobiliers — Stacking Ensemble
Prédiction du prix de vente de maisons aux USA à partir du dataset USA_Housing, avec feature engineering, suppression d'outliers sans data leakage, et optimisation d'un modèle Stacking Ensemble (Ridge + Random Forest + Gradient Boosting) via GridSearchCV.
🏗️ Architecture du modèle

Pipeline
│
├── Préprocesseur (ColumnTransformer)
│   └── Numérique : StandardScaler (toutes les features)
│
└── StackingRegressor
    ├── Base 1 : Ridge Regression
    ├── Base 2 : Random Forest Regressor
    ├── Base 3 : Gradient Boosting Regressor
    └── Méta   : Ridge Regression (final_estimator, alpha=1.0)

Le StackingRegressor entraîne les 3 modèles de base en cross-validation (cv=3), puis le méta-learner Ridge apprend à combiner leurs prédictions pour maximiser le R².
Le modèle de Stacking Ensemble atteint un R² de 0.9045 sur le jeu de test, signifiant qu'il explique plus de 90% de la variance des prix immobiliers. Avec un RMSE de 100 829 $ et un MAE de 81 609 $, les erreurs de prédiction restent cohérentes et maîtrisées. L'écart entre les performances train et test est quasi nul (0.0013), confirmant l'absence totale d'overfitting et une excellente capacité de généralisation du modèle sur de nouvelles données.

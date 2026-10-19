# Détection de fraude — PaySim

Pipeline Machine Learning de bout en bout pour détecter les transactions frauduleuses dans un système de paiement mobile, sur un dataset de 6,36 millions de transactions avec seulement 0,13 % de fraudes.

## Contexte

La fraude financière coûte des milliards chaque année, et la détection manuelle est impossible à l'échelle de millions de transactions par heure. Le système de règles existant de ce dataset (`isFlaggedFraud`) ne détectait que **16 fraudes sur 8 213 réelles**, soit un taux inférieur à 0,2 %.

## Solution

Un pipeline Scikit-Learn complet (normalisation + encodage + Régression Logistique à classes pondérées) classifie chaque transaction comme légitime ou frauduleuse, avec un objectif explicite : maximiser le **recall** plutôt que l'accuracy, car une fraude non détectée coûte bien plus cher qu'un faux positif à investiguer.

## Dataset

[PaySim Financial Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset) (Kaggle) — données synthétiques inspirées de transactions M-Pesa.

- 6 362 620 transactions, 11 colonnes brutes (6 utilisées pour la modélisation)
- Cible `isFraud` : seulement 8 213 fraudes (0,13 %)
- Aucune valeur manquante
- Fichier non inclus dans ce dépôt (≈490 Mo) — à télécharger depuis Kaggle et placer à la racine sous `AIML Dataset.csv`

## Analyse exploratoire — insights clés

- Les fraudes se concentrent **exclusivement** sur les types `TRANSFER` et `CASH_OUT` ; aucune fraude sur `PAYMENT`, `CASH_IN` ou `DEBIT`
- **1 188 074** transactions présentent un solde émetteur tombant à zéro après un `TRANSFER`/`CASH_OUT` — signal fort de vidage de compte
- Montants très asymétriques (de 0 à ~92M), nécessitant une normalisation
- Faible corrélation linéaire directe entre `isFraud` et les variables numériques (max 0,077) — justifie l'usage d'un modèle capable de capter les interactions

## Nettoyage & prétraitement

- Suppression de `step` (pas de signal prédictif sans encodage cyclique), `nameOrig`/`nameDest` (trop cardinaux : 6,35M et 2,72M valeurs distinctes, risque de surapprentissage), `isFlaggedFraud` (ne capte que 0,19 % des fraudes, risque de fuite de données)
- `ColumnTransformer` : `StandardScaler` sur les 5 variables numériques, `OneHotEncoder(drop="first")` sur `type`
- Le tout encapsulé dans un `Pipeline` Scikit-Learn pour garantir des transformations identiques entre entraînement et inférence

## Modélisation

- **Régression Logistique**, `class_weight="balanced"` pour compenser le déséquilibre 0,13 % / 99,87 %
- Split stratifié 70/30 (`stratify=y`), `max_iter=1000`

## Résultats

| Métrique | Valeur |
|---|---|
| Accuracy globale | 94,6 % |
| Recall (fraude) | **94 %** — 2 313 fraudes détectées sur 2 464 |
| Precision (fraude) | 2 % |
| F1-score (fraude) | 0,04 |
| Faux négatifs | 151 fraudes passées inaperçues |
| Faux positifs | 102 403 transactions légitimes signalées |

**Lecture métier :** face à un système existant détectant 0,19 % des fraudes, le modèle en détecte 94 % — une amélioration d'environ 500x sur le taux de détection réel. Le coût (102 403 faux positifs à investiguer) est le compromis classique detection/bruit, acceptable au regard des enjeux financiers.

## Déploiement

Le pipeline entraîné est sérialisé (`joblib`) dans `fraud_detection_pipeline.pkl` et chargé par une application **Streamlit** (`fraud_detection.py`) permettant à un analyste de saisir les paramètres d'une transaction et d'obtenir une prédiction en temps réel, sans recoder aucune transformation.

```bash
pip install streamlit pandas scikit-learn joblib
streamlit run fraud_detection.py
```

## Défis techniques & apprentissages

- **Déséquilibre de classes extrême** : un modèle naïf prédisant toujours "non-fraude" atteindrait 99,87 % d'accuracy sans détecter une seule fraude — l'accuracy seule est trompeuse ici, d'où la priorité donnée au recall
- **Variables à très haute cardinalité** (`nameOrig`, `nameDest`) : encodage direct impossible et inutile, chaque utilisateur frauduleux n'apparaissant qu'une fois
- **Distribution très asymétrique** des montants : normalisation indispensable pour la convergence du modèle linéaire
- **Volume** : entraîner sur 6,3M lignes avec `max_iter=1000` demande du temps de calcul et de la mémoire

## Stack technique

Python 3.13, Pandas, NumPy, Scikit-learn, Joblib, Matplotlib, Seaborn, Streamlit, Jupyter Notebook

## Pistes d'amélioration

- Modèles plus puissants (Random Forest, XGBoost, LightGBM) pour améliorer la precision (2 % actuellement) en gardant un recall élevé
- Rééchantillonnage (SMOTE, undersampling) en complément ou remplacement de `class_weight="balanced"`
- Seuil de décision optimisé via courbe ROC / Precision-Recall plutôt qu'un seuil fixe à 0,5
- Feature engineering avancé sur `nameOrig`/`nameDest` (fréquence, centralité dans le réseau de transactions) plutôt que suppression
- Monitoring du data drift en production (Evidently AI, WhyLogs)
- Déploiement conteneurisé (Docker) avec une API REST (FastAPI) pour supporter une charge de production
- Explicabilité (SHAP values) pour les équipes conformité

## Auteur

[Mabibè Bankati](https://github.com/Bankati)

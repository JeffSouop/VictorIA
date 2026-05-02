# VictorIA ⚽🧠

**Prédiction de matchs sportifs par intelligence artificielle**

VictorIA est un système de prédiction ML qui analyse des matchs de football et fournit des pronostics précis, explicables et accompagnés d'une analyse technique approfondie.

---

## Fonctionnalités

| Feature | Détail |
|---|---|
| 🧠 **3 modèles ML** | XGBoost · Random Forest · Réseau de neurones |
| 🔍 **Explainability SHAP** | Waterfall chart · Top facteurs décisifs |
| 📊 **27 features** | Forme, Elo proxy, H2H, buts, avantage domicile |
| 🏠 **Avantage domicile** | Intégré dans le calcul des probabilités |
| 📡 **Données réelles** | football-data.org (optionnel, clé gratuite) |
| ⚽ **Score exact** | Projection scoreline + top scénarios probables |
| ⏱️ **Stats temps réel** | Δ forme/attaque/défense, edge H2H, momentum live |
| 🔬 **Mode démo** | Données synthétiques reproductibles sans API |
| 🌐 **Interface Web** | Streamlit dark UI avec graphes interactifs |

---

## Installation

```bash
cd VictorIA
pip install -r requirements.txt
```

> **Note** : TensorFlow est optionnel. Sans lui, le réseau de neurones est désactivé et le système utilise XGBoost (40%) + Random Forest (60%).

---

## Lancement

```bash
streamlit run app.py
```

Ouvrir : **http://localhost:8501**

---

## API Key (optionnelle)

Pour utiliser les données réelles de `football-data.org` :

1. Créer un compte gratuit sur [football-data.org](https://www.football-data.org/client/register)
2. Créer un fichier `.env` à la racine :

```env
FOOTBALL_API_KEY=votre_clé_ici
```

Sans cette clé, l'app fonctionne en **mode démo** avec des données synthétiques.

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## Architecture

```
VictorIA/
├── app.py                  # Streamlit UI
├── predictor.py            # Moteur de prédiction
├── data/
│   ├── data_fetcher.py     # API + mode démo
│   └── preprocessor.py     # 27 features engineering
├── models/
│   ├── xgboost_model.py    # XGBoost (40%)
│   ├── random_forest_model.py  # Random Forest (35%)
│   ├── neural_net.py       # MLP Keras (25%)
│   └── ensemble.py         # Soft-voting ensemble
├── explainability/
│   └── shap_explainer.py   # SHAP waterfall
├── analysis/
│   └── report_generator.py # Rapport FR + résumé IA
└── tests/
```

---

## Features analysées

| Catégorie | Features |
|---|---|
| Équipe domicile | Taux victoire, buts marqués/concédés, form score, W/D/L |
| Équipe extérieure | Mêmes métriques |
| Comparatives | Écarts win rate, buts, forme, diff. buts |
| Elo proxy | `400 × log10(wr_home / wr_away)` |
| H2H | Win rates historiques, total matchs |
| Contexte | Avantage domicile |

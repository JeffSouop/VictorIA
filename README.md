# VictorIA ⚽🧠

**Prédiction de matchs sportifs par intelligence artificielle**

VictorIA analyse des matchs de football et fournit des pronostics explicables (SHAP), avec une interface Streamlit.

---

## Installation

```bash
cd VictorIA
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **TensorFlow** n'est plus requis par défaut. Sans lui, VictorIA utilise **XGBoost (55 %) + Random Forest (45 %)**.

---

## API Key (requise pour l'entraînement réel)

Créez un fichier `.env` à la racine :

```env
FOOTBALL_API_KEY=votre_clé_ici
```

Inscription gratuite : [football-data.org](https://www.football-data.org/client/register)

---

## Entraînement sur données réelles

VictorIA s'entraîne par défaut sur l'**historique réel** des 5 grands championnats (PL, PD, FL1, SA, BL1) via football-data.org :

```bash
python train.py
```

Options :

```bash
python train.py --force --refresh     # re-télécharger l'historique API
python train.py --seasons 2021 2022 2023 2024
python train.py --competitions PL PD FL1 SA BL1
python train.py --synthetic           # fallback debug (données factices)
python train.py --no-eval               # plus rapide
```

Le premier entraînement réel peut prendre **plusieurs minutes** (limites API).

Artefacts sauvegardés dans `cache/` :
- `ensemble.pkl`, `scaler.pkl` — modèles
- `metrics.json` — métriques réelles
- `real_dataset.pkl` — cache du dataset historique

---

## Lancement de l'interface

```bash
streamlit run app.py
```

Ouvrir : **http://localhost:8501**

- **Prédiction** : saisir les équipes et cliquer sur Prédire
- **Performance** : accuracy, log loss, matrice de confusion sur vrais matchs

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## Architecture

```
VictorIA/
├── app.py
├── train.py
├── predictor.py
├── config.py
├── data/
│   ├── data_fetcher.py       # stats live pour prédiction
│   ├── historical_fetcher.py # collecte matchs historiques API
│   ├── stats_builder.py      # calcul stats avant-match
│   └── preprocessor.py
├── training/
│   ├── pipeline.py
│   └── real_dataset.py       # dataset réel X/y
├── models/
│   ├── ensemble.py
│   └── evaluation.py
└── ui/performance.py
```

---

## Avertissement

VictorIA fournit des analyses **informatives**, pas des conseils de pari. Même entraîné sur de vraies données, la performance passée ne garantit pas les résultats futurs.

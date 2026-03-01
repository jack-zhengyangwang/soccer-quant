# Soccer Quant

A dual-model soccer prediction system that combines **quantitative analysis** (machine learning) with **qualitative intelligence** (LLM-powered news analysis) to identify value bets on Premier League and UEFA Champions League matches.

## Approach

Most prediction models rely on numbers alone. Soccer Quant takes a different approach by combining two independent models:

### Quantitative Model (ML)
- Trained on historical match data, xG statistics, and team performance metrics
- Predicts match outcomes (Win/Draw/Loss) and goal totals (Over/Under)
- Features include: team form, home/away record, head-to-head history, xG trends, squad fatigue (UCL rotation)

### Qualitative Model (LLM)
- Scrapes and analyzes recent news, injury reports, and team updates
- Uses Claude/GPT to extract sentiment and context that numbers miss
- Covers: manager changes, key player availability, transfer drama, team morale

### Combined Output
Both models feed into a final prediction with clear evidence:

```
Match: Arsenal vs Chelsea — Apr 5, 2026

PREDICTION: Arsenal Win (62%)

QUANTITATIVE EVIDENCE:
  - Arsenal home win rate: 74% this season
  - Arsenal xG last 5: 2.1 avg vs Chelsea xGA: 1.4 avg
  - Head-to-head last 10: Arsenal 6W 2D 2L at home

QUALITATIVE EVIDENCE:
  - Chelsea missing 2 key midfielders (injury report)
  - Arsenal on 7-game win streak
  - No UCL midweek for Arsenal (extra rest)

MARKET PRICE: Arsenal Win @ 55%
EDGE: +7% → BET RECOMMENDED
```

## Bankroll Allocation

- **70%** — Match outcome bets (Win/Draw/Loss) — lower volatility
- **30%** — Goals bets (Over/Under) — higher variance, higher reward

## Leagues

- English Premier League
- UEFA Champions League

## Tech Stack

- **Python** — data processing, ML models
- **scikit-learn / XGBoost** — quantitative prediction models
- **Claude API / OpenAI API** — qualitative news analysis
- **Kaggle / FBref** — historical data sources
- **Kalshi** — betting platform (US-based prediction market)

## Project Structure

```
soccer-quant/
├── data/               # raw + processed datasets
├── models/             # trained model files
├── scripts/
│   ├── import_data.py  # download datasets via kagglehub
│   ├── features.py     # feature engineering (rolling xG, form, etc.)
│   ├── train.py        # train logistic regression models
│   ├── qualitative.py  # LLM-powered news analysis (planned)
│   └── predict.py      # generate predictions + find value bets (planned)
├── notebooks/          # exploratory data analysis
├── config.py           # league settings, bankroll allocation
└── README.md
```

## Data Sources

| Dataset | Source | Coverage | Used For |
|---------|--------|----------|----------|
| [EPL Match Data 2000-2025](https://www.kaggle.com/datasets/marcohuiii/english-premier-league-epl-match-data-2000-2025) | football-data.co.uk via Kaggle | 9,380 matches, 25 seasons | Match results, shots, corners, cards |
| [Premier League Matches with xG 2021-2025](https://www.kaggle.com/datasets/armin2080/premier-league-matches-dataset-2021-to-2025) | FBref via Kaggle | 3,800 matches, 5 seasons | Match-level xG, xGA, possession, formation |
| [Premier League Team Stats 2016-2025](https://www.kaggle.com/datasets/danielijezie/premier-league-data-from-2016-to-2024) | premierleague.com via Kaggle | 180 team-seasons, 9 seasons | Season-level xG, advanced team stats |

Data is imported using [`kagglehub`](https://github.com/Kaggle/kagglehub) — see `scripts/import_data.py`.

## Status

- [x] Data import pipeline (`scripts/import_data.py`)
- [x] Feature engineering — 12 rolling features from xG data (`scripts/features.py`)
- [x] Logistic regression baseline — match outcome (51.2%) and over/under 2.5 goals (59.4%) (`scripts/train.py`)
- [ ] Qualitative model (LLM news analysis)
- [ ] Combined prediction + value bet detection

## Disclaimer

This project is for educational and personal use. Sports betting involves risk. Never bet more than you can afford to lose.

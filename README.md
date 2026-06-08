# Causal-Aware Recommendation Engine with RL

A personalised movie recommendation system built in 3 stages — 
from basic SVD to causal debiasing to reinforcement learning.

## Project Stages & Results

### Stage 1 — SVD Recommendation Model
- Built matrix factorisation model on MovieLens 100K (100K ratings, 943 users, 1682 movies)
- Improved RMSE by 45% (1.85 → 1.01) by tuning SVD latent factors
- Model explains 80.9% of variance in user-movie rating matrix

### Stage 2 — Causal IPS Debiasing  
- Identified popularity bias — top 10 movies taking 10.8% of all recommendations
- Applied Inverse Propensity Scoring (IPS) weighting to correct bias
- Reduced recommendation bias by 5.2% (10.8% → 5.6%)
- Validated with A/B test simulation

### Stage 3 — Reinforcement Learning Bandit
- Replaced static model with epsilon-greedy multi-armed bandit
- Agent learns from simulated user click feedback in real time
- Achieved 0.917 average reward — 13% improvement over baseline
- Model continuously improves with every interaction

## Results Summary
| Stage | Method | Key Metric |
|-------|--------|------------|
| 1 | SVD Matrix Factorisation | RMSE 1.01 (45% improvement) |
| 2 | Causal IPS Debiasing | 5.2% bias reduction |
| 3 | RL Epsilon-Greedy Bandit | 0.917 reward, 13% lift |

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (TruncatedSVD)
- Matplotlib
- Google Colab
- GitHub

## Dataset
MovieLens 100K — 100,000 ratings, 943 users, 1,682 movies  
Source: grouplens.org/datasets/movielens

## Notebooks
- `stage1_recsys.ipynb` — SVD recommendation model
- `stage2_causal.ipynb` — Causal IPS debiasing
- `stage3_rl.ipynb` — RL bandit feedback loop

## Author
Manoj-R27 | github.com/Manoj-R27/recsys-causal-rl

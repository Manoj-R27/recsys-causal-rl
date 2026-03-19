# Causal-Aware Recommendation Engine

A personalised movie recommendation system built on MovieLens 100K dataset, 
with SVD matrix factorisation and iterative model improvement.

## Results
| Metric | v1 (50 factors) | v2 (200 factors) |
|--------|----------------|-----------------|
| RMSE   | 1.8589         | 1.0193          |
| MAE    | 1.3472         | 0.7567          |
| Variance Explained | 52.4% | 80.9% |

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (TruncatedSVD)
- Matplotlib
- Google Colab

## Dataset
MovieLens 100K — 100,000 ratings, 943 users, 1,682 movies  
Source: grouplens.org

## Project Stages
- **Stage 1 (current):** SVD recommendation model ✅
- **Stage 2 (upcoming):** Causal debiasing with IPS weighting
- **Stage 3 (upcoming):** Reinforcement learning bandit loop

## How to Run
1. Open `stage1_recsys.ipynb` in Google Colab
2. Run all cells in order
3. Get personalised movie recommendations!

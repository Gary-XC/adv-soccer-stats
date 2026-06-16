# Advanced Soccer Stats & World Cup Predictor

An end-to-end MLOps pipeline for predicting soccer match outcomes using a hybrid **Top-Down/Bottom-Up EWMA** feature engineering approach, **LightGBM** stacking models, and **Live Elo Ratings**. 

The system features a **FastAPI** backend for millisecond inference, a **Streamlit** frontend for interactive dashboards, and a fully automated **GitHub Actions CI/CD** pipeline that fetches daily match data from API-Football to keep the model's understanding of team form up-to-date.

- **Live App:** [Hugging Face Space Link Here](https://huggingface.co/spaces/GaryC246/soccer-predictions)
- **Model Weights:** [Hugging Face Model Link Here](https://huggingface.co/GaryC246/soccer-models/tree/main)

CI/CD & Automated Data Updates
This project uses GitHub Actions to automatically fetch new match data and update the model's historical context without manual intervention.
Daily Data Fetch: A cron job runs scripts/update_and_generate.py every day at 6:00 AM UTC.
API Ingestion: It queries API-Football for yesterday's matches, extracts player stats, and appends them to the raw dataset.
Matrix Regeneration: It recalculates the hybrid_matrices.csv with updated EWMA form and rest days.
Auto-Deploy: It pushes the updated CSV directly to the Hugging Face Model Hub, instantly updating the live application.

## Architecture & Tech Stack
- **Feature Engineering:** Pandas, NumPy (EWMA, Rest Days)
- **Modeling:** LightGBM (Base Learners), Logistic Regression (Meta-Learner)
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **Deployment:** Docker, Hugging Face Spaces
- **CI/CD:** GitHub Actions (Automated daily data ingestion & deployment)
- **Data Source:** API-Football, World Football Elo Ratings

## Project Structure

```text
adv-soccer-stats/
├── .github/workflows/       # CI/CD pipelines for data updates and HF deployment
├── data/                    # Raw data and generated hybrid matrices
├── scripts/                 # API fetchers and matrix generation orchestrators
├── src/                     # Core ML logic (training, tuning, feature engineering)
└── requirements.txt         # Python dependencies
```


# Soccer Advanced Statistics

A soccer analytics project that adapts advanced metrics from other sports (like NBA-style “usage rate” ideas) to soccer. The goal is to support simple player comparison (1–2 players at a time), involvement-style metrics, season-end projections, and year-over-year development views.

This repository contains:
- Data collection (scraping + API ingestion)
- Data cleaning / normalization
- A Streamlit app for exploration and comparison

## Project Goals
Design and implement a soccer analytics platform that translates advanced metrics from other sports (like the NBA's “usage rate”, and other advanced metrics) into soccer, enabling:
- Player comparison (1–2 players at a time)
- Measurement of on-ball and off-ball involvement
- Season-end statistical projections
- Historical development analysis (year-over-year trends)
- Quantify player involvement using both on-ball (touches, carries, SCA, xG/xA) and off-ball proxies (pressures, defensive actions).

Scope: Europe’s Top 5 Leagues. Data updates after each match day (batch refresh).

## Current Features (Initial Commit)
- Project scaffolding for ingestion, cleaning, and app layers
- Basic dataset loading (CSV-based for now)
- Streamlit skeleton: select a player and display basic bio + season totals (placeholder data acceptable)

## Planned Features
### Streamlit App (Player Comparisons)
- Select 1–2 players to view/compare
- Display:
  - Current club + national team (when available)
  - Age, position
  - Goals/assists and core summary stats
- Tables:
  - Basic table: appearances, minutes, goals, assists (and other role-relevant basics)
  - Current season advanced-stat table (as metrics become available)
- Projections:
  - Graph projecting season-end outputs using current per-90 or per-match rates
- Historical development:
  - Select a stat (or small set of stats) to view year-by-year trends

### Advanced Soccer Metrics (Inspired by Other Sports)
- Soccer-adapted “usage rate” style metrics to quantify involvement
- Additional advanced statistics adapted from other sports analytics approaches
- Metric definitions and assumptions will be documented in `docs/metrics.md`

### Machine Learning Extensions
This project includes several machine learning components designed to enhance player evaluation
#### 1. Player Performance Rating Model
A supervised learning model predicts a player performance rating (1–10) using match and season statistics.
Objective:
- Learn a mapping from objective on-field metrics to human-evaluated performance scores.
Training Labels
Using External ratings from:
- FotMob
- SofaScore
- WhoScored

Features
- Custom soccer stats
- Per-90 offensive metrics
- xG / xA
- Defensive actions
- Minutes played
- Position-adjusted indicators

Models
- Linear Regression (baseline)
- Gradient Boosted Trees (primary)

Evaluation
- RMSE / MAE
- Correlation with external rating sources
- Feature importance analysis

#### 2. Player Ranking System
Players are ranked by their predicted performance ratings, allowing:
- In-team comparisons
- League-wide rankings
- Position-specific leaderboards
- Rankings are validated against external provider rankings.

#### 3. Season-End Statistical Prediction
Machine learning models predict season-end outputs based on:

Current production rate
- Recent form
- Expected minutes
- Historical trends
- Predicted metrics include:
- Goals
- Assists
- xG
- Advanced stats

### Data
- Scrape websites for historical matches and player statistics
- Use live soccer APIs for current and up-to-date stats
- Merge multiple sources (each source provides different coverage/metrics, with some overlap, as some sources include more advanced stats, which are needed for the custom feature engineering)


## Tech Stack
- App: Streamlit
- Data: Pandas
- Visualization: Matplotlib, Seaborn, Plotly

## Repo Structure (Initial)
  README.md
  requirements.txt
  .gitignore
  data/
    raw_csv/
  src/
    __init__.py
    config.py
    io_utils.py
    collect_fbref.py
    collect_understat.py
    collect_fotmob.py
    collect_all.py
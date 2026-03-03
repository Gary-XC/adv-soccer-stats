## Soccer Advanced Statistics: Text-to-Pandas Analytics
- An advanced soccer analytics platform that adapts statistics (like high-usage metrics) from the NBA and MLB to evaluate European football
- This project has evolved from a traditional scraper into a Local AI Agent architecture, allowing users to query complex multi-league datasets using natural language

### The Change: From Scrapers to Agents
- Originally planned as a web-scraping tool and comprehensive statistics, the project shifted in March 2026 due to increased anti-bot protections and data removal on major platforms (FBref/FotMob)
- Scraping data: Brittle scrapers prone to 403 errors.
- Static CSVs: High-fidelity ETL pipeline using static CSVs and a Text-to-Pandas AI Agent for seamless data exploration and machine learning.

### Core Innovation: NBA-Style Soccer Metrics
We bridge the gap between low-scoring soccer and high-volume basketball by engineering custom metrics:
| Metric           | Origin         | Soccer Adaptation                                                                 |
|------------------|---------------|-------------------------------------------------------------------------------------|
| Possession USG%  | NBA Usage     | % of team touches a player accounts for while on the pitch                       |
| Attacking USG%   | Ball Dominance| Quantifies "Final Third Gravity" using Shots, Progressive Carries       |
| TFE              | True Shooting | True Finishing Efficiency: Actual Goals vs. Expected Goals          |
| Goals Added      | MLB WAR       | Offensive value added above the positional average per 90 minutes                |

### Current Limitations & Roadmap
As this project currently utilizes a single-season snapshot (2024-2025), certain analytical constraints exist:
1. The "Sample Size" Challenge
- Limited Trend Analysis: With only one season, we cannot yet track a player's development trajectory or "Aging Curve"
- Volatility: Single-season Finishing Efficiency (TFE) can be subject to extreme variance (luck or simply having a good season, unable to verify if they can keep it up consistently over multiple seasons); multiple seasons are required to determine if a player is a "true" elite finisher or just on a hot streak

2. Solving for Multi-Season Data
- By integrating historical data (e.g., 2020–2024), the platform would expand to include:
- Year-over-Year Development Views: Visualizing a player's growth in "Offensive Burden" over time
- Predictive Breakout Models: Using "Usage Rate" jumps in Season N to predict "Goal Output" explosions in Season N+1
- Time-Series Forecasting: Moving from simple linear projections to advanced ARIMA or LSTM models for career path forecasting

### Machine Learning & Analytics
- The project utilizes scikit-learn for in-season evaluation:
- Clustering (K-Means): Groups players into 5 distinct "Usage Roles" to find statistical twins (e.g., "High-Volume Creators")
- Similarity Search: Euclidean distance mapping to find comparable players across different leagues
- In-Season Projections: Linear models projecting current efficiency (TFE) and volume across a full 38-game season

### Project Architecture
- The pipeline is designed for speed and modularity:
- Data Layer: 5 domain-specific CSVs (Standard, Passing, Shooting, Defense, Possession) derived from the 2024-2025 Kaggle Football Dataset
- Logic Layer: Custom Python functions for cross-sport metric engineering
- AI Layer: A Text-to-Pandas agent that translates natural language queries into executable Python code
- Visualization Layer: High-impact "Pizza" charts via mplsoccer and Volume vs. Efficiency scatter plots

### Key Features:
- Statistical Twin Search: "Who is the most similar player to Vinicius Júnior based on usage?"
- Efficiency Quadrants: Identify "High-Volume, Low-Efficiency" vs. "Elite Finishers."
- Automated Visualization: Instant generation of percentile radars for 1-to-1 player comparisons
- Top 5 League Scope: Full coverage of Premier League, La Liga, Serie A, Bundesliga, and Ligue 1

### Tech Stack:
- Language: Python 3.12
- Data Manipulation: Pandas, NumPy
- Machine Learning: Scikit-Learn, SciPy
- Visualization: mplsoccer, Seaborn, Matplotlib
- AI Integration: LangChain / Text-to-Pandas Agent Scaffolding, RAG
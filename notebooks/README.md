# Notebooks For Testing Data Collection, Data Analysis, Performance Predictions And Functions
The notebooks here are for experimentation, focused on:
- Verifying data availability
- Understanding raw data structures
- Stress-testing data sources
- Prototyping early analysis, modeling ideas, and features

Clean, reusable logic wil be migrated into scripts/modules once properly validated.


### Data Collection: Scraping and API information


#### 1/20:
Goal:
- Evaluate which public data sources can reliably provide:
- Player-level statistics
- Match-level statistics
- Historical data for Europe’s Top 5 leagues

Sources Tested
FBref
- Issue encountered:
    - Requests from Python were blocked by Cloudflare anti-bot protection
    - Resulted in persistent 403 responses and failed downloads

- Attempted solutions:
    - Cache reset
    - Season format changes
    - Library version pinning

Decided to skip FBRef, and instead focus on other working sources, FBref will be an optional source if the access is available later on.


### Data Cleaning and Pre-processing

### Analysis of Player Performance
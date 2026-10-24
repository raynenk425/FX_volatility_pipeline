# FX Volatility & Portfolio Optimisation Pipeline 💱

## Overview
This project is an automated, end-to-end data pipeline built in Python and SQL that ingests, analyses, and visualises historical foreign exchange (FX) market data. It was built to programmatically identify macroeconomic volatility clusters and mathematically minimise currency risk using constrained non-linear optimisation.

This pipeline demonstrates core data engineering and quantitative analysis skills, moving from raw API extraction to SQL data transformation, statistical anomaly detection, and portfolio optimisation.

## Key Features
* **Automated ETL Pipeline:** Ingests 5 years of daily FX market data (GBP, EUR, JPY) via the `yfinance` API into a structured format.
* **In-Memory SQL Analytics:** Utilises `SQLite` and SQL Window Functions (`LAG`, `OVER PARTITION BY`) to compute daily returns and structure the data for rapid analysis.
* **Statistical Anomaly Detection (3-Sigma):** Implements dynamic 30-day rolling means and standard deviations to calculate Z-scores, isolating true market shocks from baseline currency fluctuations.
* **Markowitz Portfolio Optimisation:** Leverages `SciPy.optimize` (Sequential Least Squares Programming) to calculate the exact multi-currency weightings required to minimise total variance (exchange-rate risk) based on historical covariance matrices.
* **Data Visualisation:** Uses `Matplotlib` to graph currency returns and overlay flagged anomalous volatility regimes.

## Tech Stack
* **Language:** Python 3
* **Database:** SQLite (In-Memory)
* **Libraries:**
    * `pandas`, `numpy` (Data manipulation & numerical computing)
    * `sqlite3` (Relational database querying)
    * `scipy.optimize` (Mathematical minimisation solvers)
    * `yfinance` (Data extraction)
    * `matplotlib` (Data visualisation)

## Use Case (FinTech / Cross-Border Payments)
For a remittance or FinTech company, identifying macro-volatility clusters programmatically is critical for dynamic pricing, while determining optimal currency weightings is essential for risk hedging and treasury management. This project simulates the backend architecture required to power those operational decisions.

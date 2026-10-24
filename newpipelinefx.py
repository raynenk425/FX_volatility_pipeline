import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import scipy.optimize as opt
import matplotlib.pyplot as plt

# STEP 1: EXTRACT (Fetch Data)
currencies = {"GBP": "GBPUSD=X", "EUR": "EURUSD=X", "JPY": "JPYUSD=X"}
dfs = []

print("Fetching historical FX data...")
for name, ticker in currencies.items():
    tkr = yf.Ticker(ticker)
    df = tkr.history(start="2021-01-01", end="2026-10-01")[['Close']].reset_index()

    df.columns = ['Date', 'Exchange_Rate']
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df['Currency'] = name
    dfs.append(df)

master_df = pd.concat(dfs, ignore_index=True)

# ==========================================
# STEP 2: HYBRID SQL & PYTHON PIPELINE
# ==========================================
print("Running In-Memory SQLite Analytics...")
# We connect to a temporary in-memory database to keep it lightning fast
conn = sqlite3.connect(':memory:')
master_df.to_sql('fx_data', conn, index=False)

# 1. SQL LAYER: Use Window Functions to calculate the Daily Return
sql_query = """
            WITH Daily_Changes AS (SELECT
                Date \
               , Currency \
               , Exchange_Rate \
               , ((Exchange_Rate - LAG(Exchange_Rate) OVER (PARTITION BY Currency ORDER BY Date))
                / LAG(Exchange_Rate) OVER (PARTITION BY Currency ORDER BY Date)) AS Daily_Return
            FROM fx_data
                )
            SELECT *
            FROM Daily_Changes
            WHERE Daily_Return IS NOT NULL; \
            """
analysis_df = pd.read_sql(sql_query, conn)
conn.close()

# 2. PYTHON LAYER: Statistical Detrending (3-Sigma Z-Score)
print("Applying Statistical Anomaly Detection...")
analysis_df['Rolling_Mean'] = analysis_df.groupby('Currency')['Daily_Return'].transform(lambda x: x.rolling(30).mean())
analysis_df['Rolling_Std'] = analysis_df.groupby('Currency')['Daily_Return'].transform(lambda x: x.rolling(30).std())

# Z-Score Calculation
analysis_df['Z_Score'] = (analysis_df['Daily_Return'] - analysis_df['Rolling_Mean']) / analysis_df['Rolling_Std']

# Flag anomalies greater than 3 standard deviations
analysis_df['Is_Anomaly'] = np.where(analysis_df['Z_Score'].abs() > 3, 1, 0)

# ==========================================
# STEP 3: SCIPY PORTFOLIO OPTIMIZATION
# ==========================================
print("Running SciPy Non-Linear Optimization...")
# Pivot the data to create a matrix of daily returns
returns_matrix = analysis_df.pivot(index='Date', columns='Currency', values='Daily_Return').dropna()
cov_matrix = returns_matrix.cov().values


# Objective Function: Minimize Portfolio Variance
def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights


# Initial guess (equal weighting)
num_assets = len(currencies)
init_guess = np.ones(num_assets) / num_assets

# Constraints & Bounds
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))

# Run Optimizer
optimal_result = opt.minimize(
    portfolio_variance,
    init_guess,
    args=(cov_matrix,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)
optimal_weights = optimal_result.x

print("\n--- OPTIMAL CURRENCY PORTFOLIO WEIGHTS ---")
print("To mathematically minimize exposure to FX volatility risk over this 5-year period:")
for i, currency in enumerate(returns_matrix.columns):
    print(f"{currency}: {optimal_weights[i] * 100:.2f}%")
print("------------------------------------------\n")

# ==========================================
# STEP 4: VISUALIZATION
# ==========================================
print("Generating Volatility Plot...")
gbp_data = analysis_df[analysis_df['Currency'] == 'GBP'].copy()
anomalies = gbp_data[gbp_data['Is_Anomaly'] == 1]

plt.style.use('dark_background')
plt.figure(figsize=(12, 6))

plt.plot(pd.to_datetime(gbp_data['Date']), gbp_data['Daily_Return'], label='GBP Daily Return', color='#00b9ff',
         alpha=0.6)
plt.scatter(pd.to_datetime(anomalies['Date']), anomalies['Daily_Return'], color='#ff00d4', label='3-Sigma Anomaly',
            zorder=5)

plt.title('GBP/USD Daily Returns & 3-Sigma Volatility Anomalies (2021-2026)')
plt.ylabel('Daily Return (%)')
plt.legend()
plt.tight_layout()
plt.show()

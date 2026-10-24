import yfinance as yf
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt


# STEP 1: EXTRACT (Fetch Data via Python)
currency_pairs = {"GBP/USD": "GBPUSD=X", "GBP/EUR": "GBPEUR=X"}
data_frames = []

print("Fetching historical FX data...")
for pair_name, ticker in currency_pairs.items():
    # Use yf.Ticker().history() to avoid MultiIndex column issues
    tkr = yf.Ticker(ticker)
    df = tkr.history(start="2021-01-01", end="2026-10-01")

    # We only need the 'Close' column
    df = df[['Close']].copy()

    # Reset the index to move the date from the index to a column
    df.reset_index(inplace=True)

    # Force exact column names so SQL knows exactly what to look for
    df.columns = ['Date', 'Exchange_Rate']

    # Convert the Date to a standard string format for cleaner SQL sorting
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    df['Currency_Pair'] = pair_name
    data_frames.append(df)

# Combine into one master Pandas DataFrame
master_df = pd.concat(data_frames, ignore_index=True)


# STEP 2: LOAD (Push to SQLite Database)

print("Building SQL Database...")
# This creates a local database file in your folder called 'wise_fx.db'
conn = sqlite3.connect('wise_fx.db')

# Load the Pandas dataframe into a SQL table called 'historical_rates'
master_df.to_sql('historical_rates', conn, if_exists='replace', index=False)


# STEP 3: TRANSFORM & ANALYZE (The SQL Flex)

# We use SQL Window Functions (LAG and AVG) to calculate volatility
sql_query = """
            WITH Volatility_Calc AS (SELECT
                Date \
               , Currency_Pair \
               , Exchange_Rate \
               ,
            -- Calculate the 30-Day Moving Average
                AVG (Exchange_Rate) OVER (
                PARTITION BY Currency_Pair
                ORDER BY Date
                ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
                ) as Moving_Avg_30D \
               ,
            -- Calculate the daily percentage change using the LAG function (previous day's rate)
               ((Exchange_Rate - LAG(Exchange_Rate) OVER (PARTITION BY Currency_Pair ORDER BY Date))
                / LAG(Exchange_Rate) OVER (PARTITION BY Currency_Pair ORDER BY Date)) * 100 as Daily_Pct_Change
            FROM historical_rates
                )
            SELECT *
            FROM Volatility_Calc
            WHERE Date IS NOT NULL; \
            """

print("Running SQL Analysis...")
# Run the SQL query and load the results back into a new Pandas DataFrame
analysis_df = pd.read_sql(sql_query, conn)

# STEP 4: VISUALIZE (Business Insight)
# Filter for GBP/USD and drop NaN values from the moving average calculation
gbp_usd = analysis_df[analysis_df['Currency_Pair'] == 'GBP/USD'].dropna()

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

# Plot 1: Exchange Rate vs 30-Day Moving Average
ax1.plot(pd.to_datetime(gbp_usd['Date']), gbp_usd['Exchange_Rate'], label='Daily Rate', color='#00b9ff', alpha=0.7)
ax1.plot(pd.to_datetime(gbp_usd['Date']), gbp_usd['Moving_Avg_30D'], label='30-Day Moving Avg', color='#ff00d4',
         linewidth=2)
ax1.set_title('GBP/USD Exchange Rate & 30-Day Moving Average (2021-2026)')
ax1.set_ylabel('Exchange Rate')
ax1.legend()

# Plot 2: Daily Percentage Change (Volatility Spikes)
ax2.bar(pd.to_datetime(gbp_usd['Date']), gbp_usd['Daily_Pct_Change'], color='white', alpha=0.5)
ax2.axhline(0, color='grey', linewidth=0.5)
ax2.set_title('Daily Volatility (% Change)')
ax2.set_ylabel('% Change')

plt.tight_layout()
plt.show()

# Close the database connection
conn.close()

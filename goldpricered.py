import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load & clean dataset
df = pd.read_csv('future-gc00-daily-prices.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
for col in ['Open', 'High', 'Low', 'Close']:
    df[col] = df[col].str.replace(',', '').astype(float)
df = df.sort_values('Date')

# Lag features
df['Close_lag1'] = df['Close'].shift(1)
df['Close_lag7'] = df['Close'].shift(7)
df['Close_rolling7'] = df['Close'].rolling(window=7).mean()
df['Close_pct_change'] = df['Close'].pct_change()

# Time-based features
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek

df.dropna(inplace=True)

# Feature columns (predict short-term)
features = ['year', 'month', 'day', 'dayofweek', 
            'Close_lag1', 'Close_lag7', 'Close_rolling7', 'Close_pct_change',
            'Open', 'High', 'Low']

# Split
split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train[features]
y_train = train['Close']
X_test = test[features]
y_test = test['Close']

# Train model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

#Plot
import matplotlib.pyplot as plt

# Make date column -> datetime
df['Date'] = pd.to_datetime(df['Date'])

# Make test data to weekly (or monthly) to reduce clutter & smoother plot
test_plot = pd.DataFrame({
    'Date': df.loc[X_test.index, 'Date'],
    'Actual': y_test,
    'Predicted': y_pred
}).set_index('Date').resample('W').mean().dropna()

# Do Plot
plt.figure(figsize=(12, 6))
plt.plot(test_plot.index, test_plot['Actual'], label='Actual', color='blue')
plt.plot(test_plot.index, test_plot['Predicted'], label='Predicted', color='orange')
plt.xlabel('Date')
plt.ylabel('Gold Future Close Price')
plt.title('Gold Future Price Prediction (XGBoost Regressor, Weekly Avg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def predict_price(date_str):
    date = pd.to_datetime(date_str)

    # Get latest known row to fill lag values
    latest = df.iloc[-1]

    # Build a single-row input DataFrame
    input_data = pd.DataFrame({
        'year': [date.year],
        'month': [date.month],
        'day': [date.day],
        'dayofweek': [date.dayofweek],
        'Close_lag1': [latest['Close']],
        'Close_lag7': [df.iloc[-7]['Close']],
        'Close_rolling7': [df['Close'].rolling(window=7).mean().iloc[-1]],
        'Close_pct_change': [(latest['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close']],
        'Open': [latest['Open']],
        'High': [latest['High']],
        'Low': [latest['Low']]
    })

    prediction = model.predict(input_data)[0]
    return round(prediction, 2)

# Test
future_dates = [
    "2024-04-14",
    "2024-04-21",
    "2024-05-05",
    "2024-05-12",
    "2024-06-02"
]

# Predict gold future prices for each date
for date_str in future_dates:
    predicted_price = predict_price(date_str)
    print(f"Predicted Gold Future Price for {date_str}: ${predicted_price}")

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# Feature columns
features = ['year', 'month', 'day', 'dayofweek', 
            'Close_lag1', 'Close_lag7', 'Close_rolling7', 'Close_pct_change',
            'Open', 'High', 'Low']

# Split data
split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train[features]
y_train = train['Close']
X_test = test[features]
y_test = test['Close']

# Train model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot
test_plot = pd.DataFrame({
    'Date': df.loc[X_test.index, 'Date'],
    'Actual': y_test,
    'Predicted': y_pred
}).set_index('Date').resample('W').mean().dropna()

# Predict a future price given a date
def predict_price(date_str):
    date = pd.to_datetime(date_str)

    # Use latest known data to simulate features
    latest = df.iloc[-1]

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

# Test predictions for future dates
future_dates = [
    "2024-04-14",
    "2024-04-21",
    "2024-05-05",
    "2024-05-12",
    "2024-06-02"
]

future_preds = []
for date_str in future_dates:
    predicted_price = predict_price(date_str)
    future_preds.append({'Date': pd.to_datetime(date_str), 'Predicted': predicted_price})
    print(f"Predicted Gold Future Price for {date_str}: ${predicted_price}")

# Build future prediction DataFrame
future_df = pd.DataFrame(future_preds).set_index('Date')
future_df['Actual'] = None  # No actual values

# Combine with test_plot for extended plot
extended_plot = pd.concat([test_plot, future_df]).sort_index()

# Plot
plt.figure(figsize=(10, 5))  # smaller figure

# Plot actual values
plt.plot(extended_plot.index, extended_plot['Actual'], label='Actual (Test)', color='blue', linewidth=2)

# Plot predicted values up to latest known date
past_preds = extended_plot.loc[extended_plot.index <= df['Date'].iloc[-1], 'Predicted']
plt.plot(past_preds.index, past_preds, label='Model Prediction (Test)', color='orange', linewidth=2)

# Plot future forecast (after last dataset date) 
future_preds_plot = extended_plot.loc[extended_plot.index > df['Date'].iloc[-1], 'Predicted']
plt.plot(future_preds_plot.index, future_preds_plot, label='Forecast (Future)', color='red', linestyle='--', marker='o')

# Vertical line at forecast start
plt.axvline(df['Date'].iloc[-1], color='gray', linestyle=':', label='Forecast Start')

# Formatting
plt.xlabel('Date')
plt.ylabel('Gold Future Close Price')
plt.title('Gold Future Price Prediction (XGBoost, Weekly Avg)')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.4)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

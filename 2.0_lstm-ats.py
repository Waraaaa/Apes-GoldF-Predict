import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Load & clean dataset
df = pd.read_csv('future-gc00-daily-prices.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
for col in ['Open', 'High', 'Low', 'Close']:
    df[col] = df[col].str.replace(',', '').astype(float)
df = df.sort_values('Date')
df.reset_index(drop=True, inplace=True)

# Normalize
scaler = MinMaxScaler()
df['Close_scaled'] = scaler.fit_transform(df[['Close']])

# Prepare sequences for LSTM
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LEN = 20
data = df['Close_scaled'].values
X, y = create_sequences(data, SEQ_LEN)

# Train-test split
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Convert -> tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take output from the last timestep
        return out

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    output = model(X_train_tensor.unsqueeze(-1)).squeeze()
    loss = criterion(output, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# Evaluate
model.eval()
preds = model(X_test_tensor.unsqueeze(-1)).detach().numpy()
preds_rescaled = scaler.inverse_transform(preds)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_rescaled, preds_rescaled)
r2 = r2_score(y_test_rescaled, preds_rescaled)
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Make test data for weekly plot
test_dates = df['Date'].iloc[SEQ_LEN + split_idx:]
test_plot = pd.DataFrame({
    'Date': test_dates,
    'Actual': y_test_rescaled.flatten(),
    'Predicted': preds_rescaled.flatten()
}).set_index('Date').resample('W').mean().dropna()

# Future dates to predict
future_dates = [
    "2024-04-14",  
    "2024-04-21",  
    "2024-05-05",  
    "2024-05-12",  
    "2024-06-02"   
]

# Predict future values based on the last sequence of the training data
future_steps = len(future_dates)

# Initialize with the last sequence from the training data
input_seq = torch.tensor(data[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

model.eval()
future_preds_scaled = []
with torch.no_grad():
    for _ in range(future_steps):
        pred = model(input_seq).item()
        future_preds_scaled.append(pred)
        next_input = torch.tensor([[pred]], dtype=torch.float32).unsqueeze(0)
        input_seq = torch.cat([input_seq[:, 1:, :], next_input], dim=1)

# Convert scaled predictions back -> original values
future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))

# Build future prediction DataFrame
future_df = pd.DataFrame({
    'Date': pd.to_datetime(future_dates),
    'Predicted': future_preds.flatten()
}).set_index('Date')

# Add actual future prices for comparison
actual_closing_prices = {
    "2024-04-14": 2436.10,
    "2024-04-21": 2369.30,
    "2024-05-05": 2397.70,
    "2024-05-12": 2440.40,
    "2024-06-02": 2347.60
}
future_df['Actual'] = future_df.index.map(lambda d: actual_closing_prices.get(d.strftime('%Y-%m-%d')))

# Show comparison
print("\nFuture Predictions vs Actual:")
print(future_df[['Predicted', 'Actual']])

# Combine with test_plot for full plot
extended_plot = pd.concat([test_plot, future_df]).sort_index()

# Plot
plt.figure(figsize=(10, 5))
# Actual values (test set)
plt.plot(extended_plot.index, extended_plot['Actual'], label='Actual (Test)', color='blue', linewidth=2)
# Predicted values (test set)
past_preds = extended_plot.loc[extended_plot.index <= df['Date'].iloc[-1], 'Predicted']
plt.plot(past_preds.index, past_preds, label='Model Prediction (Test)', color='orange', linewidth=2)
# Forecasted predictions (future)
future_preds_plot = extended_plot.loc[extended_plot.index > df['Date'].iloc[-1], 'Predicted']
plt.plot(future_preds_plot.index, future_preds_plot, label='Forecast (Future)', color='red', linestyle='--', marker='o')
# Actual future prices
actual_future = extended_plot.loc[future_preds_plot.index, 'Actual']
plt.plot(actual_future.index, actual_future, label='Actual (Future)', color='green', linestyle='None', marker='x', markersize=8)

# Error lines (prediction VS actual)
for date in actual_future.index:
    if pd.notna(actual_future[date]) and pd.notna(future_preds_plot[date]):
        plt.plot([date, date], [actual_future[date], future_preds_plot[date]], color='gray', linestyle=':', linewidth=1)

# Forecast line
plt.axvline(df['Date'].iloc[-1], color='gray', linestyle=':', label='Forecast Start')

# Formatting
plt.xlabel('Date')
plt.ylabel('Gold Future Close Price')
plt.title('Gold Future Price Prediction (LSTM, Weekly Avg)')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.4)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

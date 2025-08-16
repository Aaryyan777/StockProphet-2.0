import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import joblib

tkr = "AAPL"
df_raw = yf.download(tkr, start="2020-01-01", end="2024-01-01")

df_proc = df_raw.reset_index()
df_proc = df_proc[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
df_proc.columns = ['ds', 'y', 'Open', 'High', 'Low', 'Volume']

df_proc['SMA_10'] = df_proc['y'].rolling(window=10).mean()
df_proc['SMA_30'] = df_proc['y'].rolling(window=30).mean()

def calc_rsi(d, window=14):
    dif = d.diff(1)
    gn = dif.where(dif > 0, 0)
    ls = -dif.where(dif < 0, 0)
    ag = gn.ewm(com=window - 1, min_periods=window).mean()
    al = ls.ewm(com=window - 1, min_periods=window).mean()
    rs_val = ag / al
    rsi_val = 100 - (100 / (1 + rs_val))
    return rsi_val

def calc_atr(df, window=14):
    tr1 = df['High'] - df['Low']
    tr2 = np.abs(df['High'] - df['y'].shift(1))
    tr3 = np.abs(df['Low'] - df['y'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(com=window - 1, min_periods=window).mean()
    return atr

def calc_macd(d, fast=12, slow=26, signal=9):
    exp1 = d.ewm(span=fast, adjust=False).mean()
    exp2 = d.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calc_bb(d, window=20, std=2):
    ma = d.rolling(window=window).mean()
    sd = d.rolling(window=window).std()
    upper_band = ma + (sd * std)
    lower_band = ma - (sd * std)
    return upper_band, lower_band

df_proc['RSI'] = calc_rsi(df_proc['y'])
df_proc['ATR'] = calc_atr(df_proc)
df_proc['MACD'], df_proc['MACD_signal'] = calc_macd(df_proc['y'])
df_proc['BB_upper'], df_proc['BB_lower'] = calc_bb(df_proc['y'])
df_proc['y_lag1'] = df_proc['y'].shift(1)
df_proc['y_lag2'] = df_proc['y'].shift(2)
df_proc['y_lag3'] = df_proc['y'].shift(3)
df_proc.dropna(inplace=True)

def calc_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0]
}

all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
rmses = []

for params in all_params:
    m = Prophet(**params).fit(df_proc)
    df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])

best_params = all_params[np.argmin(rmses)]
print(f"Best Parameters: {best_params}")

init_train_sz = int(len(df_proc) * 0.7)
hrz = 30
stp = 30

maes = []
rmses = []
mapes = []
r2s = []
dir_accs = []

print("Starting Rolling Window Cross-Validation with Best Parameters...")

for i in range(init_train_sz, len(df_proc) - hrz + 1, stp):
    df_train = df_proc.iloc[:i].copy()
    df_test = df_proc.iloc[i:i + hrz].copy()

    if len(df_test) == 0:
        continue

    print(f"Fold: Training on {df_train['ds'].min()} to {df_train['ds'].max()}, Testing on {df_test['ds'].min()} to {df_test['ds'].max()}")

    model = Prophet(**best_params)
    model.add_regressor('SMA_10')
    model.add_regressor('SMA_30')
    model.add_regressor('RSI')
    model.add_regressor('ATR')
    model.add_regressor('MACD')
    model.add_regressor('MACD_signal')
    model.add_regressor('BB_upper')
    model.add_regressor('BB_lower')
    model.add_regressor('y_lag1')
    model.add_regressor('y_lag2')
    model.add_regressor('y_lag3')
    model.fit(df_train)

    future = df_test[['ds', 'SMA_10', 'SMA_30', 'RSI', 'ATR', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'y_lag1', 'y_lag2', 'y_lag3']]
    forecast = model.predict(future)

    df_test_align = df_test.set_index('ds')
    forecast_align = forecast.set_index('ds')

    idx_common = df_test_align.index.intersection(forecast_align.index)
    if len(idx_common) == 0:
        print("No common dates for evaluation in this fold. Skipping.")
        continue

    y_true = df_test_align.loc[idx_common, 'y']
    y_pred = forecast_align.loc[idx_common, 'yhat']

    maes.append(mean_absolute_error(y_true, y_pred))
    rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))
    mapes.append(calc_mape(y_true, y_pred))
    r2s.append(r2_score(y_true, y_pred))

    dir_true = np.sign(y_true.diff().dropna())
    dir_pred = np.sign(y_pred.diff().dropna())

    idx_dir_common = dir_true.index.intersection(dir_pred.index)
    if len(idx_dir_common) > 0:
        dir_true_align = dir_true.loc[idx_dir_common]
        dir_pred_align = dir_pred.loc[idx_dir_common]
        dir_acc = np.mean(dir_true_align == dir_pred_align) * 100
        dir_accs.append(dir_acc)
    else:
        print("Not enough data for directional accuracy in this fold.")

print("\n--- Cross-Validation Results (Prophet Model) ---")
print(f"Average MAE: {np.mean(maes):.2f} (Std: {np.std(maes):.2f})")
print(f"Average RMSE: {np.mean(rmses):.2f} (Std: {np.std(rmses):.2f})")
print(f"Average MAPE: {np.mean(mapes):.2f}% (Std: {np.std(mapes):.2f}%)")
print(f"Average R-squared (R²): {np.mean(r2s):.2f} (Std: {np.std(r2s):.2f})")
print(f"Average Directional Accuracy: {np.mean(dir_accs):.2f}% (Std: {np.std(dir_accs):.2f}%)")

# Retrain the best model on the full dataset for deployment
print("\nRetraining best model on full dataset for deployment...")
model_final = Prophet(**best_params)
model_final.add_regressor('SMA_10')
model_final.add_regressor('SMA_30')
model_final.add_regressor('RSI')
model_final.add_regressor('ATR')
model_final.add_regressor('MACD')
model_final.add_regressor('MACD_signal')
model_final.add_regressor('BB_upper')
model_final.add_regressor('BB_lower')
model_final.add_regressor('y_lag1')
model_final.add_regressor('y_lag2')
model_final.add_regressor('y_lag3')
model_final.fit(df_proc)

model_file = 'prophet_model.joblib'
joblib.dump(model_final, model_file)
print(f"Model saved as {model_file}")

# Plotting the final forecast (optional, can be removed for production script)
future_full = model_final.make_future_dataframe(periods=30)

# Create a dataframe for the forecast period to calculate regressors
last_date = df_proc['ds'].max()
future_dates = pd.to_datetime(future_full[future_full['ds'] > last_date]['ds'])

# Temporarily predict future y values to calculate future regressors
temp_future = future_full.copy()
temp_future['SMA_10'] = df_proc['SMA_10'].iloc[-1]
temp_future['SMA_30'] = df_proc['SMA_30'].iloc[-1]
temp_future['RSI'] = df_proc['RSI'].iloc[-1]
temp_future['ATR'] = df_proc['ATR'].iloc[-1]
temp_future['y_lag1'] = df_proc['y_lag1'].iloc[-1]
temp_future['y_lag2'] = df_proc['y_lag2'].iloc[-1]
temp_future['y_lag3'] = df_proc['y_lag3'].iloc[-1]
temp_forecast = model_final.predict(temp_future)

# Combine historical and forecasted y values
combined_y = pd.concat([
    df_proc.set_index('ds')['y'],
    temp_forecast.set_index('ds')['yhat']
])

# Recalculate regressors based on the combined y series
combined_y_df = combined_y.to_frame(name='y').reset_index()
combined_y_df['SMA_10'] = combined_y_df['y'].rolling(window=10).mean()
combined_y_df['SMA_30'] = combined_y_df['y'].rolling(window=30).mean()
combined_y_df['RSI'] = calc_rsi(combined_y_df['y'])
combined_y_df['ATR'] = df_proc['ATR'].iloc[-1]
combined_y_df['y_lag1'] = combined_y_df['y'].shift(1)
combined_y_df['y_lag2'] = combined_y_df['y'].shift(2)
combined_y_df['y_lag3'] = combined_y_df['y'].shift(3)

# Merge the forecasted regressors into the future dataframe
future_full = pd.merge(
    future_full,
    combined_y_df[['ds', 'SMA_10', 'SMA_30', 'RSI', 'ATR', 'y_lag1', 'y_lag2', 'y_lag3']],
    on='ds',
    how='left'
)

# Fill any remaining NaNs with the last known value from the historical data
for col in ['SMA_10', 'SMA_30', 'RSI', 'ATR', 'y_lag1', 'y_lag2', 'y_lag3']:
    if future_full[col].isnull().any():
        last_valid_value = df_proc[col].iloc[-1]
        future_full[col].fillna(last_valid_value, inplace=True)


forecast_full = model_final.predict(future_full)

plt.figure(figsize=(12, 6))
model_final.plot(forecast_full, ax=plt.gca())
plt.title(f'{tkr} Stock Price Forecast (Full Data)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid(True)
plt.show()

model_final.plot_components(forecast_full)
plt.show()

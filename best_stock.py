from datetime import datetime
import os
import tkinter as tk
from tkinter import messagebox
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import ta
import yfinance as yf

# 1. 解決繪圖中文亂碼
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 2. 自動判斷與下載台股資料 (上市 .TW / 上櫃 .TWO 防呆機制)
stock_input = (
    input("請輸入台股代碼 (例如 2330 或 0050): ").strip().upper()
)
stock_input = stock_input.replace(".TW", "").replace(".TWO", "")

stock_code = f"{stock_input}.TW"
today_str = datetime.today().strftime("%Y-%m-%d")

df = yf.download(stock_code, start="2018-01-01", end=today_str)

# 若上市抓不到資料，自動切換至上櫃市場
if df.empty:
  stock_code = f"{stock_input}.TWO"
  df = yf.download(stock_code, start="2018-01-01", end=today_str)

if df.empty:
  print("❌ 錯誤：找不到該股票資料，請檢查代碼。")
  exit()

# 處理 Yahoo Finance 多層欄位問題
if isinstance(df.columns, pd.MultiIndex):
  df.columns = df.columns.droplevel(1)
df = df.dropna()

# 3. 特徵工程 (技術指標與時序特徵)
df['MA5'] = df['Close'].rolling(5).mean()
df['MA20'] = df['Close'].rolling(20).mean()
df['MA60'] = df['Close'].rolling(60).mean()
df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['MACD'] = ta.trend.MACD(df['Close']).macd()
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility_20'] = df['Daily_Return'].rolling(20).std()

# 未來 5 日預估報酬率 (目標變數)
target_cols = []
for delay in range(1, 6):
  col_name = f'Next_Return_{delay}'
  df[col_name] = df['Daily_Return'].shift(-delay)
  target_cols.append(col_name)

latest_data = df.iloc[[-1]].copy()  # 攔截最新一天的資料做預測
df = df.dropna()

feature_cols = [
    'MA5',
    'MA20',
    'MA60',
    'RSI_14',
    'MACD',
    'Volume',
    'Daily_Return',
    'Volatility_20',
]
X = df[feature_cols]
y = df[target_cols]

# 4. 機器學習模型訓練 (隨機森林)
model = RandomForestRegressor(
    n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
)
model.fit(X, y)

# 5. 進行今日預測與價格指南計算
X_today = latest_data[feature_cols]
pred_returns = model.predict(X_today)[0]

today_close = latest_data['Close'].values[0]
today_volatility = latest_data['Volatility_20'].values[0]

# 計算預測價格軌跡
predicted_prices = []
curr_price = today_close
for r in pred_returns:
  curr_price *= 1 + r
  predicted_prices.append(curr_price)

total_week_return = (
    (np.prod([1 + r for r in pred_returns]) - 1) * 100
)
max_pred_price = max(predicted_prices)  # 預估目標停利價
stop_loss_price = today_close * (
    1 - 2 * today_volatility
)  # 2倍標準差動態停損價

# 6. 彈出機器人提醒小視窗 (GUI Alert)
root = tk.Tk()
root.withdraw()  # 隱藏主視窗

alert_title = f"🤖 {stock_code} 機器人實戰提醒"
message_text = (
    f"💰 今日收盤價：{today_close:.2f} 元\n"
    f"📊 未來 5 日預估總變動：{total_week_return:+.2f}%\n"
    f"🎯 建議波段停利價：{max_pred_price:.2f} 元\n"
    f"🛡️ 風控建議停損價：{stop_loss_price:.2f} 元\n"
    f"----------------------------------------\n"
)

tomorrow_return = pred_returns[0]
if tomorrow_return > 0 and total_week_return > 0:
  messagebox.showinfo(
      alert_title, message_text + "🟢 訊號：明日看漲且週線偏多，可考慮分批進場。"
  )
elif tomorrow_return <= 0 and total_week_return > 0:
  messagebox.showwarning(
      alert_title, message_text + "🟡 訊號：短線拉回但週線看漲，建議逢低布局。"
  )
else:
  messagebox.showerror(
      alert_title, message_text + "🔴 訊號：短線與週線趨勢偏弱，建議觀望避開。"
  )

root.destroy()

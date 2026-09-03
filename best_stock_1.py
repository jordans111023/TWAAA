from datetime import datetime, timedelta
import os
import tkinter as tk
from tkinter import messagebox
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

today_close = float(latest_data['Close'].values[0])
today_volatility = float(latest_data['Volatility_20'].values[0])

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
target_5d_price = predicted_prices[-1]  # 預估第5日目標價

# =========================================================================
# 6. 新增功能：歷史預測紀錄、新舊比對與「準確率驗證」
# =========================================================================
history_file = 'stock_predictions_history.csv'

# 比對與準確率文字初始化
comparison_text = "📜 歷史比對：尚無歷史紀錄"
accuracy_text = "🎯 歷史準確率：尚無已到期的驗證紀錄"

if os.path.exists(history_file):
  try:
    history_df = pd.read_csv(history_file)
    stock_history = history_df[
        history_df['Stock_Code'] == stock_code
    ].copy()

    if not stock_history.empty:
      stock_history = stock_history.sort_values(by='Predict_Date')

      # -----------------------------------------------------------------
      # Part A. 更新與計算過往預測的「實際結果與準確率」
      # -----------------------------------------------------------------
      updated_rows = False
      for idx, row in stock_history.iterrows():
        # 如果該筆紀錄尚未驗證，嘗試進行驗證
        if pd.isna(row.get('Actual_5D_Close')) or row.get(
            'Actual_5D_Close'
        ) == '':
          pred_date = datetime.strptime(row['Predict_Date'], '%Y-%m-%d')
          # 檢查預測日距今是否已過至少 7 天 (確保 5 個交易日已發生)
          if (datetime.now() - pred_date).days >= 7:
            # 抓取預測日之後的 10 天歷史資料 (避免休假日影響)
            eval_df = yf.download(
                stock_code,
                start=pred_date.strftime('%Y-%m-%d'),
                end=(pred_date + timedelta(days=12)).strftime('%Y-%m-%d'),
            )
            if isinstance(eval_df.columns, pd.MultiIndex):
              eval_df.columns = eval_df.columns.droplevel(1)

            # 確保有預測日當天 + 未來 5 個交易日 (至少 6 筆)
            if len(eval_df) >= 6:
              actual_close = float(eval_df['Close'].iloc[5])  # 第 5 個交易日
              start_close = float(row['Close_Price'])
              actual_return = (
                  (actual_close - start_close) / start_close
              ) * 100

              pred_return = row['Pred_5D_Return_%']

              # 方向判斷 (1 代表方向正確，0 代表錯誤)
              direction_correct = 1 if (pred_return * actual_return) > 0 else 0
              # 價格預估絕對百分比誤差
              price_error_pct = (
                  abs(actual_close - row['Target_5D_Price'])
                  / actual_close
                  * 100
              )

              # 回填數據
              stock_history.at[idx, 'Actual_5D_Close'] = round(actual_close, 2)
              stock_history.at[idx, 'Actual_5D_Return_%'] = round(
                  actual_return, 2
              )
              stock_history.at[idx, 'Direction_Correct'] = direction_correct
              stock_history.at[idx, 'Error_Pct'] = round(price_error_pct, 2)

              # 同步更新全域 DataFrame
              history_df.loc[
                  history_df['Predict_Date'] == row['Predict_Date'],
                  [
                      'Actual_5D_Close',
                      'Actual_5D_Return_%',
                      'Direction_Correct',
                      'Error_Pct',
                  ],
              ] = [
                  round(actual_close, 2),
                  round(actual_return, 2),
                  direction_correct,
                  round(price_error_pct, 2),
              ]
              updated_rows = True

      if updated_rows:
        history_df.to_csv(history_file, index=False)

      # -----------------------------------------------------------------
      # Part B. 計算歷史整體準確率統計
      # -----------------------------------------------------------------
      validated_history = stock_history.dropna(subset=['Direction_Correct'])
      if not validated_history.empty:
        acc_rate = validated_history['Direction_Correct'].mean() * 100
        avg_error = validated_history['Error_Pct'].mean()
        total_eval = len(validated_history)
        accuracy_text = (
            f"🎯 歷史表現 (已驗證 {total_eval} 次)：\n"
            f"   • 漲跌方向準確率：{acc_rate:.1f}%\n"
            f"   • 平均價格偏離度：{avg_error:.2f}%"
        )

      # -----------------------------------------------------------------
      # Part C. 與「上一次預測」進行新舊對比
      # -----------------------------------------------------------------
      last_record = stock_history.iloc[-1]
      last_date = last_record['Predict_Date']
      last_return = last_record['Pred_5D_Return_%']
      last_close = last_record['Close_Price']

      diff_return = total_week_return - last_return
      diff_close = round(today_close - last_close, 2)

      trend_change = "上修" if diff_return > 0 else "下修"
      comparison_text = (
          f"📜 歷史比對 (與 {last_date} 相比)：\n"
          f"   • 上次預估 5日變動：{last_return:+.2f}%\n"
          f"   • 本次預估趨勢：{trend_change} {abs(diff_return):.2f}%\n"
          f"   • 股價變化：{diff_close:+.2f} 元"
      )

  except Exception as e:
    print(f"讀取或驗證歷史紀錄時發生錯誤: {e}")

# 記錄本次預測至 CSV (包含預估第 5 日目標價欄位)
new_record = pd.DataFrame([{
    'Predict_Date': today_str,
    'Stock_Code': stock_code,
    'Close_Price': round(today_close, 2),
    'Pred_5D_Return_%': round(total_week_return, 2),
    'Target_5D_Price': round(target_5d_price, 2),
    'Target_Take_Profit': round(max_pred_price, 2),
    'Stop_Loss': round(stop_loss_price, 2),
    'Actual_5D_Close': np.nan,
    'Actual_5D_Return_%': np.nan,
    'Direction_Correct': np.nan,
    'Error_Pct': np.nan,
}])

if os.path.exists(history_file):
  new_record.to_csv(history_file, mode='a', header=False, index=False)
else:
  new_record.to_csv(history_file, mode='w', header=True, index=False)

# =========================================================================
# 7. 彈出機器人提醒小視窗 (GUI Alert)
# =========================================================================
root = tk.Tk()
root.withdraw()  # 隱藏主視窗

alert_title = f"🤖 {stock_code} 機器人實戰提醒"
message_text = (
    f"💰 今日收盤價：{today_close:.2f} 元\n"
    f"📊 未來 5 日預估總變動：{total_week_return:+.2f}%\n"
    f"🎯 預估 5 日後目標價：{target_5d_price:.2f} 元\n"
    f"🎯 建議波段停利價：{max_pred_price:.2f} 元\n"
    f"🛡️ 風控建議停損價：{stop_loss_price:.2f} 元\n"
    f"----------------------------------------\n"
    f"{comparison_text}\n"
    f"----------------------------------------\n"
    f"{accuracy_text}\n"
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

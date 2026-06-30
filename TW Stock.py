import os
import yfinance as yf
import pandas as pd
import ta
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib

# 📌 解決中文亂碼
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================================
# 🔥 【動態輸入股票代碼與防呆處理】—— 升級版支援英文字母代碼
# ==========================================================
stock_input = input("請輸入台灣股票/ETF代碼 (例如 2330 或 0050): ").strip().upper()

# 如果使用者自己輸入了後綴，先將其剝除，統一由程式後續判斷
if stock_input.endswith(".TW"):
    stock_input = stock_input.replace(".TW", "")
elif stock_input.endswith(".TWO"):
    stock_input = stock_input.replace(".TWO", "")

# 預設先嘗試上市市場 (.TW)
stock_code = f"{stock_input}.TW"
print(f"\n🚀 正在下載 {stock_code} 的歷史數據...")

# 1. 下載資料 (預設先抓上市)
df = yf.download(stock_code, start="2015-01-01", end="2026-07-01")

# 防呆機制：如果上市抓不到，自動切換到上櫃市場 (.TWO)
if df.empty:
    alternative_code = f"{stock_input}.TWO"
    print(f"⚠️ 在上市市場找不到該股票，正在嘗試上櫃市場 {alternative_code}...")
    df = yf.download(alternative_code, start="2015-01-01", end="2026-07-01")
    if not df.empty:
        stock_code = alternative_code

# 如果兩個市場都抓不到，才報錯退出
if df.empty:
    print(f"❌ 錯誤：找不到代碼 {stock_input} 的資料，請檢查代碼是否輸入正確（台灣上市櫃皆未尋獲）。")
    exit()

# 處理 Yahoo Finance 新版多層欄位問題
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
df = df.dropna()

# 2. 特徵工程
df['MA5'] = df['Close'].rolling(5).mean()
df['MA20'] = df['Close'].rolling(20).mean()
df['MA60'] = df['Close'].rolling(60).mean()
df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['MACD'] = ta.trend.MACD(df['Close']).macd()
df['Daily_Return'] = df['Close'].pct_change()

# 3. 進階特徵工程
df['RSI_lag1'] = df['RSI_14'].shift(1)
df['RSI_lag2'] = df['RSI_14'].shift(2)
df['Return_lag1'] = df['Daily_Return'].shift(1)
df['Volatility_20'] = df['Daily_Return'].rolling(20).std()
df['Weekday'] = df.index.weekday

# ==========================================================
# 🔥 【將目標設定為未來 5 個交易日的報酬率】
# ==========================================================
target_cols = []
for delay in range(1, 6):
    col_name = f'Next_Return_{delay}'
    df[col_name] = df['Daily_Return'].shift(-delay)
    target_cols.append(col_name)

# 💡 攔截最後一天（今天）作為預測未來一週的依據
latest_data = df.iloc[[-1]].copy()

# 清洗所有因計算與 shift 產生的空值
df = df.dropna()

if len(df) <= 1050:
    print(f"❌ 錯誤：該股票有效歷史數據過少 ({len(df)} 筆)，不足以進行滾動訓練。")
    exit()

# 5. 定義特徵群組與多目標 y
feature_cols = [
    'MA5', 'MA20', 'MA60', 'RSI_14', 'MACD', 'Volume', 'Daily_Return',
    'RSI_lag1', 'RSI_lag2', 'Return_lag1', 'Volatility_20', 'Weekday'
]
X = df[feature_cols]
y = df[target_cols]

# ==========================================================
# 🔥 【滾動式自我學習與回測機制】
# ==========================================================
train_window = 1000
predict_step = 20

y_test_list = []
y_pred_list = []

print(f"--- 啟動 {stock_code} 動態多目標自我學習滾動訓練 ---")
for i in range(train_window, len(df), predict_step):
    X_train_roll = X.iloc[i - train_window: i]
    y_train_roll = y.iloc[i - train_window: i]

    end_idx = min(i + predict_step, len(df))
    X_test_roll = X.iloc[i:end_idx]
    y_test_roll = y.iloc[i:end_idx]

    if len(X_test_roll) == 0:
        break

    model_roll = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    model_roll.fit(X_train_roll, y_train_roll)

    pred_roll = model_roll.predict(X_test_roll)

    y_test_list.extend(y_test_roll.values)
    y_pred_list.extend(pred_roll)

y_test_final = np.array(y_test_list)
y_pred_final = np.array(y_pred_list)

# 6. 策略回測與視覺化
strategy_returns = y_test_final[:, 0] * (y_pred_final[:, 0] > 0)
market_cumulative = (1 + y_test_final[:, 0]).cumprod() - 1
strategy_cumulative = (1 + strategy_returns).cumprod() - 1

plt.figure(figsize=(10, 5))
plt.plot(market_cumulative, label=f"Market (單純持有 {stock_code})", color='blue', alpha=0.7)
plt.plot(strategy_cumulative, label="Strategy (滾動多目標策略)", color='red', linewidth=2)
plt.title(f"{stock_code} 機器學習-動態多目標滾動回測績效")
plt.ylabel("累積報酬率")
plt.xlabel("滾動預測交易天數")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 7. 部署最終進化版模型
if not os.path.exists("models"):
    os.makedirs("models")
model_filename = f"models/{stock_code}_multi_day_model.pkl"
joblib.dump(model_roll, model_filename)
print(f"🎉 成果：最終進化版多日預測模型已儲存至 {model_filename}")

# ==========================================================
# 🔥 🔥 🔥 【明日訊號、未來一週趨勢與適合賣出價預測】 🔥 🔥 🔥
# ==========================================================
print("\n" + "=" * 50)
print(f"🔮 {stock_code} 最終進化模型 —— 明日實戰訊號與目標價建議 🔮")
print("=" * 50)

X_today = latest_data[feature_cols]

if X_today.isnull().values.any():
    print("⚠️ 警告：今日技術指標數據不完整，無法進行趨勢預測。")
else:
    # 同時預測未來 5 個交易日的報酬率
    weekly_pred_returns = model_roll.predict(X_today)[0]

    today_date = latest_data.index[0].strftime('%Y-%m-%d')
    today_close = latest_data['Close'].values[0]
    today_volatility = latest_data['Volatility_20'].values[0]  # 獲取近期波動度

    print(f"📅 資料基準日（今天）：{today_date}")
    print(f"💰 今日實際收盤價：{today_close:.2f} 元\n")
    print("📋 【未來 5 個交易日預測軌跡】:")
    print(f"{'交易日':<8}{'預估單日漲跌':<14}{'預估收盤價':<12}")
    print("-" * 35)

    current_price = today_close
    cumulative_return = 1.0
    predicted_prices = []  # 用於紀錄每天的預測價，找出最高點

    for day_idx, r_pred in enumerate(weekly_pred_returns, start=1):
        current_price = current_price * (1 + r_pred)
        cumulative_return *= (1 + r_pred)
        predicted_prices.append(current_price)
        print(f"第 {day_idx} 天   {r_pred * 100:+.2f}%         {current_price:.2f} 元")

    print("-" * 35)
    total_week_return = (cumulative_return - 1) * 100
    print(f"📊 未來一週累積漲跌幅預測：{total_week_return:+.2f}%")
    print("-" * 50)

    # 🔥 【新增：動態計算停利與停損價格】
    max_pred_price = max(predicted_prices)  # 未來一週模型預測的最高點
    # 停損價採用當前價格向下扣除 2 倍的 20日歷史波動金額 (2 * 今日價格 * 日波動率)
    stop_loss_price = today_close * (1 - 2 * today_volatility)

    print("🎯 【實戰交易價格指南】:")
    if total_week_return > 0:
        print(f"📈 適合賣出價格 (波段停利點)：{max_pred_price:.2f} 元 (預期波段高點，觸及可分批獲利)")
    else:
        # 如果模型看空，建議以明日短線反彈高點或今日收盤價作為隨時離場參考
        print(
            f"📉 適合賣出價格 (反彈分批解套)：{max(predicted_prices[0], today_close):.2f} 元 (趨勢偏弱，建議縮短操作週期)")

    print(f"🛡️ 風控防守價格 (歷史波動停損)：{stop_loss_price:.2f} 元 (跌破此價代表趨勢破壞，應果斷停損)")
    print("-" * 50)

    # 綜合策略判斷
    tomorrow_return = weekly_pred_returns[0]
    if tomorrow_return > 0 and total_week_return > 0:
        print(f"🟢 買入訊號：明日看漲 ({tomorrow_return * 100:+.2f}%) 且一週格局偏多，【可以買入】。")
    elif tomorrow_return <= 0 and total_week_return > 0:
        print(f"🟡 觀望/低吸訊號：明日可能短線拉回，但週線趨勢看漲，【建議分批進場或逢低布局】。")
    elif tomorrow_return > 0 and total_week_return <= 0:
        print(f"🟠 短線訊號：明日雖有反彈機會，但中期一週趨勢偏弱，【僅限極短線交易，不宜波段持有】。")
    else:
        print(f"🔴 空倉訊號：短線與中期週線均看空，【強烈建議觀望避開】。")
print("=" * 50)
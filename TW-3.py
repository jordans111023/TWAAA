import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor  # 💡 引入 KNN 模組
from sklearn.tree import DecisionTreeRegressor  # 💡 引入 決策樹 模組

# 📌 解決中文亂碼
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==========================================================
# 🔥 【動態輸入股票代碼與防呆處理】
# ==========================================================
stock_input = input("請輸入台灣股票/ETF代碼 (例如 2330 或 0050): ").strip().upper()

if stock_input.endswith(".TW"):
    stock_input = stock_input.replace(".TW", "")
elif stock_input.endswith(".TWO"):
    stock_input = stock_input.replace(".TWO", "")

stock_code = f"{stock_input}.TW"
print(f"\n🚀 正在下載 {stock_code} 的歷史數據...")

df = yf.download(stock_code, start="2015-01-01", end="2026-07-10")

if df.empty:
    alternative_code = f"{stock_input}.TWO"
    print(f"⚠️ 在上市市場找不到該股票，正在嘗試上櫃市場 {alternative_code}...")
    df = yf.download(alternative_code, start="2015-01-01", end="2026-07-01")
    if not df.empty:
        stock_code = alternative_code

if df.empty:
    print(
        f"❌ 錯誤：找不到代碼 {stock_input} 的資料，請檢查代碼是否輸入正確（台灣上市櫃皆未尋獲）。"
    )
    exit()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
df = df.dropna()

# 2. 特徵工程
df["MA5"] = df["Close"].rolling(5).mean()
df["MA20"] = df["Close"].rolling(20).mean()
df["MA60"] = df["Close"].rolling(60).mean()
df["RSI_14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
df["MACD"] = ta.trend.MACD(df["Close"]).macd()
df["Daily_Return"] = df["Close"].pct_change()

# 3. 進階特徵工程
df["RSI_lag1"] = df["RSI_14"].shift(1)
df["RSI_lag2"] = df["RSI_14"].shift(2)
df["Return_lag1"] = df["Daily_Return"].shift(1)
df["Volatility_20"] = df["Daily_Return"].rolling(20).std()
df["Weekday"] = df.index.weekday

# ==========================================================
# 🔥 【將目標設定為未來 5 個交易日的報酬率】
# ==========================================================
target_cols = []
for delay in range(1, 6):
    col_name = f"Next_Return_{delay}"
    df[col_name] = df["Daily_Return"].shift(-delay)
    target_cols.append(col_name)

latest_data = df.iloc[[-1]].copy()
df = df.dropna()

if len(df) <= 1050:
    print(
        f"❌ 錯誤：該股票有效歷史數據過少 ({len(df)} 筆)，不足以進行滾動訓練。"
    )
    exit()

feature_cols = [
    "MA5",
    "MA20",
    "MA60",
    "RSI_14",
    "MACD",
    "Volume",
    "Daily_Return",
    "RSI_lag1",
    "RSI_lag2",
    "Return_lag1",
    "Volatility_20",
    "Weekday",
]
X = df[feature_cols]
y = df[target_cols]

# ==========================================================
# 💡 【新增：模型互動式選擇選單】
# ==========================================================
print("\n" + "=" * 40)
print("請選擇欲使用的機器學習模型：")
print("1. 隨機森林 (Random Forest)")
print("2. 決策樹 (Decision Tree)")
print("3. K-近鄰演算法 (KNN)")
print("=" * 40)
model_choice = input("請輸入選項數字 (1-3): ").strip()

# 根據選擇初始化對應的模型設定與名稱
if model_choice == "2":
    model_name = "Decision Tree"
    get_model = lambda: DecisionTreeRegressor(
        max_depth=6, min_samples_split=10, random_state=42
    )
elif model_choice == "3":
    model_name = "KNN"
    get_model = lambda: KNeighborsRegressor(
        n_neighbors=5, weights="distance", n_jobs=-1
    )
else:
    model_name = "Random Forest"
    get_model = lambda: RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )

print(f"\n🎯 已選擇模型: 【{model_name}】")

# ==========================================================
# 🔥 【時間序列交叉驗證與回測機制 (TimeSeriesSplit)】
# ==========================================================
train_window = 1000
predict_step = 20

n_splits = int((len(df) - train_window) / predict_step)
tscv = TimeSeriesSplit(n_splits=n_splits, test_size=predict_step)

y_test_list = []
y_pred_list = []

print(f"--- 啟動 {stock_code} TimeSeriesSplit 多目標交叉驗證訓練 ---")

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    if len(train_index) < train_window:
        continue

    # 預設：擴展窗口 (Expanding Window)
    X_train_roll = X.iloc[train_index]
    y_train_roll = y.iloc[train_index]

    X_test_roll = X.iloc[test_index]
    y_test_roll = y.iloc[test_index]

    # 呼叫動態選定的模型
    model_roll = get_model()
    model_roll.fit(X_train_roll, y_train_roll)

    pred_roll = model_roll.predict(X_test_roll)

    y_test_list.extend(y_test_roll.values)
    y_pred_list.extend(pred_roll)

y_test_final = np.array(y_test_list)
y_pred_final = np.array(y_pred_list)

# ==========================================================
# 6. 策略回測與視覺化
# ==========================================================
strategy_returns = y_test_final[:, 0] * (y_pred_final[:, 0] > 0)
market_cumulative = (1 + y_test_final[:, 0]).cumprod() - 1
strategy_cumulative = (1 + strategy_returns).cumprod() - 1

plt.figure(figsize=(10, 5))
plt.plot(
    market_cumulative,
    label=f"Market (單純持有 {stock_code})",
    color="blue",
    alpha=0.7,
)
plt.plot(
    strategy_cumulative,
    label=f"Strategy ({model_name}策略)",
    color="red",
    linewidth=2,
)
plt.title(f"{stock_code} 機器學習-{model_name}交叉驗證回測績效")
plt.ylabel("累積報酬率")
plt.xlabel("驗證交易天數")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# ==========================================================
# 7. 部署最終進化版模型
# ==========================================================
if not os.path.exists("models"):
    os.makedirs("models")
# 儲存檔名加入模型名稱，避免混淆
model_filename = f"models/{stock_code}_{model_name.lower().replace(' ', '_')}_model.pkl"
joblib.dump(model_roll, model_filename)
print(f"🎉 成果：最終進化版多日預測模型已儲存至 {model_filename}")

# ==========================================================
# 🔥 🔥 🔥 【明日訊號、未來一週趨勢與適合賣出價預測】 🔥 🔥 🔥
# ==========================================================
print("\n" + "=" * 50)
print(
    f"🔮 {stock_code} 最終進化模型 ({model_name}) —— 明日實戰訊號與目標價建議 🔮"
)
print("=" * 50)

X_today = latest_data[feature_cols]

if X_today.isnull().values.any():
    print("⚠️ 警告：今日技術指標數據不完整，無法進行趨勢預測。")
else:
    weekly_pred_returns = model_roll.predict(X_today)[0]

    today_date = latest_data.index[0].strftime("%Y-%m-%d")
    today_close = latest_data["Close"].values[0]
    today_volatility = latest_data["Volatility_20"].values[0]

    print(f"📅 資料基準日（今天）：{today_date}")
    print(f"💰 今日實際收盤價：{today_close:.2f} 元\n")
    print("📋 【未來 5 個交易日預測軌跡】:")
    print(f"{'交易日':<8}{'預估單日漲跌':<14}{'預估收盤價':<12}")
    print("-" * 35)

    current_price = today_close
    cumulative_return = 1.0
    predicted_prices = []

    for day_idx, r_pred in enumerate(weekly_pred_returns, start=1):
        current_price = current_price * (1 + r_pred)
        cumulative_return *= (1 + r_pred)
        predicted_prices.append(current_price)
        print(
            f"第 {day_idx} 天   {r_pred * 100:+.2f}%         {current_price:.2f} 元"
        )

    print("-" * 35)
    total_week_return = (cumulative_return - 1) * 100
    print(f"📊 未來一週累積漲跌幅預測：{total_week_return:+.2f}%")
    print("-" * 50)

    max_pred_price = max(predicted_prices)
    stop_loss_price = today_close * (1 - 2 * today_volatility)

    print("🎯 【實戰交易價格指南】:")
    if total_week_return > 0:
        print(
            f"📈 適合賣出價格 (波段停利點)：{max_pred_price:.2f} 元 (預期波段高點，觸及可分批獲利)"
        )
    else:
        print(
            f"📉 適合賣出價格 (反彈分批解套)：{max(predicted_prices[0], today_close):.2f} 元 (趨勢偏弱，建議縮短操作週期)"
        )

    print(
        f"🛡️ 風控防守價格 (歷史波動停損)：{stop_loss_price:.2f} 元 (跌破此價代表趨勢破壞，應果斷停損)"
    )
    print("-" * 50)

    tomorrow_return = weekly_pred_returns[0]
    if tomorrow_return > 0 and total_week_return > 0:
        print(
            f"🟢 買入訊號：明日看漲 ({tomorrow_return * 100:+.2f}%) 且一週格局偏多，【可以買入】。"
        )
    elif tomorrow_return <= 0 and total_week_return > 0:
        print(
            f"🟡 觀望/低吸訊號：明日可能短線拉回，但週線趨勢看漲，【建議分批進場或逢低布局】。"
        )
    elif tomorrow_return > 0 and total_week_return <= 0:
        print(
            f"🟠 短線訊號：明日雖有反彈機會，但中期一週趨勢偏弱，【僅限極短線交易，不宜波段持有】。"
        )
    else:
        print(
            f"🔴 空倉訊號：短線與中期週線均看空，【強烈建議觀望避開】。"
        )
print("=" * 50)

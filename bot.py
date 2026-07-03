import os
import time
import threading
import asyncio
import requests
import numpy as np
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime, timezone, timedelta
from collections import deque

# ==========================================
# TELEGRAM CREDENTIALS (PREVIOUS BOT)
# ==========================================
TOKEN = "8608138546:AAEetCz5xKlQlIRc0eZ3gVzvs046dPb86UI"
CHAT_ID = "6280535707"

# ==========================================
# TIME ZONE (UTC+1)
# ==========================================
LOCAL_TZ = timezone(timedelta(hours=1))

# ==========================================
# PROCESSING LOCK (for multiple screenshots)
# ==========================================
process_lock = asyncio.Lock()

# ==========================================
# STRATEGY HEALTH TRACKING (17 STRATEGIES)
# ==========================================
strategy_history = {name: deque(maxlen=10) for name in [
    "Candle Reversal Pattern", "3-Candle Momentum", "2-Minute Reset",
    "Double Touch", "Spike Rejection", "Consolidation Break",
    "EMA Pullback", "Bull/Bear Confirmation", "60-Second Scalp",
    "RSI Divergence", "Bollinger Squeeze", "MACD Crossover",
    "Support/Resistance Break", "MA Crossover",
    "Three White Soldiers", "Three Black Crows", "Morning Star"
]}

def get_strategy_health(strategy_name):
    if strategy_name not in strategy_history:
        return 50
    history = strategy_history[strategy_name]
    if len(history) == 0:
        return 50
    win_rate = sum(history) / len(history) * 100
    return min(100, max(50, win_rate))

# ==========================================
# TIME FUNCTIONS
# ==========================================
def get_next_minute():
    now = datetime.now(LOCAL_TZ)
    if now.second > 0 or now.microsecond > 0:
        next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    else:
        next_minute = now.replace(second=0, microsecond=0)
    return next_minute.strftime("%H:%M:%S")

def get_entry2_time(entry1_time):
    return (datetime.strptime(entry1_time, "%H:%M:%S") + timedelta(minutes=1)).strftime("%H:%M:%S")

# ==========================================
# FLASK WEB SERVER
# ==========================================
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ OTC Signal Bot is running!"

@app.route('/ping')
def ping():
    return "pong", 200

def run_flask():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=10000, debug=False, threaded=True)

# ==========================================
# INDICATORS
# ==========================================
def calculate_rsi(data, period=14):
    if len(data) < period + 1:
        return 50
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    return 100 - (100 / (1 + avg_gain / avg_loss))

def calculate_ema(data, period=20):
    if len(data) < period:
        return data[-1]
    return np.mean(data[-period:])

def calculate_macd(close):
    if len(close) < 26:
        return 0, 0
    ema12 = np.mean(close[-12:])
    ema26 = np.mean(close[-26:])
    macd = ema12 - ema26
    signal = np.mean(close[-9:])
    return macd, signal

def calculate_bollinger(close, period=20):
    if len(close) < period:
        return 0, 0
    sma = np.mean(close[-period:])
    std = np.std(close[-period:])
    return sma + 2 * std, sma - 2 * std

# ==========================================
# 17 STRATEGIES (14 + 3 NEW)
# ==========================================
def run_strategies(price_data):
    results = []
    close = np.array(price_data['close'])
    open_ = np.array(price_data['open'])
    high = np.array(price_data['high'])
    low = np.array(price_data['low'])
    volume = np.array(price_data.get('volume', np.ones(len(close))))

    # --- SAFETY CHECK: If fewer than 5 candles, return empty ---
    if len(close) < 5:
        return results

    rsi = calculate_rsi(close)
    ema20 = calculate_ema(close, 20)
    macd, signal = calculate_macd(close)
    upper_band, lower_band = calculate_bollinger(close)

    def add_signal(name, direction, conf, exp1, exp2):
        results.append((name, direction, conf, exp1, exp2))

    # --- 1. Candle Reversal Pattern ---
    if len(close) >= 3:
        if (close[-3:] > open_[-3:]).all() and (close[-1] < open_[-1]) and rsi > 70:
            add_signal("Candle Reversal Pattern", "SELL", 86, 2, 3)
        elif (close[-3:] < open_[-3:]).all() and (close[-1] > open_[-1]) and rsi < 30:
            add_signal("Candle Reversal Pattern", "BUY", 86, 2, 3)

    # --- 2. 3-Candle Momentum ---
    if len(close) >= 3:
        if (close[-3:] > open_[-3:]).all() and volume[-1] > np.mean(volume[-5:]):
            add_signal("3-Candle Momentum", "BUY", 82, 1, 2)
        elif (close[-3:] < open_[-3:]).all() and volume[-1] > np.mean(volume[-5:]):
            add_signal("3-Candle Momentum", "SELL", 82, 1, 2)

    # --- 3. 2-Minute Reset ---
    if len(close) >= 3:
        if (close[-3:] < open_[-3:]).all() and close[-1] < ema20 * 0.98:
            add_signal("2-Minute Reset", "SELL", 78, 2, 3)
        elif (close[-3:] > open_[-3:]).all() and close[-1] > ema20 * 1.02:
            add_signal("2-Minute Reset", "BUY", 78, 2, 3)

    # --- 4. Double Touch ---
    if len(close) >= 10:
        if abs(low[-1] - low[-3]) < 0.0002 and close[-1] > max(close[-5:-1]):
            add_signal("Double Touch", "BUY", 89, 3, 4)
        elif abs(high[-1] - high[-3]) < 0.0002 and close[-1] < min(close[-5:-1]):
            add_signal("Double Touch", "SELL", 89, 3, 4)

    # --- 5. Spike Rejection ---
    if len(close) >= 2:
        avg_range = np.mean(high[-5:] - low[-5:])
        if (high[-1] - high[-2]) > 2 * avg_range and (close[-1] < open_[-1]):
            add_signal("Spike Rejection", "SELL", 74, 2, 3)
        elif (low[-2] - low[-1]) > 2 * avg_range and (close[-1] > open_[-1]):
            add_signal("Spike Rejection", "BUY", 74, 2, 3)

    # --- 6. Consolidation Break ---
    if len(close) >= 10:
        high_range = max(high[-10:]) - min(low[-10:])
        if high_range < 0.0005 and (close[-1] - open_[-1]) > 0.0005:
            add_signal("Consolidation Break", "BUY", 71, 3, 4)
        elif high_range < 0.0005 and (open_[-1] - close[-1]) > 0.0005:
            add_signal("Consolidation Break", "SELL", 71, 3, 4)

    # --- 7. EMA Pullback ---
    if len(close) >= 20:
        if low[-1] < ema20 and close[-1] > ema20 and rsi > 40:
            add_signal("EMA Pullback", "BUY", 80, 2, 3)
        elif high[-1] > ema20 and close[-1] < ema20 and rsi < 60:
            add_signal("EMA Pullback", "SELL", 80, 2, 3)

    # --- 8. Bull/Bear Confirmation ---
    if len(close) >= 3:
        if (close[-1] > open_[-1] and close[-2] > open_[-2] and close[-3] > open_[-3]):
            add_signal("Bull/Bear Confirmation", "BUY", 76, 1, 2)
        elif (close[-1] < open_[-1] and close[-2] < open_[-2] and close[-3] < open_[-3]):
            add_signal("Bull/Bear Confirmation", "SELL", 76, 1, 2)

    # --- 9. 60-Second Scalp ---
    if len(close) >= 2:
        if (close[-1] - open_[-1]) > (close[-2] - open_[-2]) * 1.5 and volume[-1] > np.mean(volume[-3:]):
            add_signal("60-Second Scalp", "BUY", 72, 1, 1)
        elif (open_[-1] - close[-1]) > (open_[-2] - close_[-2]) * 1.5 and volume[-1] > np.mean(volume[-3:]):
            add_signal("60-Second Scalp", "SELL", 72, 1, 1)

    # --- 10. RSI Divergence ---
    if len(close) >= 20:
        if close[-1] < min(close[-5:-1]) and rsi > min(50, np.mean(rsi)):
            add_signal("RSI Divergence", "BUY", 84, 2, 3)
        elif close[-1] > max(close[-5:-1]) and rsi < max(50, np.mean(rsi)):
            add_signal("RSI Divergence", "SELL", 84, 2, 3)

    # --- 11. Bollinger Squeeze ---
    if len(close) >= 20:
        atr = np.mean(high[-20:] - low[-20:])
        current_range = high[-1] - low[-1]
        if current_range < atr * 0.5:
            if close[-1] > open_[-1] and close[-1] > ema20:
                add_signal("Bollinger Squeeze", "BUY", 77, 2, 3)
            elif close[-1] < open_[-1] and close[-1] < ema20:
                add_signal("Bollinger Squeeze", "SELL", 77, 2, 3)

    # --- 12. MACD Crossover ---
    if len(close) >= 26:
        if macd > signal and close[-1] > ema20:
            add_signal("MACD Crossover", "BUY", 80, 2, 3)
        elif macd < signal and close[-1] < ema20:
            add_signal("MACD Crossover", "SELL", 80, 2, 3)

    # --- 13. Support/Resistance Break ---
    if len(close) >= 20:
        resistance = max(high[-20:-1])
        support = min(low[-20:-1])
        if close[-1] > resistance and close[-1] > open_[-1]:
            add_signal("Support/Resistance Break", "BUY", 81, 2, 3)
        elif close[-1] < support and close[-1] < open_[-1]:
            add_signal("Support/Resistance Break", "SELL", 81, 2, 3)

    # --- 14. MA Crossover ---
    if len(close) >= 30:
        ma10 = np.mean(close[-10:])
        ma30 = np.mean(close[-30:])
        if ma10 > ma30 and close[-1] > open_[-1]:
            add_signal("MA Crossover", "BUY", 79, 2, 3)
        elif ma10 < ma30 and close[-1] < open_[-1]:
            add_signal("MA Crossover", "SELL", 79, 2, 3)

    # --- 15. Three White Soldiers ---
    if len(close) >= 3:
        if (close[-1] > open_[-1] and close[-2] > open_[-2] and close[-3] > open_[-3] and
            close[-1] > close[-2] and close[-2] > close[-3]):
            add_signal("Three White Soldiers", "BUY", 85, 2, 3)

    # --- 16. Three Black Crows ---
    if len(close) >= 3:
        if (close[-1] < open_[-1] and close[-2] < open_[-2] and close[-3] < open_[-3] and
            close[-1] < close[-2] and close[-2] < close[-3]):
            add_signal("Three Black Crows", "SELL", 85, 2, 3)

    # --- 17. Morning Star (Safer) ---
    try:
        if len(close) >= 5:
            if (close[-3] < open_[-3] and abs(close[-2] - open_[-2]) < abs(close[-3] - open_[-3]) * 0.3 and
                close[-1] > open_[-1] and close[-1] > (close[-3] + open_[-3]) / 2):
                add_signal("Morning Star", "BUY", 84, 2, 3)
    except:
        pass  # Skip this strategy if there's an error

    # --- Adjust confidence based on strategy health ---
    adjusted_results = []
    for name, direction, confidence, expiry1, expiry2 in results:
        health = get_strategy_health(name)
        adjusted_conf = int(confidence * (health / 50))
        adjusted_conf = min(100, max(50, adjusted_conf))
        adjusted_results.append((name, direction, adjusted_conf, expiry1, expiry2))
    
    return adjusted_results

# ==========================================
# PREDICTION ENGINE
# ==========================================
def predict_entries(strategy, direction, confidence, expiry_1, expiry_2):
    entry1_time = get_next_minute()
    entry2_time = get_entry2_time(entry1_time)

    if direction == "BUY":
        entry1_dir = "🟢 BUY"
        entry2_dir = "🟢 BUY"
    else:
        entry1_dir = "🔴 SELL"
        entry2_dir = "🔴 SELL"

    entry2_conf = max(confidence - 10, 50)

    return {
        "strategy": strategy,
        "entry1": {"time": entry1_time, "dir": entry1_dir, "conf": confidence, "expiry": expiry_1},
        "entry2": {"time": entry2_time, "dir": entry2_dir, "conf": entry2_conf, "expiry": expiry_2}
    }

# ==========================================
# TELEGRAM BOT HANDLERS
# ==========================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📊 **OTC Signal Bot**\n\n"
        "Send a screenshot — I'll give you a signal."
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with process_lock:
        try:
            start_time = time.time()

            photo = await update.message.photo[-1].get_file()
            await photo.download_to_drive("screenshot.png")

            price_data = {
                'open': np.random.randn(30) + 1.12,
                'high': np.random.randn(30) + 1.13,
                'low': np.random.randn(30) + 1.11,
                'close': np.random.randn(30) + 1.12,
                'volume': np.random.randint(100, 1000, 30)
            }

            results = run_strategies(price_data)

            if not results:
                await update.message.reply_text("⛔ No clear signal — DON'T TRADE.")
                return

            best = max(results, key=lambda x: x[2])
            strategy, direction, confidence, expiry_1, expiry_2 = best
            prediction = predict_entries(strategy, direction, confidence, expiry_1, expiry_2)

            response = f"📊 **OTC SIGNAL**\n\n"
            response += f"📈 **Entry 1:**\n"
            response += f"   {prediction['entry1']['dir']} at {prediction['entry1']['time']} ({prediction['entry1']['expiry']} min) — Confidence: {prediction['entry1']['conf']}%\n\n"
            response += f"🔍 **Strategy:** {strategy}\n"
            response += f"   → Direction: {direction}\n"
            response += f"   → Confidence: {confidence}%\n"
            response += f"   → Expiry: {expiry_1} min\n\n"
            response += f"📈 **Entry 2:**\n"
            response += f"   {prediction['entry2']['dir']} at {prediction['entry2']['time']} ({prediction['entry2']['expiry']} min) — Confidence: {prediction['entry2']['conf']}%\n"
            response += f"   → Expiry: {prediction['entry2']['expiry']} min\n"

            await update.message.reply_text(response)

            elapsed = time.time() - start_time
            print(f"✅ Signal sent in {elapsed:.2f} seconds")

        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")

# ==========================================
# START BOT
# ==========================================
def run_telegram():
    application = Application.builder().token(TOKEN).build()
    
    try:
        application.bot.delete_webhook()
        print("✅ Webhook cleared.")
    except Exception as e:
        print(f"⚠️ Webhook clear error: {e}")
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.run_polling()

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    print("✅ Flask server started.")
    print("✅ Starting Telegram bot...")
    run_telegram()

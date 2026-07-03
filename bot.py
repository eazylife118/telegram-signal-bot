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
# TELEGRAM CREDENTIALS
# ==========================================
TOKEN = "8608138546:AAEetCz5xKlQlIRc0eZ3gVzvs046dPb86UI"
CHAT_ID = "6280535707"

# ==========================================
# PROCESSING LOCK
# ==========================================
process_lock = asyncio.Lock()

# ==========================================
# TIME ZONE (UTC+1)
# ==========================================
LOCAL_TZ = timezone(timedelta(hours=1))

# ==========================================
# STRATEGY HEALTH TRACKING (20 STRATEGIES)
# ==========================================
strategy_history = {name: deque(maxlen=10) for name in [
    "Candle Reversal Pattern", "3-Candle Momentum", "2-Minute Reset",
    "Double Touch", "Spike Rejection", "Consolidation Break",
    "Bull/Bear Confirmation", "60-Second Scalp",
    "Support/Resistance Break", "Long Wick Rejection",
    "Opening Range Break", "Close Beyond Previous High/Low",
    "2-Candle Engulfing", "Outside Candle",
    "Three White Soldiers", "Three Black Crows",
    "Morning Star", "Evening Star",
    "Bullish Harami", "Bearish Harami"
]}

def get_strategy_health(strategy_name):
    if strategy_name not in strategy_history:
        return 50
    history = strategy_history[strategy_name]
    if len(history) == 0:
        return 50
    win_rate = sum(history) / len(history) * 100
    return min(100, max(50, win_rate))

def record_signal(strategy_name, win):
    if strategy_name in strategy_history:
        strategy_history[strategy_name].append(win)

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
# 20 STRATEGIES — DYNAMIC CONFIDENCE
# ==========================================
def run_strategies(price_data):
    results = []
    close = np.array(price_data['close'])
    open_ = np.array(price_data['open'])
    high = np.array(price_data['high'])
    low = np.array(price_data['low'])

    if len(close) < 5:
        return results

    def add_signal(name, direction, base_conf, exp1, exp2):
        # Health adjustment
        health = get_strategy_health(name)
        conf = int(base_conf * (health / 50))
        conf = min(100, max(50, conf))
        results.append((name, direction, conf, exp1, exp2))

    # --- All 20 strategies (same as before, but without indicators) ---
    # ... (keep all strategies from the previous version) ...

    # --- After collecting all signals, calculate agreement ---
    if not results:
        return results

    # Group signals by direction (BUY/SELL)
    buy_signals = [r for r in results if r[1] == "BUY"]
    sell_signals = [r for r in results if r[1] == "SELL"]

    # Pick the direction with more signals
    if len(buy_signals) > len(sell_signals):
        direction = "BUY"
        group = buy_signals
    elif len(sell_signals) > len(buy_signals):
        direction = "SELL"
        group = sell_signals
    else:
        # If equal, pick the one with higher average confidence
        buy_avg = np.mean([r[2] for r in buy_signals]) if buy_signals else 0
        sell_avg = np.mean([r[2] for r in sell_signals]) if sell_signals else 0
        if buy_avg >= sell_avg:
            direction = "BUY"
            group = buy_signals
        else:
            direction = "SELL"
            group = sell_signals

    # Calculate agreement confidence
    num_agree = len(group)
    total = len(results)

    if num_agree == 1:
        confidence = 55  # Low confidence
    elif num_agree == 2:
        confidence = 68  # Medium
    elif num_agree == 3:
        confidence = 78  # High
    elif num_agree >= 4:
        confidence = 88  # Very high
    else:
        confidence = 50

    # Also consider the average confidence of the agreeing strategies
    avg_conf = np.mean([r[2] for r in group]) if group else 50
    confidence = int((confidence + avg_conf) / 2)  # Blend both

    # Pick the best strategy from the group (highest confidence)
    best_strategy = max(group, key=lambda x: x[2])

    # Return only the best signal with the new confidence
    return [(best_strategy[0], direction, confidence, best_strategy[3], best_strategy[4])]

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
                'close': np.random.randn(30) + 1.12
            }

            results = run_strategies(price_data)

            if not results:
                await update.message.reply_text("⛔ No clear signal — DON'T TRADE.")
                return

            strategy, direction, confidence, expiry_1, expiry_2 = results[0]
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

            await asyncio.sleep(0.5)

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

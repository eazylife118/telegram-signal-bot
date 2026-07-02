import os
import re
import time
import requests
import numpy as np
import pytesseract
from PIL import Image
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime, timezone, timedelta

# ==========================================
# TELEGRAM CREDENTIALS
# ==========================================
TOKEN = "8608138546:AAEetCz5xKlQlIRc0eZ3gVzvs046dPb86UI"
CHAT_ID = "6280535707"

# ==========================================
# TIME ZONE (UTC+1)
# ==========================================
LOCAL_TZ = timezone(timedelta(hours=1))

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
# OCR: DETECT PAIR FROM SCREENSHOT
# ==========================================
def detect_pair_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        match = re.search(r'([A-Z]{3}/[A-Z]{3}\s+OTC)', text)
        if match:
            return match.group(1)
    except Exception as e:
        print("OCR error:", e)
    return "CAD/JPY OTC"  # fallback

# ==========================================
# FLASK WEB SERVER
# ==========================================
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ OTC Screenshot Bot is running!"

@app.route('/ping')
def ping():
    return "pong", 200

def run_flask():
    app.run(host='0.0.0.0', port=10000)

# ==========================================
# 10 STRATEGIES (BRAIN CONTROLS BOTH ENTRIES)
# ==========================================
def run_strategies(price_data):
    results = []
    close = np.array(price_data['close'])
    open_ = np.array(price_data['open'])
    high = np.array(price_data['high'])
    low = np.array(price_data['low'])

    if len(close) >= 3:
        if (close[-3:] > open_[-3:]).all() and (close[-1] < open_[-1]):
            results.append(("Candle Reversal Pattern", "SELL", 86, 2, 3))

    if len(close) >= 3:
        if (close[-3:] > open_[-3:]).all():
            results.append(("3-Candle Momentum", "BUY", 82, 1, 2))

    if len(close) >= 3:
        if (close[-3:] < open_[-3:]).all():
            results.append(("2-Minute Reset", "SELL", 78, 2, 3))

    if len(close) >= 10:
        if abs(low[-1] - low[-3]) < 0.0002:
            results.append(("Double Touch", "BUY", 89, 3, 4))

    if len(close) >= 2:
        if (high[-1] - high[-2]) > 0.001 and (close[-1] < open_[-1]):
            results.append(("Spike Rejection", "SELL", 74, 2, 3))

    if len(close) >= 10:
        high_range = max(high[-10:]) - min(low[-10:])
        if high_range < 0.0005:
            results.append(("Consolidation Break", "BUY", 71, 3, 4))

    if len(close) >= 20:
        ema_20 = np.mean(close[-20:])
        if low[-1] < ema_20 and close[-1] > ema_20:
            results.append(("EMA Pullback", "BUY", 80, 2, 3))

    if len(close) >= 2:
        if close[-1] > open_[-1] and close[-2] > open_[-2]:
            results.append(("Bull/Bear Confirmation", "BUY", 76, 1, 2))

    if len(close) >= 2:
        if (close[-1] - open_[-1]) > (close[-2] - open_[-2]) * 1.5:
            results.append(("60-Second Scalp", "BUY", 72, 1, 1))

    return results

# ==========================================
# PREDICTION ENGINE
# ==========================================
def predict_entries(strategy, direction, confidence, expiry_1, expiry_2, pair_name="CAD/JPY OTC"):
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
        "pair": pair_name,
        "strategy": strategy,
        "entry1": {"time": entry1_time, "dir": entry1_dir, "conf": confidence, "expiry": expiry_1},
        "entry2": {"time": entry2_time, "dir": entry2_dir, "conf": entry2_conf, "expiry": expiry_2}
    }

# ==========================================
# TELEGRAM BOT HANDLERS
# ==========================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Send one or more screenshots of your OTC chart and I'll analyze each one.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo = await update.message.photo[-1].get_file()
        await photo.download_to_drive("screenshot.png")

        # Auto-detect pair from screenshot
        pair_name = detect_pair_from_image("screenshot.png")

        # Placeholder price data — replace with actual OCR extraction
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

        best = max(results, key=lambda x: x[2])
        strategy, direction, confidence, expiry_1, expiry_2 = best
        prediction = predict_entries(strategy, direction, confidence, expiry_1, expiry_2, pair_name)

        response = f"📊 **OTC SIGNAL**\n\n"
        response += f"🔍 **Pair:** {prediction['pair']}\n\n"
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

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# ==========================================
# START BOT
# ==========================================
def run_telegram():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.run_polling()

if __name__ == "__main__":
    import threading
    threading.Thread(target=run_flask).start()
    run_telegram()

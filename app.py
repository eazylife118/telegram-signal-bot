import os
import re
import time
import threading
import requests
import numpy as np
import cv2
import pytesseract

from flask import Flask
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from datetime import datetime, timezone, timedelta


# ============================================================
# TELEGRAM CREDENTIALS
# ============================================================

TOKEN = "8937673241:AAGvyTA-G12xfwMlhif3Nh4_2Ag8OStq3tU"
CHAT_ID = "6280535707"
CHANNEL_ID = "-1004324805205"


# ============================================================
# TIMEZONE
# ============================================================

LOCAL_TZ = timezone(timedelta(hours=1))


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "✅ OTC Reversal Bot is running!"


@app.route("/ping")
def ping():
    return "pong", 200


def run_flask():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=10000, debug=False, threaded=True)


# ============================================================
# REAL CANDLE COUNTER — NO FAKE DATA
# ============================================================

def count_real_candles(image_path):
    """
    Count ONLY real candles from the screenshot.
    No generation. No fake OHLC. Just what is actually visible.
    """

    img = cv2.imread(image_path)
    if img is None:
        return None, "Could not load image"

    height, width = img.shape[:2]

    # Crop to chart area only (remove UI/text)
    top = int(height * 0.15)
    bottom = int(height * 0.83)
    left = int(width * 0.04)
    right = int(width * 0.92)

    chart = img[top:bottom, left:right]
    chart_height, chart_width = chart.shape[:2]

    # HSV for accurate color detection
    hsv = cv2.cvtColor(chart, cv2.COLOR_BGR2HSV)

    # ---- REAL GREEN ----
    green_lower = np.array([30, 35, 35])
    green_upper = np.array([95, 255, 255])
    green = cv2.inRange(hsv, green_lower, green_upper)

    # ---- REAL RED ----
    red1_lower = np.array([0, 35, 35])
    red1_upper = np.array([12, 255, 255])
    red2_lower = np.array([165, 35, 35])
    red2_upper = np.array([180, 255, 255])
    red1 = cv2.inRange(hsv, red1_lower, red1_upper)
    red2 = cv2.inRange(hsv, red2_lower, red2_upper)
    red = cv2.bitwise_or(red1, red2)

    # Combine all candle pixels (real only)
    combined = cv2.bitwise_or(green, red)

    # Find connected components (each candle is a contour)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candles = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Only reject obvious noise
        if h < 4:
            continue
        if w < 1:
            continue
        if w > chart_width * 0.15:
            continue
        if h > chart_height * 0.85:
            continue

        roi_green = green[y:y+h, x:x+w]
        roi_red = red[y:y+h, x:x+w]

        green_pixels = np.sum(roi_green > 0)
        red_pixels = np.sum(roi_red > 0)
        total_pixels = green_pixels + red_pixels

        if total_pixels < 5:
            continue

        color = "GREEN" if green_pixels > red_pixels else "RED"

        candles.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "color": color
        })

    # Merge overlapping/touching candles (avoid double-count)
    candles.sort(key=lambda c: c["x"])

    merged = []
    for candle in candles:
        if not merged:
            merged.append(candle)
            continue

        last = merged[-1]

        if candle["x"] <= last["x"] + last["w"] + 2:
            new_x = min(last["x"], candle["x"])
            new_w = max(last["x"] + last["w"], candle["x"] + candle["w"]) - new_x
            new_y = min(last["y"], candle["y"])
            new_h = max(last["y"] + last["h"], candle["y"] + candle["h"]) - new_y

            last["x"] = new_x
            last["y"] = new_y
            last["w"] = new_w
            last["h"] = new_h
        else:
            merged.append(candle)

    total = len(merged)
    green_count = sum(1 for c in merged if c["color"] == "GREEN")
    red_count = sum(1 for c in merged if c["color"] == "RED")

    return {
        "total": total,
        "green": green_count,
        "red": red_count,
        "chart_width": chart_width,
        "chart_height": chart_height,
    }, None


# ============================================================
# TELEGRAM BOT
# ============================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📊 OTC REAL CANDLE COUNTER\n\n"
        "Send a Pocket Option screenshot.\n\n"
        "I will count ONLY real candles from the chart.\n"
        "No fake data. No generated candles.\n"
        "Just what is actually visible."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start_time = time.time()

    try:
        await update.message.reply_text("📸 Analyzing screenshot...")

        photo = await update.message.photo[-1].get_file()
        await photo.download_to_drive("screenshot.png")

        result, error = count_real_candles("screenshot.png")

        elapsed = time.time() - start_time

        if error:
            await update.message.reply_text(f"❌ Error: {error}")
            return

        if result is None:
            await update.message.reply_text("❌ Could not read the screenshot.")
            return

        total = result["total"]
        green = result["green"]
        red = result["red"]

        message = (
            "📊 **REAL CANDLE COUNT**\n\n"
            f"🕯️ **Total candles:** {total}\n"
            f"🟢 **Green:** {green}\n"
            f"🔴 **Red:** {red}\n\n"
            f"⚡ Time: {elapsed:.2f}s\n\n"
            "✅ No fake data — only what is visible in the screenshot."
        )

        await update.message.reply_text(message)

        print(f"✅ Sent: {total} candles ({green} green, {red} red)")

    except Exception as e:
        print(f"❌ Error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")


# ============================================================
# START TELEGRAM BOT
# ============================================================

def run_telegram():
    application = Application.builder().token(TOKEN).build()
    application.bot.delete_webhook()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("✅ Telegram bot started. Waiting for screenshots...")
    application.run_polling(drop_pending_updates=True)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 40)
    print("📊 OTC REAL CANDLE COUNTER")
    print("=" * 40)

    threading.Thread(target=run_flask, daemon=True).start()
    print("✅ Flask server started on port 10000")

    run_telegram()

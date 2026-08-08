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
    return "✅ OTC Max Candle Counter is running!"


@app.route("/ping")
def ping():
    return "pong", 200


def run_flask():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=10000, debug=False, threaded=True)


# ============================================================
# MAXIMUM CANDLE COUNTER — REAL DATA ONLY
# ============================================================

def count_real_candles(image_path):
    """
    Count EVERY visible candle in the screenshot.
    Maximum detection range — captures full chart.
    No fake data. No generation. Just real pixels.
    """

    img = cv2.imread(image_path)
    if img is None:
        return None, "Could not load image"

    height, width = img.shape[:2]

    # ==========================================================
    # MAXIMUM CHART CROP — Capture the ENTIRE chart area
    # ==========================================================

    # Aggressive crop to capture every possible candle
    top = int(height * 0.05)      # Almost to the top
    bottom = int(height * 0.92)   # Almost to the bottom
    left = int(width * 0.01)      # Almost to the left edge
    right = int(width * 0.98)     # Almost to the right edge

    chart = img[top:bottom, left:right]
    chart_height, chart_width = chart.shape[:2]

    print(f"📏 Chart crop: {chart_width} x {chart_height} pixels")

    # ==========================================================
    # ENHANCE IMAGE FOR BETTER DETECTION
    # ==========================================================

    # Increase contrast
    lab = cv2.cvtColor(chart, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    chart = cv2.merge((l, a, b))
    chart = cv2.cvtColor(chart, cv2.COLOR_LAB2BGR)

    # Convert to HSV for accurate color detection
    hsv = cv2.cvtColor(chart, cv2.COLOR_BGR2HSV)

    # ==========================================================
    # REAL GREEN CANDLES — Wide range
    # ==========================================================

    green_lower = np.array([25, 30, 30])
    green_upper = np.array([100, 255, 255])
    green = cv2.inRange(hsv, green_lower, green_upper)

    # ==========================================================
    # REAL RED CANDLES — Wide range
    # ==========================================================

    red1_lower = np.array([0, 30, 30])
    red1_upper = np.array([15, 255, 255])
    red2_lower = np.array([160, 30, 30])
    red2_upper = np.array([180, 255, 255])
    red1 = cv2.inRange(hsv, red1_lower, red1_upper)
    red2 = cv2.inRange(hsv, red2_lower, red2_upper)
    red = cv2.bitwise_or(red1, red2)

    # ==========================================================
    # DETECT WHITE/BULLISH CANDLES (sometimes Pocket Option uses white)
    # ==========================================================

    white_lower = np.array([0, 0, 180])
    white_upper = np.array([180, 30, 255])
    white = cv2.inRange(hsv, white_lower, white_upper)

    # Combine all candle pixels
    combined = cv2.bitwise_or(green, red)
    combined = cv2.bitwise_or(combined, white)

    # ==========================================================
    # MORPHOLOGICAL CLEANING — Preserve small candles
    # ==========================================================

    kernel = np.ones((1, 1), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # ==========================================================
    # FIND ALL CANDLES — Every contour is a candle
    # ==========================================================

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candles = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # --- MINIMAL FILTERING — Only reject absolute noise ---
        if h < 3:       # Too short to be a candle
            continue
        if w < 1:       # Too narrow
            continue
        if w > chart_width * 0.18:  # Too wide (likely not a candle)
            continue
        if h > chart_height * 0.90:  # Too tall (likely not a candle)
            continue

        # Check actual color in this region
        roi_green = green[y:y+h, x:x+w]
        roi_red = red[y:y+h, x:x+w]
        roi_white = white[y:y+h, x:x+w]

        green_pixels = np.sum(roi_green > 0)
        red_pixels = np.sum(roi_red > 0)
        white_pixels = np.sum(roi_white > 0)
        total_pixels = green_pixels + red_pixels + white_pixels

        if total_pixels < 3:  # Almost no color — noise
            continue

        # Determine color (white counts as bullish/green)
        if green_pixels + white_pixels > red_pixels:
            color = "GREEN"
        else:
            color = "RED"

        candles.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "color": color
        })

    # ==========================================================
    # MERGE OVERLAPPING/TOUCHING CANDLES
    # ==========================================================

    candles.sort(key=lambda c: c["x"])

    merged = []
    for candle in candles:
        if not merged:
            merged.append(candle)
            continue

        last = merged[-1]

        # If overlapping or touching horizontally
        if candle["x"] <= last["x"] + last["w"] + 3:
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

    # ==========================================================
    # FINAL COUNT — REAL DATA ONLY
    # ==========================================================

    total = len(merged)
    green_count = sum(1 for c in merged if c["color"] == "GREEN")
    red_count = sum(1 for c in merged if c["color"] == "RED")

    print(f"📊 Detected: {total} candles ({green_count} green, {red_count} red)")

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
        "📊 **OTC MAX CANDLE COUNTER**\n\n"
        "Send a Pocket Option screenshot.\n\n"
        "I will count **EVERY** visible candle:\n"
        "✅ Green candles\n"
        "✅ Red candles\n"
        "✅ White candles (bullish)\n"
        "✅ Small candles\n"
        "✅ Large candles\n"
        "✅ Doji candles\n\n"
        "🚫 **No fake data**\n"
        "🚫 **No generated candles**\n"
        "✅ **Just what is actually visible**"
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

        # Build confidence message
        if total >= 40:
            status = "✅ **Excellent** — Full chart detected"
        elif total >= 25:
            status = "⚠️ **Good** — Partial chart detected"
        elif total >= 10:
            status = "⚠️ **Low** — Only part of the chart captured"
        else:
            status = "❌ **Very Low** — Chart crop may be too narrow"

        message = (
            "📊 **REAL CANDLE COUNT**\n\n"
            f"🕯️ **Total candles:** {total}\n"
            f"🟢 **Green:** {green}\n"
            f"🔴 **Red:** {red}\n\n"
            f"📏 **Chart size:** {result['chart_width']} x {result['chart_height']} px\n"
            f"⚡ **Time:** {elapsed:.2f}s\n\n"
            f"{status}\n\n"
            "✅ **No fake data** — only what is visible in the screenshot."
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
    print("=" * 50)
    print("📊 OTC MAX CANDLE COUNTER")
    print("=" * 50)
    print("✅ Detects EVERY visible candle")
    print("✅ No fake data")
    print("✅ Maximum chart capture")
    print("=" * 50)

    threading.Thread(target=run_flask, daemon=True).start()
    print("✅ Flask server started on port 10000")

    run_telegram()

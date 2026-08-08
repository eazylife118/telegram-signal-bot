import os
import re
import random
import time
import cv2
import pytesseract
from flask import Flask
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
# ============================================================
# TELEGRAM SETTINGS
# ============================================================
TOKEN = "8937673241:AAGvyTA-G12xfwMlhif3Nh4_2Ag8OStq3tU"
# ============================================================
# FLASK SERVER FOR RENDER
# ============================================================
app = Flask(__name__)
@app.route("/")
def home():
    return "Screenshot Pair Detection Test Bot is running!"
@app.route("/ping")
def ping():
    return "pong", 200
def run_flask():
    app.run(
        host="0.0.0.0",
        port=10000,
        debug=False,
        threaded=True
    )
# ============================================================
# CURRENCY PAIR DETECTION
# ============================================================
CURRENCY_CODES = [
    "USD", "EUR", "GBP", "JPY", "AUD",
    "CAD", "CHF", "NZD", "SGD", "HKD",
    "CNY", "TRY", "ZAR", "MXN", "BRL"
]
def detect_currency_pair(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Could not load screenshot."
    height, width = image.shape[:2]
    if width < 1500:
        scale = 1500 / width
        image = cv2.resize(
            image,
            (1500, int(height * scale)),
            interpolation=cv2.INTER_CUBIC
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normal = gray
    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    images = [normal, otsu, adaptive]
    all_text = []
    for img in images:
        try:
            text = pytesseract.image_to_string(
                img,
                config="--psm 6"
            )
            if text:
                all_text.append(text)
        except Exception as e:
            print("OCR error:", e)
    combined_text = "\n".join(all_text)
    print("\n========== OCR TEXT ==========")
    print(combined_text)
    print("================================\n")
    text = combined_text.upper()
    replacements = {
        "USO": "USD",
        "EURO": "EUR",
        "EUP": "EUR",
        "6BP": "GBP",
        "G8P": "GBP",
        "JPV": "JPY",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    pair_pattern = re.compile(
        r"\b("
        + "|".join(CURRENCY_CODES)
        + r")"
        r"\s*[/\\\-_:]?\s*"
        r"\b("
        + "|".join(CURRENCY_CODES)
        + r")\b"
    )
    matches = pair_pattern.findall(text)
    detected_pairs = []
    for base, quote in matches:
        if base == quote:
            continue
        pair = f"{base}/{quote}"
        if pair not in detected_pairs:
            detected_pairs.append(pair)
    if detected_pairs:
        pair = detected_pairs[0]
        print("✅ CURRENCY PAIR DETECTED:", pair)
        return pair, combined_text
    print("❌ NO CURRENCY PAIR DETECTED")
    return None, combined_text
# ============================================================
# RANDOM TEST RESULT
# ============================================================
def random_test_result(pair):
    return random.choice([
        "🟢 BUY",
        "🔴 SELL"
    ])
# ============================================================
# TELEGRAM /START
# ============================================================
async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "📸 SCREENSHOT PAIR DETECTION TEST\n\n"
        "Send a Pocket Option screenshot.\n\n"
        "The bot will ONLY try to detect the currency pair.\n\n"
        "No strategy.\n"
        "No candle analysis.\n"
        "No prediction.\n"
        "No real trading signal."
    )
# ============================================================
# TELEGRAM PHOTO HANDLER
# ============================================================
async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    start_time = time.time()
    try:
        await update.message.reply_text(
            "📸 Screenshot received.\n"
            "🔎 Reading currency pair..."
        )
        photo = await update.message.photo[-1].get_file()
        file_path = "test_screenshot.png"
        await photo.download_to_drive(file_path)
        print("\n📸 Screenshot downloaded.")
        print("🔎 Starting OCR pair detection...\n")
        pair, ocr_text = detect_currency_pair(file_path)
        if pair:
            random_result = random_test_result(pair)
            elapsed = time.time() - start_time
            response = (
                "✅ SCREENSHOT TEST RESULT\n\n"
                f"💱 Currency Pair: {pair}\n"
                f"🧪 Random Test: {random_result}\n\n"
                "📸 Screenshot was successfully received.\n"
                "🔎 Currency pair was detected by Python OCR.\n\n"
                f"⚡ Processing time: {elapsed:.2f}s\n\n"
                "⚠️ RANDOM TEST ONLY — NOT A TRADING SIGNAL."
            )
            await update.message.reply_text(response)
            print("================================")
            print("✅ TEST PASSED")
            print("PAIR:", pair)
            print("RANDOM RESULT:", random_result)
            print(f"TIME: {elapsed:.2f}s")
            print("================================")
        else:
            await update.message.reply_text(
                "❌ CURRENCY PAIR NOT DETECTED\n\n"
                "Python received the screenshot, but "
                "the OCR could not identify a supported "
                "currency pair.\n\n"
                "Try sending a clearer screenshot with "
                "the pair name visible."
            )
            print("❌ TEST FAILED: PAIR NOT DETECTED")
    except Exception as e:
        print("ERROR:", str(e))
        await update.message.reply_text(
            f"❌ TEST ERROR\n\n{str(e)}"
        )
# ============================================================
# START TELEGRAM BOT
# ============================================================
def run_bot():
    application = (
        Application.builder()
        .token(TOKEN)
        .build()
    )
    application.add_handler(
        CommandHandler("start", start)
    )
    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_photo
        )
    )
    print("✅ Telegram bot started.")
    print("📸 Waiting for screenshot...")
    application.run_polling(
        drop_pending_updates=True
    )
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import threading
    print("========================================")
    print("📸 SCREENSHOT PAIR DETECTION TEST")
    print("========================================")
    threading.Thread(
        target=run_flask,
        daemon=True
    ).start()
    print("✅ Flask server started.")
    print("✅ Starting Telegram bot...")
    run_bot()

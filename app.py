import os
import re
import time
import threading
import cv2
import easyocr
from flask import Flask
from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    ContextTypes,
    filters,
)
# ==========================================
# TELEGRAM
# ==========================================
TOKEN = os.getenv("8937673241:AAHIb8yOEPj38vCPsS2Nir9b0CtCtsEfsaM")
# ==========================================
# FLASK
# ==========================================
app = Flask(__name__)
@app.route("/")
def home():
    return "✅ Python 3 + EasyOCR Pair Detector is running!"
@app.route("/ping")
def ping():
    return "pong", 200
def run_flask():
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
        debug=False
    )
# ==========================================
# OCR
# ==========================================
print("🔎 Loading EasyOCR...")
reader = easyocr.Reader(
    ["en"],
    gpu=False
)
print("✅ EasyOCR loaded.")
# ==========================================
# CURRENCY PAIR DETECTION
# ==========================================
CURRENCIES = [
    "EUR",
    "USD",
    "GBP",
    "JPY",
    "AUD",
    "CAD",
    "CHF",
    "NZD",
    "SGD",
    "HKD",
]
def detect_currency_pair(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Could not load screenshot."
    height, width = image.shape[:2]
    # Enlarge small screenshots
    if width < 1500:
        scale = 1500 / width
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )
    print("📸 Running OCR...")
    try:
        results = reader.readtext(image)
    except Exception as e:
        return None, f"OCR error: {e}"
    detected_text = []
    for result in results:
        if len(result) >= 2:
            text = result[1]
            if text:
                detected_text.append(text)
    print("========== OCR TEXT ==========")
    for text in detected_text:
        print(text)
    print("================================")
    # Combine OCR text
    combined = " ".join(detected_text).upper()
    # Normalize common OCR problems
    normalized = combined
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("|", "/")
    normalized = normalized.replace("\\", "/")
    # Search for EUR/USD etc.
    currency_pattern = (
        r"(EUR|USD|GBP|JPY|AUD|CAD|CHF|NZD|SGD|HKD)"
        r"[/\-]"
        r"(EUR|USD|GBP|JPY|AUD|CAD|CHF|NZD|SGD|HKD)"
    )
    match = re.search(currency_pattern, normalized)
    if match:
        base = match.group(1)
        quote = match.group(2)
        if base != quote:
            pair = f"{base}/{quote}"
            if "OTC" in normalized:
                pair += " OTC"
            return pair, combined
    # Try pair without slash
    for base in CURRENCIES:
        for quote in CURRENCIES:
            if base == quote:
                continue
            pair_without_slash = base + quote
            if pair_without_slash in normalized:
                pair = f"{base}/{quote}"
                if "OTC" in normalized:
                    pair += " OTC"
                return pair, combined
    return None, combined
# ==========================================
# TELEGRAM PHOTO HANDLER
# ==========================================
async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    filename = "test_screenshot.png"
    try:
        await update.message.reply_text(
            "📸 Screenshot received.\n"
            "🔎 Detecting currency pair..."
        )
        photo = await update.message.photo[-1].get_file()
        await photo.download_to_drive(filename)
        start_time = time.time()
        pair, ocr_text = detect_currency_pair(filename)
        elapsed = time.time() - start_time
        if pair:
            await update.message.reply_text(
                f"✅ CURRENCY PAIR DETECTED\n\n"
                f"💱 {pair}\n\n"
                f"⏱ OCR time: {elapsed:.2f} seconds\n\n"
                f"✅ Python 3 screenshot test successful."
            )
        else:
            preview = ocr_text.strip()
            if len(preview) > 1500:
                preview = preview[:1500]
            if not preview:
                preview = "[No readable text detected]"
            await update.message.reply_text(
                "❌ CURRENCY PAIR NOT DETECTED\n\n"
                "Python received the screenshot, "
                "but the pair was not recognized.\n\n"
                "OCR READ:\n"
                f"{preview}"
            )
    except Exception as e:
        print("❌ ERROR:", e)
        await update.message.reply_text(
            f"❌ Test error:\n{str(e)}"
        )
    finally:
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except:
            pass
# ==========================================
# TELEGRAM BOT
# ==========================================
def run_telegram():
    if not TOKEN:
        print(
            "❌ BOT_TOKEN environment variable "
            "is missing."
        )
        return
    application = (
        Application
        .builder()
        .token(TOKEN)
        .build()
    )
    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_photo
        )
    )
    print(
        "✅ Telegram screenshot "
        "pair detector started."
    )
    application.run_polling()
# ==========================================
# START
# ==========================================
if __name__ == "__main__":
    threading.Thread(
        target=run_flask,
        daemon=True
    ).start()
    print("✅ Flask server started.")
    run_telegram()

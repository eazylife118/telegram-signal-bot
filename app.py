import os
import re
import time
import threading
import cv2
import pytesseract
import requests
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
# FLASK SERVER
# ==========================================
app = Flask(__name__)
@app.route("/")
def home():
    return "✅ Python 3 Screenshot Pair Detector is running!"
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
# CURRENCY PAIR DETECTION
# ==========================================
KNOWN_CURRENCIES = [
    "EUR", "USD", "GBP", "JPY", "AUD",
    "CAD", "CHF", "NZD", "SGD", "HKD"
]
def detect_currency_pair(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Could not load screenshot."
    # Make the image larger for OCR
    height, width = image.shape[:2]
    if width < 1500:
        scale = 1500 / width
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )
    # Several OCR attempts
    images_to_read = []
    # Original enlarged image
    images_to_read.append(image)
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images_to_read.append(gray)
    # Threshold
    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    images_to_read.append(threshold)
    # Inverted threshold
    inverted = cv2.bitwise_not(threshold)
    images_to_read.append(inverted)
    all_text = []
    for img in images_to_read:
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
    print("========== OCR TEXT ==========")
    print(combined_text)
    print("================================")
    # Normalize OCR text
    text = combined_text.upper()
    # Remove common OCR spacing problems
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.replace("|", "/")
    # Search for standard currency pairs
    currencies = "|".join(KNOWN_CURRENCIES)
    patterns = [
        rf"\b({currencies})[/\-]({currencies})\b",
        rf"\b({currencies})({currencies})\b"
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            base = match[0]
            quote = match[1]
            if base != quote:
                pair = f"{base}/{quote}"
                # Check nearby original text for OTC
                if "OTC" in text:
                    pair += " OTC"
                return pair, combined_text
    # Additional OCR cleanup for common mistakes
    corrections = {
        "EURSUD": "EURUSD",
        "EURIUSD": "EURUSD",
        "EUR/USO": "EUR/USD",
        "GBP/USO": "GBP/USD",
        "USO/JPY": "USD/JPY",
        "AUD/USO": "AUD/USD",
        "USD/JFV": "USD/JPY",
    }
    cleaned = text
    for wrong, correct in corrections.items():
        cleaned = cleaned.replace(wrong, correct)
    for pattern in patterns:
        matches = re.findall(pattern, cleaned)
        for match in matches:
            base = match[0]
            quote = match[1]
            if base != quote:
                pair = f"{base}/{quote}"
                if "OTC" in cleaned:
                    pair += " OTC"
                return pair, combined_text
    return None, combined_text
# ==========================================
# TELEGRAM PHOTO HANDLER
# ==========================================
async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    try:
        await update.message.reply_text(
            "📸 Screenshot received.\n"
            "🔎 Reading currency pair..."
        )
        photo = await update.message.photo[-1].get_file()
        filename = "test_screenshot.png"
        await photo.download_to_drive(filename)
        start_time = time.time()
        pair, raw_text = detect_currency_pair(filename)
        elapsed = time.time() - start_time
        if pair:
            await update.message.reply_text(
                f"✅ PAIR DETECTED\n\n"
                f"💱 {pair}\n\n"
                f"⏱ OCR time: {elapsed:.2f}s\n\n"
                f"Python 3 screenshot test successful."
            )
        else:
            # Send the OCR text so we can see what Python actually read
            preview = raw_text.strip()
            if len(preview) > 1000:
                preview = preview[:1000]
            await update.message.reply_text(
                "❌ Currency pair was not detected.\n\n"
                "Python DID receive and read the screenshot, "
                "but the pair was not recognized.\n\n"
                "OCR text detected:\n"
                f"{preview if preview else '[No text detected]'}"
            )
        # Remove test screenshot
        try:
            os.remove(filename)
        except:
            pass
    except Exception as e:
        print("ERROR:", e)
        await update.message.reply_text(
            f"❌ Test error:\n{str(e)}"
        )
# ==========================================
# TELEGRAM BOT
# ==========================================
def run_telegram():
    if not TOKEN:
        print("❌ BOT_TOKEN environment variable is missing.")
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
    print("✅ Telegram screenshot detector started.")
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

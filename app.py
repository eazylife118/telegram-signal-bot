import os
import re
import threading
import cv2
import pytesseract
from flask import Flask
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters
TOKEN = os.getenv("8937673241:AAHIb8yOEPj38vCPsS2Nir9b0CtCtsEfsaM")
app = Flask(__name__)
@app.route("/")
def home():
    return "Pair detector running"
@app.route("/ping")
def ping():
    return "pong"
def run_flask():
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000"))
    )
def detect_pair(filename):
    image = cv2.imread(filename)
    if image is None:
        return None, "Image could not be loaded."
    # Enlarge screenshot
    height, width = image.shape[:2]
    if width < 1200:
        scale = 1200 / width
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Simple OCR
    text = pytesseract.image_to_string(
        gray,
        config="--psm 6"
    )
    print("========== OCR ==========")
    print(text)
    print("=========================")
    text = text.upper()
    # Normalize common OCR mistakes
    text = text.replace(" ", "")
    text = text.replace("|", "/")
    text = text.replace("\\", "/")
    currencies = (
        "EUR|USD|GBP|JPY|AUD|CAD|CHF|NZD|"
        "SGD|HKD"
    )
    # EUR/USD
    pattern = (
        rf"({currencies})[/\-]"
        rf"({currencies})"
    )
    match = re.search(pattern, text)
    if match:
        base = match.group(1)
        quote = match.group(2)
        if base != quote:
            pair = f"{base}/{quote}"
            if "OTC" in text:
                pair += " OTC"
            return pair, text
    # EURUSD without slash
    pattern2 = (
        rf"({currencies})"
        rf"({currencies})"
    )
    match = re.search(pattern2, text)
    if match:
        base = match.group(1)
        quote = match.group(2)
        if base != quote:
            pair = f"{base}/{quote}"
            if "OTC" in text:
                pair += " OTC"
            return pair, text
    return None, text
async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    filename = "screenshot.png"
    try:
        await update.message.reply_text(
            "📸 Screenshot received.\n"
            "🔎 Reading currency pair..."
        )
        photo = await update.message.photo[-1].get_file()
        await photo.download_to_drive(filename)
        pair, text = detect_pair(filename)
        if pair:
            await update.message.reply_text(
                "✅ PAIR DETECTED\n\n"
                f"💱 {pair}\n\n"
                "Python 3 OCR test successful."
            )
        else:
            preview = text.strip()
            if not preview:
                preview = "[No readable text detected]"
            if len(preview) > 1200:
                preview = preview[:1200]
            await update.message.reply_text(
                "❌ PAIR NOT DETECTED\n\n"
                "OCR read:\n"
                f"{preview}"
            )
    except Exception as e:
        await update.message.reply_text(
            f"❌ Error:\n{e}"
        )
    finally:
        if os.path.exists(filename):
            os.remove(filename)
def run_bot():
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
    print("✅ Bot started")
    application.run_polling()
if __name__ == "__main__":
    threading.Thread(
        target=run_flask,
        daemon=True
    ).start()
    run_bot()

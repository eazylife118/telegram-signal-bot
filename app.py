import os
import re
import time
import threading
from flask import Flask
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

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
    return "Screenshot OCR Test Bot is running!"

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
# FAST SCREENSHOT OCR
# ============================================================

def read_screenshot(image_path):

    try:
        image = Image.open(image_path)

        # ----------------------------------------------------
        # Resize only when necessary
        # ----------------------------------------------------

        width, height = image.size

        if width < 1200:
            scale = 1200 / width
            image = image.resize(
                (
                    int(width * scale),
                    int(height * scale)
                )
            )

        # ----------------------------------------------------
        # Improve contrast
        # ----------------------------------------------------

        image = ImageEnhance.Contrast(image).enhance(1.5)
        image = ImageEnhance.Sharpness(image).enhance(1.5)

        # ----------------------------------------------------
        # OCR the COMPLETE screenshot
        # ----------------------------------------------------

        text = pytesseract.image_to_string(
            image,
            config="--psm 11"
        )

        # ----------------------------------------------------
        # Clean OCR output
        # ----------------------------------------------------

        lines = []

        for line in text.splitlines():

            line = line.strip()

            if line:
                lines.append(line)

        cleaned_text = "\n".join(lines)

        return cleaned_text

    except Exception as e:

        print("OCR ERROR:", e)

        return None

# ============================================================
# TELEGRAM START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📸 SCREENSHOT OCR TEST\n\n"
        "Send ANY screenshot.\n\n"
        "The bot will try to read ANY visible text.\n\n"
        "It does NOT require USD/CHF.\n"
        "It does NOT require EUR/USD.\n"
        "It does NOT require a currency pair.\n\n"
        "It can attempt to read:\n"
        "• OTC names\n"
        "• Currency pairs\n"
        "• Asset names\n"
        "• Prices\n"
        "• Buttons\n"
        "• Chart text\n"
        "• Other visible text"
    )

# ============================================================
# PHOTO HANDLER
# ============================================================

async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    start_time = time.time()

    try:

        await update.message.reply_text(
            "📸 Screenshot received.\n"
            "🔎 Reading screenshot..."
        )

        # ----------------------------------------------------
        # Download screenshot
        # ----------------------------------------------------

        photo = await update.message.photo[-1].get_file()

        file_path = "screenshot_test.png"

        await photo.download_to_drive(file_path)

        print("📸 Screenshot downloaded.")

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        detected_text = read_screenshot(file_path)

        elapsed = time.time() - start_time

        # ----------------------------------------------------
        # Nothing detected
        # ----------------------------------------------------

        if not detected_text:

            await update.message.reply_text(
                "❌ NO TEXT DETECTED\n\n"
                "The screenshot was received, but "
                "OCR could not find readable text.\n\n"
                f"⚡ Processing time: {elapsed:.2f}s"
            )

            return

        # ----------------------------------------------------
        # Limit Telegram message size
        # ----------------------------------------------------

        if len(detected_text) > 3500:
            detected_text = detected_text[:3500] + "\n..."

        # ----------------------------------------------------
        # Send result
        # ----------------------------------------------------

        response = (
            "✅ SCREENSHOT OCR RESULT\n\n"
            "🔎 TEXT DETECTED:\n"
            "────────────────────\n"
            f"{detected_text}\n"
            "────────────────────\n\n"
            f"⚡ Processing time: {elapsed:.2f}s\n\n"
            "🧪 TEST ONLY — NO TRADING SIGNAL."
        )

        await update.message.reply_text(response)

        print("\n========================================")
        print("✅ OCR TEST PASSED")
        print("========================================")
        print(detected_text)
        print("----------------------------------------")
        print(f"⚡ Time: {elapsed:.2f}s")
        print("========================================\n")

    except Exception as e:

        print("ERROR:", e)

        await update.message.reply_text(
            f"❌ TEST ERROR\n\n{str(e)}"
        )

# ============================================================
# TELEGRAM BOT
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

    print("========================================")
    print("📸 FAST SCREENSHOT OCR TEST BOT")
    print("========================================")
    print("✅ Telegram bot started.")
    print("📸 Waiting for screenshots...")

    application.run_polling(
        drop_pending_updates=True
    )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("========================================")
    print("📸 SCREENSHOT OCR TEST")
    print("========================================")

    threading.Thread(
        target=run_flask,
        daemon=True
    ).start()

    print("✅ Flask server started.")
    print("✅ Starting Telegram bot...")

    run_bot()

import os
import re
import time
import cv2
import pytesseract
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

# ============================================================
# TELEGRAM TOKEN
# ============================================================

TOKEN = "8937673241:AAGvyTA-G12xfwMlhif3Nh4_2Ag8OStq3tU"

# ============================================================
# FLASK SERVER FOR RENDER
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "Fast Screenshot OCR Test Bot is running!"


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

def analyze_screenshot(image_path):

    start = time.time()

    image = cv2.imread(image_path)

    if image is None:
        return None, "Could not load screenshot."

    original_height, original_width = image.shape[:2]

    # --------------------------------------------------------
    # Resize only when necessary
    # --------------------------------------------------------

    max_width = 1400

    if original_width > max_width:

        scale = max_width / original_width

        image = cv2.resize(
            image,
            (
                max_width,
                int(original_height * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Light sharpening
    # --------------------------------------------------------

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # --------------------------------------------------------
    # ONE FAST OCR PASS
    #
    # psm 11 works well for screenshots containing
    # scattered text such as Pocket Option.
    # --------------------------------------------------------

    text = pytesseract.image_to_string(
        gray,
        config="--psm 11"
    )

    elapsed = time.time() - start

    if not text.strip():
        return None, f"OCR found no readable text. Time: {elapsed:.2f}s"

    # --------------------------------------------------------
    # Clean OCR text
    # --------------------------------------------------------

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:
            lines.append(line)

    cleaned_text = "\n".join(lines)

    # --------------------------------------------------------
    # Detect asset / pair
    #
    # This DOES NOT require USDCHF.
    # It searches for common Pocket Option style names.
    # --------------------------------------------------------

    asset = detect_asset(cleaned_text)

    return {
        "text": cleaned_text,
        "asset": asset,
        "time": elapsed,
        "width": image.shape[1],
        "height": image.shape[0]
    }, None


# ============================================================
# ASSET DETECTION
# ============================================================

def detect_asset(text):

    upper = text.upper()

    # --------------------------------------------------------
    # Common OTC / Pocket Option assets
    # --------------------------------------------------------

    known_assets = [

        # Major currencies
        "EUR/USD",
        "GBP/USD",
        "USD/JPY",
        "USD/CHF",
        "AUD/USD",
        "USD/CAD",
        "NZD/USD",
        "EUR/GBP",
        "EUR/JPY",
        "GBP/JPY",

        # Metals
        "GOLD",
        "SILVER",

        # Crypto
        "BTC/USD",
        "ETH/USD",

        # Stocks / assets
        "AMERICAN EXPRESS",
        "APPLE",
        "AMAZON",
        "TESLA",
        "MICROSOFT",
        "GOOGLE",
        "META",
        "NVIDIA",
        "MCDONALD",
        "COCA-COLA",
        "NIKE",

        # Common Pocket Option wording
        "OTC"
    ]

    # --------------------------------------------------------
    # Direct name detection
    # --------------------------------------------------------

    for asset in known_assets:

        if asset in upper:

            return asset

    # --------------------------------------------------------
    # Try to detect currency pairs without slash
    #
    # USDCHF
    # EURUSD
    # GBPJPY
    # --------------------------------------------------------

    pair_pattern = re.compile(
        r"\b("
        r"USD|EUR|GBP|JPY|AUD|CAD|CHF|NZD"
        r")"
        r"\s*"
        r"[/\\\-_:]?"
        r"\s*"
        r"(USD|EUR|GBP|JPY|AUD|CAD|CHF|NZD)"
        r"\b"
    )

    match = pair_pattern.search(upper)

    if match:

        base = match.group(1)
        quote = match.group(2)

        if base != quote:

            return f"{base}/{quote}"

    # --------------------------------------------------------
    # If no known asset is found, look for an OTC line.
    #
    # Example:
    # American Express OTC
    # --------------------------------------------------------

    for line in upper.splitlines():

        if "OTC" in line:

            line = line.strip()

            if len(line) > 3:

                return line

    # --------------------------------------------------------
    # Last fallback:
    # search lines that look like an asset name.
    # --------------------------------------------------------

    for line in upper.splitlines():

        line = line.strip()

        if (
            3 <= len(line) <= 40
            and any(c.isalpha() for c in line)
            and not line.isdigit()
        ):

            # Ignore common interface words
            ignored = [
                "TRADES",
                "SIGNALS",
                "SOCIAL TRADING",
                "MORE",
                "AMOUNT",
                "TIME",
                "EXPIRATION TIME",
                "PROFIT",
                "PAYOUT",
                "USD",
                "DEMO"
            ]

            if line not in ignored:

                return line

    return None


# ============================================================
# TELEGRAM /START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📸 FAST SCREENSHOT OCR TEST\n\n"
        "Send ANY Pocket Option screenshot.\n\n"
        "The bot will:\n"
        "🔎 Read visible text\n"
        "💱 Try to identify the asset/pair\n"
        "📊 Detect OTC names such as American Express OTC\n"
        "⚡ Return the result as quickly as possible\n\n"
        "⚠️ TEST ONLY — NO TRADING SIGNAL."
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
            "⚡ Fast OCR analysis started..."
        )

        # ----------------------------------------------------
        # Download image
        # ----------------------------------------------------

        photo = await update.message.photo[-1].get_file()

        file_path = "screenshot.png"

        await photo.download_to_drive(file_path)

        download_time = time.time() - start_time

        print(
            f"📸 Screenshot downloaded "
            f"({download_time:.2f}s)"
        )

        # ----------------------------------------------------
        # Analyze
        # ----------------------------------------------------

        result, error = analyze_screenshot(
            file_path
        )

        if result is None:

            await update.message.reply_text(
                "❌ OCR FAILED\n\n"
                f"{error}\n\n"
                "Try another clear screenshot."
            )

            return

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        ocr_text = result["text"]
        asset = result["asset"]
        ocr_time = result["time"]

        total_time = time.time() - start_time

        if asset:

            asset_text = (
                f"💱 DETECTED ASSET:\n"
                f"**{asset}**"
            )

        else:

            asset_text = (
                "💱 DETECTED ASSET:\n"
                "❌ Could not confidently identify the asset"
            )

        # ----------------------------------------------------
        # Limit displayed OCR text
        # ----------------------------------------------------

        display_text = ocr_text

        if len(display_text) > 5000:

            display_text = display_text[:5000]

            display_text += "\n...[TEXT TRIMMED]"

        response = (
            "✅ SCREENSHOT OCR RESULT\n\n"

            f"{asset_text}\n\n"

            "🔎 TEXT DETECTED:\n"
            "────────────────────\n"
            f"{display_text}\n"
            "────────────────────\n\n"

            f"⚡ OCR processing: {ocr_time:.2f}s\n"
            f"⚡ Total processing: {total_time:.2f}s\n\n"

            "🧪 TEST ONLY — NO TRADING SIGNAL."
        )

        await update.message.reply_text(
            response,
            parse_mode="Markdown"
        )

        print("\n========================================")
        print("✅ SCREENSHOT TEST COMPLETE")
        print("========================================")
        print("ASSET:", asset)
        print(f"OCR TIME: {ocr_time:.2f}s")
        print(f"TOTAL TIME: {total_time:.2f}s")
        print("========================================\n")

    except Exception as e:

        print(
            "❌ ERROR:",
            str(e)
        )

        await update.message.reply_text(
            "❌ TEST ERROR\n\n"
            f"{str(e)}"
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
        CommandHandler(
            "start",
            start
        )
    )

    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_photo
        )
    )

    print("========================================")
    print("⚡ FAST SCREENSHOT OCR BOT")
    print("========================================")
    print("✅ Telegram bot started")
    print("📸 Waiting for screenshots...")
    print("========================================")

    application.run_polling(
        drop_pending_updates=True
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("========================================")
    print("📸 FAST SCREENSHOT OCR TEST")
    print("========================================")

    threading.Thread(
        target=run_flask,
        daemon=True
    ).start()

    print("✅ Flask server started.")
    print("✅ Starting Telegram bot...")

    run_bot()

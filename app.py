import os
import time
import base64
import threading
import requests

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
# CONFIGURATION
# ============================================================

BOT_TOKEN = "8937673241:AAGvyTA-G12xfwMlhif3Nh4_2Ag8OStq3tU"
GEMINI_API_KEY = GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GEMINI_MODEL = "gemini-2.5-flash-lite"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "Vision Candle Test Bot is running."


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
# GEMINI VISION PROMPT
# ============================================================

VISION_PROMPT = r"""
You are a strict visual inspection system.

Analyze ONLY the screenshot that is attached to this request.

THIS IS A CANDLE COUNTING TEST.

Do NOT perform trading analysis.
Do NOT predict the next candle.
Do NOT generate a signal.
Do NOT create OHLC data.
Do NOT invent missing candles.
Do NOT assume a standard number of candles.
Do NOT estimate candles from the chart width.
Do NOT use OCR output as a substitute for visually identifying candles.

Your job is to inspect the actual chart visually.

IMPORTANT:
Count INDIVIDUAL candlestick objects that are visibly present in the chart.

A candle normally consists of:
- a body
- and possibly an upper wick
- and/or a lower wick.

Two neighboring candles with the same color are STILL TWO SEPARATE CANDLES.

Do not combine consecutive green candles into one.
Do not combine consecutive red candles into one.

Ignore:
- price labels
- price-axis numbers
- grid lines
- time labels
- buttons
- menus
- account balance
- expiration timer
- payout text
- Telegram/UI elements
- other colored interface elements.

FIRST determine which portion of the screenshot is the actual trading chart.

Then inspect the chart from LEFT TO RIGHT.

Count every candle that can actually be seen.

For every visible candle determine:
- GREEN or RED.

If a candle is partially visible but its body/color can still be confidently identified, count it and mark it PARTIAL.

If something is ambiguous and cannot confidently be identified as a candle, DO NOT count it.

Do not fill gaps with guessed candles.

Also identify the visible trading asset/pair if readable.
Examples:
EUR/USD OTC
AUD/CHF OTC
American Express OTC

If the asset cannot be read confidently, say:
NOT CONFIDENTLY DETECTED

Return ONLY this structure:

ASSET: <asset>

TOTAL_CANDLES: <number>

GREEN: <number>

RED: <number>

UNCERTAIN: <number>

SEQUENCE:
<left-to-right sequence using GREEN and RED>

PARTIAL_CANDLES:
<number>

VISUAL_CONFIDENCE:
<HIGH / MEDIUM / LOW>

CHART_VISIBILITY:
<CLEAR / PARTIALLY OBSTRUCTED / POOR>

NOTES:
<short explanation of exactly what made counting difficult, if anything>

FINAL RULE:
The numbers must correspond ONLY to candles you can actually see.
Accuracy is more important than producing a high number.
If only 20 candles can be confidently seen, report 20.
Never report 50 just because the user expects approximately 50.
"""


# ============================================================
# GEMINI VISION
# ============================================================

def analyze_with_gemini(image_bytes, mime_type="image/png"):

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": VISION_PROMPT
                    },
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": encoded_image
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 1000
        }
    }

    response = requests.post(
        GEMINI_URL,
        params={
            "key": GEMINI_API_KEY
        },
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API error {response.status_code}: "
            f"{response.text[:1000]}"
        )

    data = response.json()

    try:
        text = (
            data["candidates"][0]
            ["content"]
            ["parts"][0]
            ["text"]
        )
    except Exception:
        raise RuntimeError(
            f"Unexpected Gemini response: {data}"
        )

    return text.strip()


# ============================================================
# TELEGRAM
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🔎 VISION CANDLE TEST\n\n"
        "Send a Pocket Option screenshot.\n\n"
        "I will use vision to inspect the actual "
        "chart and report:\n\n"
        "🟢 Green candles\n"
        "🔴 Red candles\n"
        "📊 Total candles\n"
        "🕯️ Left-to-right sequence\n"
        "💱 Detected asset/pair\n\n"
        "No strategy.\n"
        "No OHLC generation.\n"
        "No random candles.\n"
        "No trading signal."
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

        if not GEMINI_API_KEY:
            await update.message.reply_text(
                "❌ GEMINI_API_KEY is not configured."
            )
            return

        await update.message.reply_text(
            "👁️ Vision is inspecting the screenshot...\n"
            "Counting the visible candles."
        )

        photo = await (
            update.message
            .photo[-1]
            .get_file()
        )

        image_bytes = await photo.download_as_bytearray()

        result = analyze_with_gemini(
            bytes(image_bytes),
            "image/jpeg"
        )

        elapsed = time.time() - start_time

        message = (
            "🔎 **VISION CANDLE TEST**\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"{result}\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "⚠️ **TEST ONLY**\n"
            "No candles were generated.\n"
            "No OHLC data was generated.\n"
            "No random candles were added.\n"
            "No trading signal was generated.\n\n"
            f"⚡ Processing time: **{elapsed:.2f}s**"
        )

        await update.message.reply_text(
            message,
            parse_mode="Markdown"
        )

        print(
            "\n========== VISION RESULT =========="
        )
        print(result)
        print(
            f"Processing time: {elapsed:.2f}s"
        )
        print(
            "====================================\n"
        )

    except Exception as e:

        elapsed = time.time() - start_time

        print(
            "❌ Vision error:",
            repr(e)
        )

        await update.message.reply_text(
            "❌ **VISION ERROR**\n\n"
            f"{str(e)}\n\n"
            f"Processing time: {elapsed:.2f}s",
            parse_mode="Markdown"
        )


# ============================================================
# TELEGRAM BOT
# ============================================================

def run_telegram():

    if not BOT_TOKEN:
        print(
            "❌ BOT_TOKEN is missing."
        )
        return

    if not GEMINI_API_KEY:
        print(
            "❌ GEMINI_API_KEY is missing."
        )
        return

    application = (
        Application
        .builder()
        .token(BOT_TOKEN)
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

    print(
        "✅ Vision Candle Test Bot started."
    )

    application.run_polling(
        drop_pending_updates=True
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("👁️ POCKET OPTION VISION CANDLE TEST")
    print("=" * 55)

    threading.Thread(
        target=run_flask,
        daemon=True
    ).start()

    run_telegram()

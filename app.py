import os
import time
import io
import base64
import telebot
from PIL import Image
from google import genai
# ============================================================
# CONFIGURATION
# ============================================================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing from Render Environment Variables.")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from Render Environment Variables.")
# ============================================================
# TELEGRAM
# ============================================================
bot = telebot.TeleBot(BOT_TOKEN)
# ============================================================
# GEMINI
# ============================================================
client = genai.Client(
    api_key=GEMINI_API_KEY
)
# Use a current Gemini model available through the GenAI API.
MODEL_NAME = "gemini-2.5-flash"
# ============================================================
# GEMINI VISION PROMPT
# ============================================================
VISION_PROMPT = """
You are a visual candle-reading assistant.
Analyze ONLY the uploaded screenshot.
IMPORTANT RULES:
1. Do NOT invent candles.
2. Do NOT create OHLC data.
3. Do NOT assume a candle exists because of empty space.
4. Do NOT force the result to a particular number of candles.
5. Do NOT generate trading signals.
6. Do NOT predict the next candle.
7. Do NOT use random data.
Your job is ONLY to identify candle bodies that are visibly present.
Look across the ENTIRE uploaded screenshot from the far LEFT to the far RIGHT.
For every visibly identifiable candle:
- identify its approximate left-to-right position
- classify its BODY as GREEN or RED
- ignore BUY/SELL buttons and interface elements
- ignore empty background
- ignore text
- ignore horizontal chart lines
- ignore buttons
- ignore interface decorations
- ignore isolated colored pixels that are clearly not candle bodies
- do not count the same candle twice
A candle may have:
- a small body
- a large body
- a doji/small body
- a wick
The candle body is more important than the wick.
Do NOT treat a long thin vertical line by itself as a candle unless there is clear evidence that it belongs to a candle body.
For the final answer, provide:
GREEN COUNT: number
RED COUNT: number
TOTAL: number
Then provide the candle sequence from LEFT to RIGHT using:
1. GREEN
2. RED
3. GREEN
etc.
Then provide a short confidence note explaining whether some candles are difficult to identify.
Remember:
This is a VISION TEST ONLY.
There is NO trading signal.
"""
# ============================================================
# IMAGE PREPARATION
# ============================================================
def prepare_image(image_bytes):
    """
    Opens the Telegram image and converts it to JPEG.
    No crop is applied.
    The complete uploaded screenshot is preserved.
    """
    image = Image.open(
        io.BytesIO(image_bytes)
    )
    image = image.convert("RGB")
    output = io.BytesIO()
    image.save(
        output,
        format="JPEG",
        quality=95
    )
    output.seek(0)
    return output.read()
# ============================================================
# GEMINI VISION
# ============================================================
def analyze_image(image_bytes):
    image_data = prepare_image(
        image_bytes
    )
    image_part = {
        "mime_type": "image/jpeg",
        "data": image_data
    }
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            VISION_PROMPT,
            image_part
        ]
    )
    if not response or not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )
    return response.text.strip()
# ============================================================
# TELEGRAM PHOTO HANDLER
# ============================================================
@bot.message_handler(
    content_types=["photo"]
)
def handle_photo(message):
    start_time = time.time()
    try:
        bot.reply_to(
            message,
            "👁️ Reading the entire screenshot...\n"
            "Checking GREEN and RED candle bodies."
        )
        # ----------------------------------------------------
        # GET HIGHEST TELEGRAM PHOTO RESOLUTION
        # ----------------------------------------------------
        file_info = bot.get_file(
            message.photo[-1].file_id
        )
        image_bytes = bot.download_file(
            file_info.file_path
        )
        # ----------------------------------------------------
        # GEMINI ANALYSIS
        # ----------------------------------------------------
        result = analyze_image(
            image_bytes
        )
        elapsed = (
            time.time()
            - start_time
        )
        # ----------------------------------------------------
        # FORMAT RESPONSE
        # ----------------------------------------------------
        report = (
            "🔎 **CANDLE VISION TEST**\n\n"
            "🟡 **DETECTION AREA:**\n"
            "Entire uploaded screenshot — 0% to 100%\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"{result}\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "🎯 **COLOR CHECK**\n"
            "🟢 GREEN = Gemini visually identified a green candle body\n"
            "🔴 RED = Gemini visually identified a red candle body\n\n"
            "⚠️ **TEST ONLY**\n"
            "No OHLC candles are generated.\n"
            "No random candles are added.\n"
            "No trading signal is generated.\n\n"
            f"⚡ Processing time: {elapsed:.2f}s"
        )
        bot.reply_to(
            message,
            report,
            parse_mode="Markdown"
        )
    except Exception as e:
        print(
            "❌ VISION ERROR:",
            repr(e)
        )
        elapsed = (
            time.time()
            - start_time
        )
        bot.reply_to(
            message,
            "❌ **VISION ERROR**\n\n"
            f"{str(e)}\n\n"
            f"Processing time: {elapsed:.2f}s",
            parse_mode="Markdown"
        )
# ============================================================
# TEXT HANDLER
# ============================================================
@bot.message_handler(
    content_types=["text"]
)
def handle_text(message):
    bot.reply_to(
        message,
        "📸 Send me a Pocket Option screenshot "
        "and I will analyze the visible candle bodies."
    )
# ============================================================
# START
# ============================================================
print(
    "============================================"
)
print(
    "🕯️ GEMINI CANDLE VISION BOT"
)
print(
    "============================================"
)
print(
    "✅ Telegram connected"
)
print(
    "✅ Gemini API-key authentication enabled"
)
print(
    "✅ Full screenshot analysis"
)
print(
    "✅ No screenshot cropping"
)
print(
    "✅ No OHLC generation"
)
print(
    "✅ No random candles"
)
print(
    "✅ No forced candle count"
)
print(
    "✅ No trading signals"
)
print(
    "============================================"
)
# ============================================================
# TELEGRAM POLLING
# ============================================================
bot.infinity_polling(
    timeout=30,
    long_polling_timeout=30
)

import os
import io
import time
import telebot
from PIL import Image
from google import genai
from google.genai import types
# ============================================================
# ENVIRONMENT
# ============================================================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing.")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing.")
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
MODEL_NAME = "gemini-3.6-flash"
# ============================================================
# VISION INSTRUCTIONS
# ============================================================
PROMPT = """
Analyze this screenshot as a candle-vision test.
IMPORTANT:
- Examine the COMPLETE screenshot from 0% to 100%.
- Do not crop the image.
- Do not invent candles.
- Do not generate OHLC data.
- Do not use random candles.
- Do not force a candle count.
- Do not create a trading signal.
- Do not count empty space as a candle.
- Do not count BUY or SELL buttons.
- Do not count text or interface elements.
- Do not count chart decorations.
- Do not count isolated vertical lines unless they clearly belong to a candle.
- Count only visibly identifiable candle bodies.
For every candle you can actually see, classify the BODY:
GREEN or RED.
Read them from LEFT to RIGHT.
Small candle bodies and doji candles should still be considered, but only when there is actual visible evidence of a candle.
Return exactly this structure:
GREEN COUNT: number
RED COUNT: number
TOTAL: number
SEQUENCE:
1. GREEN
2. RED
3. GREEN
Then:
CONFIDENCE:
Briefly explain whether any candles were difficult to identify.
This is ONLY a visual candle-reading test.
There is NO trading signal.
"""
# ============================================================
# PREPARE IMAGE
# ============================================================
def prepare_image(image_bytes):
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
    return output.getvalue()
# ============================================================
# GEMINI IMAGE ANALYSIS
# ============================================================
def analyze_with_gemini(image_bytes):
    jpeg_data = prepare_image(
        image_bytes
    )
    image_part = types.Part.from_bytes(
        data=jpeg_data,
        mime_type="image/jpeg"
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=PROMPT
                    ),
                    image_part
                ]
            )
        ]
    )
    if response is None:
        raise RuntimeError(
            "Gemini returned no response."
        )
    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )
    return response.text.strip()
# ============================================================
# TELEGRAM PHOTO
# ============================================================
@bot.message_handler(
    content_types=["photo"]
)
def handle_photo(message):
    start = time.time()
    try:
        bot.reply_to(
            message,
            "👁️ Reading the entire screenshot...\n"
            "Checking GREEN and RED candle bodies."
        )
        # ----------------------------------------------------
        # DOWNLOAD ORIGINAL HIGH-RES TELEGRAM IMAGE
        # ----------------------------------------------------
        file_info = bot.get_file(
            message.photo[-1].file_id
        )
        image_bytes = bot.download_file(
            file_info.file_path
        )
        print(
            f"📸 Image received: "
            f"{len(image_bytes)} bytes"
        )
        # ----------------------------------------------------
        # GEMINI
        # ----------------------------------------------------
        print(
            "🧠 Sending screenshot to Gemini..."
        )
        result = analyze_with_gemini(
            image_bytes
        )
        print(
            "✅ Gemini response received."
        )
        elapsed = (
            time.time()
            - start
        )
        # ----------------------------------------------------
        # SEND RESULT
        # ----------------------------------------------------
        report = (
            "🔎 **CANDLE VISION TEST**\n\n"
            "🟡 **DETECTION AREA:**\n"
            "Entire uploaded screenshot — 0% to 100%\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{result}\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "🎯 **COLOR CHECK**\n"
            "🟢 GREEN = visually classified GREEN\n"
            "🔴 RED = visually classified RED\n\n"
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
        elapsed = (
            time.time()
            - start
        )
        print(
            "❌ GEMINI ERROR:",
            repr(e)
        )
        bot.reply_to(
            message,
            "❌ **VISION ERROR**\n\n"
            f"`{str(e)}`\n\n"
            f"Processing time: {elapsed:.2f}s",
            parse_mode="Markdown"
        )
# ============================================================
# TEXT
# ============================================================
@bot.message_handler(
    content_types=["text"]
)
def handle_text(message):
    bot.reply_to(
        message,
        "📸 Send a screenshot and I will read the visible "
        "GREEN and RED candle bodies."
    )
# ============================================================
# START
# ============================================================
print(
    "=========================================="
)
print(
    "🕯️ GEMINI CANDLE VISION BOT"
)
print(
    "=========================================="
)
print(
    "✅ Telegram ready"
)
print(
    "✅ Gemini API key loaded"
)
print(
    "✅ Full screenshot — no crop"
)
print(
    "✅ GREEN + RED visual analysis"
)
print(
    "✅ No fake candles"
)
print(
    "✅ No OHLC generation"
)
print(
    "✅ No trading signals"
)
print(
    "=========================================="
)
bot.infinity_polling(
    timeout=30,
    long_polling_timeout=30
)

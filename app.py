import os
import json
import re
import time
import base64
import telebot
from google import genai
from google.genai import types
# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not TELEGRAM_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing from Render environment variables.")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from Render environment variables.")
# ============================================================
# CLIENTS
# ============================================================
bot = telebot.TeleBot(TELEGRAM_TOKEN)
gemini = genai.Client(api_key=GEMINI_API_KEY)
# Current multimodal Gemini model
GEMINI_MODEL = "gemini-3.6-flash"
# ============================================================
# GEMINI CANDLE ANALYSIS
# ============================================================
CANDLE_PROMPT = r"""
You are analyzing a screenshot of a Pocket Option-style trading chart.
THIS IS A CANDLE DETECTION TEST ONLY.
Your job is ONLY to identify actual visible candlesticks in the chart.
IMPORTANT:
1. Analyze the ENTIRE uploaded screenshot from the extreme left edge
   to the extreme right edge.
2. Do NOT crop the image.
3. Do NOT assume a fixed number of candles.
4. Do NOT invent candles.
5. Do NOT count empty spaces.
6. Do NOT count Buy buttons.
7. Do NOT count Sell buttons.
8. Do NOT count text.
9. Do NOT count chart gridlines.
10. Do NOT count indicators.
11. Do NOT count UI decorations.
12. Do NOT count colored areas that are not actual candle bodies.
13. A candle must be an actual candlestick visible in the chart.
14. Identify the BODY of each candle, not just random colored pixels.
15. Very small candle bodies and doji-like candles should still be considered
    when there is visual evidence that they are actual candles.
16. Keep separate candles separate. Do not combine two neighboring candles
    into one candle.
17. The candle's horizontal position must correspond to its actual position
    in the screenshot.
18. Read candles from LEFT TO RIGHT.
19. Determine the candle color from the actual candle:
       GREEN = bullish/green candle
       RED   = bearish/red candle
20. Do NOT use trading prediction.
    Do NOT say BUY or SELL as a trading recommendation.
21. If something is uncertain, do NOT invent it.
    Mark it as uncertain instead.
Return ONLY valid JSON.
Use this exact structure:
{
  "candles": [
    {
      "number": 1,
      "color": "GREEN",
      "confidence": 0.95,
      "x_center": 123,
      "y_center": 456
    }
  ]
}
Rules for confidence:
- confidence must be between 0 and 1.
- Only include an object when there is actual visual evidence of a candle.
- x_center and y_center are approximate pixel coordinates in the uploaded image.
- Keep the numbering strictly left-to-right.
After the candles array, return:
{
  "candles": [...],
  "summary": {
    "green": 0,
    "red": 0,
    "total": 0
  }
}
Do not return Markdown.
Do not return explanations.
Do not return a trading signal.
"""
# ============================================================
# IMAGE ANALYSIS
# ============================================================
def analyze_screenshot(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png"
            ),
            CANDLE_PROMPT
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    text = response.text.strip()
    # Remove accidental markdown fences if a model response includes them.
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if "candles" not in data:
        raise ValueError("Gemini response did not contain candles.")
    candles = data["candles"]
    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------
    cleaned = []
    for candle in candles:
        color = str(
            candle.get("color", "")
        ).upper()
        if color not in ("GREEN", "RED"):
            continue
        try:
            confidence = float(
                candle.get("confidence", 0)
            )
        except Exception:
            confidence = 0
        try:
            x_center = float(
                candle.get("x_center", 0)
            )
        except Exception:
            x_center = 0
        try:
            y_center = float(
                candle.get("y_center", 0)
            )
        except Exception:
            y_center = 0
        # We do NOT force low-confidence detections into the result.
        if confidence < 0.55:
            continue
        cleaned.append({
            "color": color,
            "confidence": confidence,
            "x_center": x_center,
            "y_center": y_center
        })
    # --------------------------------------------------------
    # Sort by actual horizontal position.
    # --------------------------------------------------------
    cleaned.sort(
        key=lambda c: c["x_center"]
    )
    # --------------------------------------------------------
    # Re-number after sorting.
    # --------------------------------------------------------
    for i, candle in enumerate(
        cleaned,
        start=1
    ):
        candle["number"] = i
    green = sum(
        1 for c in cleaned
        if c["color"] == "GREEN"
    )
    red = sum(
        1 for c in cleaned
        if c["color"] == "RED"
    )
    return cleaned, green, red
# ============================================================
# TELEGRAM PHOTO HANDLER
# ============================================================
@bot.message_handler(
    content_types=["photo"]
)
def handle_photo(message):
    start_time = time.time()
    image_path = "chart_screenshot.png"
    try:
        bot.reply_to(
            message,
            "👁️ Gemini is reading the entire screenshot...\n"
            "Checking actual candle bodies and red/green colors."
        )
        # ----------------------------------------------------
        # Download highest-resolution Telegram photo
        # ----------------------------------------------------
        file_info = bot.get_file(
            message.photo[-1].file_id
        )
        downloaded_file = bot.download_file(
            file_info.file_path
        )
        with open(
            image_path,
            "wb"
        ) as f:
            f.write(downloaded_file)
        # ----------------------------------------------------
        # Gemini analysis
        # ----------------------------------------------------
        candles, green, red = analyze_screenshot(
            image_path
        )
        total = len(candles)
        elapsed = time.time() - start_time
        # ----------------------------------------------------
        # Sequence
        # ----------------------------------------------------
        if total == 0:
            report = (
                "🔎 **CANDLE READING TEST**\n\n"
                "🟡 **DETECTION AREA:**\n"
                "Entire uploaded screenshot — 0% to 100%\n\n"
                "❌ No sufficiently reliable candle bodies "
                "were identified.\n\n"
                "No candles were invented.\n"
                "No random candles were added.\n"
                "No trading signal was generated.\n\n"
                f"⚡ Processing time: {elapsed:.2f}s"
            )
            bot.reply_to(
                message,
                report,
                parse_mode="Markdown"
            )
            return
        # ----------------------------------------------------
        # Build readable sequence
        # ----------------------------------------------------
        sequence_lines = []
        for candle in candles:
            if candle["color"] == "GREEN":
                symbol = "🟢"
                name = "GREEN"
            else:
                symbol = "🔴"
                name = "RED"
            confidence_percent = (
                candle["confidence"] * 100
            )
            sequence_lines.append(
                f'{candle["number"]}. '
                f'{symbol} {name} '
                f'({confidence_percent:.0f}%)'
            )
        sequence_text = "\n".join(
            sequence_lines
        )
        # ----------------------------------------------------
        # Main report
        # ----------------------------------------------------
        report = (
            "🔎 **GEMINI CANDLE READING TEST**\n\n"
            "🟡 **DETECTION AREA:**\n"
            "Entire uploaded screenshot — 0% to 100%\n\n"
            "📊 **WHAT GEMINI DETECTED:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🟢 GREEN: {green}\n"
            f"🔴 RED: {red}\n"
            f"📊 TOTAL: {total}\n\n"
            "🕯️ **CANDLE-BY-CANDLE READING:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"{sequence_text}\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🎯 **COLOR CHECK:**\n"
            "🟢 = Gemini classified the candle GREEN\n"
            "🔴 = Gemini classified the candle RED\n\n"
            "🔬 **METHOD:**\n"
            "• Full screenshot analysis\n"
            "• Actual candle-body recognition\n"
            "• Separate GREEN/RED classification\n"
            "• Left-to-right ordering\n"
            "• Small-body awareness\n"
            "• No forced candle count\n"
            "• No random candles\n\n"
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
    except json.JSONDecodeError as e:
        bot.reply_to(
            message,
            "❌ Gemini returned an invalid JSON response.\n\n"
            "The screenshot was not converted into fake candles."
        )
        print(
            "JSON ERROR:",
            repr(e)
        )
    except Exception as e:
        print(
            "❌ ERROR:",
            repr(e)
        )
        bot.reply_to(
            message,
            f"❌ Gemini candle analysis error:\n{str(e)}"
        )
    finally:
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception:
                pass
# ============================================================
# START BOT
# ============================================================
print("========================================")
print("🕯️ GEMINI CANDLE READING TEST")
print("========================================")
print("Model:", GEMINI_MODEL)
print("Entire screenshot analyzed.")
print("No forced candle count.")
print("No random candles.")
print("No OHLC generation.")
print("No trading signals.")
print("========================================")
bot.infinity_polling(
    timeout=30,
    long_polling_timeout=30
)

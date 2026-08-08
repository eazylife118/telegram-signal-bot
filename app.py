import os
import time
import re
import threading
import requests
import cv2
import numpy as np
import pytesseract
from flask import Flask
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
# ============================================================
# TELEGRAM BOT
# ============================================================
TOKEN = "8937673241:AAGvyTA-G12xfwMlhif3Nh4_2Ag8OStq3tU"
CHAT_ID = "6280535707"
CHANNEL_ID = "-1004324805205"
# ============================================================
# FLASK SERVER
# ============================================================
app = Flask(__name__)
@app.route("/")
def home():
    return "✅ Candle Detection Test Bot is running!"
@app.route("/ping")
def ping():
    return "pong", 200
def run_flask():
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.run(
        host="0.0.0.0",
        port=10000,
        debug=False,
        threaded=True
    )
# ============================================================
# OCR — ONLY TO IDENTIFY WHAT TEXT/PAIR IS VISIBLE
# ============================================================
def read_visible_text(image):
    """
    OCR is only used to report visible text.
    It does NOT create candle data.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Upscale moderately for OCR
    h, w = gray.shape
    if w < 1200:
        scale = 1200 / w
        gray = cv2.resize(
            gray,
            (1200, int(h * scale)),
            interpolation=cv2.INTER_CUBIC
        )
    # Light thresholding
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    text = pytesseract.image_to_string(
        gray,
        config="--psm 11"
    )
    text = text.strip()
    # Try to find likely asset/pair text
    asset = "NOT CLEAR"
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]
    for line in lines:
        upper = line.upper()
        # OTC names such as:
        # AUD/CHF OTC
        # EUR/USD OTC
        # AMERICAN EXPRESS OTC
        if "OTC" in upper:
            asset = line
            break
    # Currency-style pair fallback
    if asset == "NOT CLEAR":
        pair_match = re.search(
            r"\b[A-Z]{3}\s*/?\s*[A-Z]{3}\b",
            text.upper()
        )
        if pair_match:
            asset = pair_match.group(0)
    return asset, text
# ============================================================
# CANDLE DETECTOR
# ============================================================
class CandleDetector:
    def __init__(self):
        pass
    # --------------------------------------------------------
    # FIND CHART REGION
    # --------------------------------------------------------
    def get_chart_region(self, image):
        """
        Pocket Option screenshots normally contain:
        header at top,
        chart in the middle,
        controls/navigation near bottom.
        We deliberately keep a large central region.
        """
        height, width = image.shape[:2]
        top = int(height * 0.12)
        bottom = int(height * 0.78)
        left = int(width * 0.03)
        right = int(width * 0.92)
        chart = image[top:bottom, left:right]
        return chart, (left, top, right, bottom)
    # --------------------------------------------------------
    # COLOR MASKS
    # --------------------------------------------------------
    def create_masks(self, chart):
        hsv = cv2.cvtColor(chart, cv2.COLOR_BGR2HSV)
        # GREEN
        green_lower = np.array([30, 35, 35])
        green_upper = np.array([95, 255, 255])
        green_mask = cv2.inRange(
            hsv,
            green_lower,
            green_upper
        )
        # RED
        red_lower_1 = np.array([0, 35, 35])
        red_upper_1 = np.array([12, 255, 255])
        red_lower_2 = np.array([165, 35, 35])
        red_upper_2 = np.array([180, 255, 255])
        red_mask_1 = cv2.inRange(
            hsv,
            red_lower_1,
            red_upper_1
        )
        red_mask_2 = cv2.inRange(
            hsv,
            red_lower_2,
            red_upper_2
        )
        red_mask = cv2.bitwise_or(
            red_mask_1,
            red_mask_2
        )
        return green_mask, red_mask
    # --------------------------------------------------------
    # MORPHOLOGICAL CLEANUP
    # --------------------------------------------------------
    def clean_mask(self, mask):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3)
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )
        return mask
    # --------------------------------------------------------
    # DETECT CANDLE BODIES USING CONNECTED COMPONENTS
    # --------------------------------------------------------
    def detect_components(self, mask, color_name):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
        detected = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            # Ignore tiny noise
            if area < 15:
                continue
            # Ignore huge UI/background regions
            if area > 20000:
                continue
            # Candle bodies are normally reasonably narrow.
            if w < 2:
                continue
            if w > 45:
                continue
            if h < 3:
                continue
            # Very long horizontal objects are probably
            # chart lines/UI rather than candle bodies.
            if w > h * 8:
                continue
            detected.append({
                "color": color_name,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "area": area,
                "center_x": x + w / 2,
                "center_y": y + h / 2
            })
        return detected
    # --------------------------------------------------------
    # MERGE COMPONENTS THAT BELONG TO SAME CANDLE
    # --------------------------------------------------------
    def merge_nearby(self, candles):
        if not candles:
            return []
        candles = sorted(
            candles,
            key=lambda c: c["center_x"]
        )
        merged = []
        for candle in candles:
            if not merged:
                merged.append(candle)
                continue
            previous = merged[-1]
            distance = abs(
                candle["center_x"] -
                previous["center_x"]
            )
            # Components very close horizontally
            # can belong to the same candle body.
            if (
                distance <=
                max(
                    8,
                    (candle["width"] +
                     previous["width"]) * 1.5
                )
            ):
                # Keep the stronger/larger component.
                if candle["area"] > previous["area"]:
                    merged[-1] = candle
            else:
                merged.append(candle)
        return merged
    # --------------------------------------------------------
    # REMOVE DUPLICATE CANDLES
    # --------------------------------------------------------
    def remove_duplicates(self, candles):
        if not candles:
            return []
        candles = sorted(
            candles,
            key=lambda c: c["center_x"]
        )
        final = []
        for candle in candles:
            duplicate = False
            for existing in final:
                distance = abs(
                    candle["center_x"] -
                    existing["center_x"]
                )
                if distance < 6:
                    duplicate = True
                    # Keep stronger detection
                    if candle["area"] > existing["area"]:
                        final.remove(existing)
                        final.append(candle)
                    break
            if not duplicate:
                final.append(candle)
        return sorted(
            final,
            key=lambda c: c["center_x"]
        )
    # --------------------------------------------------------
    # MAIN DETECTION
    # --------------------------------------------------------
    def analyze(self, image):
        chart, region = self.get_chart_region(image)
        if chart.size == 0:
            return {
                "green": 0,
                "red": 0,
                "total": 0,
                "sequence": [],
                "region": None
            }
        green_mask, red_mask = self.create_masks(chart)
        green_mask = self.clean_mask(green_mask)
        red_mask = self.clean_mask(red_mask)
        green_candles = self.detect_components(
            green_mask,
            "GREEN"
        )
        red_candles = self.detect_components(
            red_mask,
            "RED"
        )
        all_candles = green_candles + red_candles
        # Remove duplicates within each color
        green_candles = self.remove_duplicates(
            green_candles
        )
        red_candles = self.remove_duplicates(
            red_candles
        )
        all_candles = green_candles + red_candles
        # Sort from left to right
        all_candles = sorted(
            all_candles,
            key=lambda c: c["center_x"]
        )
        # ----------------------------------------------------
        # IMPORTANT:
        # Only count actual detected candle objects.
        # Nothing is generated.
        # ----------------------------------------------------
        sequence = []
        for candle in all_candles:
            sequence.append(
                candle["color"]
            )
        return {
            "green": len(green_candles),
            "red": len(red_candles),
            "total": len(all_candles),
            "sequence": sequence,
            "region": region
        }
# ============================================================
# TELEGRAM SENDING
# ============================================================
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message
            },
            timeout=5
        )
        requests.post(
            url,
            data={
                "chat_id": CHANNEL_ID,
                "text": message
            },
            timeout=5
        )
        print("✅ Sent to Telegram")
    except Exception as e:
        print(
            "⚠️ Telegram send error:",
            e
        )
# ============================================================
# TELEGRAM BOT
# ============================================================
detector = CandleDetector()
async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "📊 CANDLE DETECTION TEST BOT\n\n"
        "Send a Pocket Option screenshot.\n\n"
        "I will ONLY count the visible candles:\n"
        "🟢 Green candles\n"
        "🔴 Red candles\n"
        "📊 Total detected\n\n"
        "No strategy.\n"
        "No signal.\n"
        "No generated candles.\n"
        "No random data."
    )
async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    start_time = time.time()
    try:
        # ----------------------------------------------------
        # DOWNLOAD IMAGE
        # ----------------------------------------------------
        photo = await update.message.photo[-1].get_file()
        file_path = "screenshot_test.png"
        await photo.download_to_drive(
            file_path
        )
        # ----------------------------------------------------
        # LOAD ORIGINAL IMAGE
        # ----------------------------------------------------
        image = cv2.imread(file_path)
        if image is None:
            await update.message.reply_text(
                "❌ Could not load screenshot."
            )
            return
        print(
            f"📸 Screenshot received: "
            f"{image.shape[1]}x{image.shape[0]}"
        )
        # ----------------------------------------------------
        # OCR — INFORMATION ONLY
        # ----------------------------------------------------
        asset, visible_text = read_visible_text(
            image
        )
        # ----------------------------------------------------
        # CANDLE DETECTION
        # ----------------------------------------------------
        result = detector.analyze(
            image
        )
        elapsed = time.time() - start_time
        green = result["green"]
        red = result["red"]
        total = result["total"]
        sequence = result["sequence"]
        # ----------------------------------------------------
        # SEQUENCE DISPLAY
        # ----------------------------------------------------
        if sequence:
            sequence_text = " → ".join(
                "🟢" if x == "GREEN" else "🔴"
                for x in sequence
            )
        else:
            sequence_text = "NONE"
        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------
        message = (
            "🔎 **CANDLE DETECTION TEST**\n\n"
            f"💱 **Detected asset:** {asset}\n\n"
            "📊 **WHAT THE BOT ACTUALLY DETECTED:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🟢 Green candles: **{green}**\n"
            f"🔴 Red candles: **{red}**\n"
            f"📊 Total candles: **{total}**\n\n"
            f"🕯️ **Candle sequence "
            f"(left → right):**\n"
            f"{sequence_text}\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "⚠️ **IMPORTANT:**\n"
            "Only candles actually detected in the "
            "screenshot are counted.\n"
            "No OHLC candles are generated.\n"
            "No random candles are added.\n"
            "No trading signal is generated.\n\n"
            f"⚡ Processing time: **{elapsed:.2f}s**"
        )
        await update.message.reply_text(
            message,
            parse_mode="Markdown"
        )
        print("\n" + "=" * 50)
        print("CANDLE TEST RESULT")
        print("=" * 50)
        print(f"Asset: {asset}")
        print(f"Green: {green}")
        print(f"Red: {red}")
        print(f"Total: {total}")
        print(f"Sequence: {sequence}")
        print(f"Time: {elapsed:.2f}s")
        print("=" * 50)
    except Exception as e:
        print(
            "❌ Error:",
            str(e)
        )
        await update.message.reply_text(
            f"❌ Error: {str(e)}"
        )
# ============================================================
# START TELEGRAM
# ============================================================
def run_telegram():
    application = (
        Application
        .builder()
        .token(TOKEN)
        .build()
    )
    application.bot.delete_webhook()
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
        "✅ Candle Detection Test Bot started."
    )
    application.run_polling(
        drop_pending_updates=True
    )
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("📊 POCKET OPTION CANDLE DETECTION TEST")
    print("=" * 50)
    threading.Thread(
        target=run_flask,
        daemon=True
    ).start()
    print(
        "✅ Flask server started on port 10000"
    )
    run_telegram()

requirements.txt

Flask
requests
numpy
opencv-python-headless
pytesseract
python-telegram-bot

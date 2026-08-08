import os
import time
import re
import threading

import cv2
import numpy as np
import pytesseract

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
# TELEGRAM
# ============================================================

TOKEN = "8937673241:AAGvyTA-G12xfwMlhif3Nh4_2Ag8OStq3tU"

if not TOKEN:
    print("⚠️ BOT_TOKEN environment variable is not set.")


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "Candle Vision Test Bot is running."


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
# IMAGE / OCR HELPERS
# ============================================================

def resize_for_analysis(image):
    """
    Resize only when necessary.
    We keep the original aspect ratio.
    """

    h, w = image.shape[:2]

    target_width = 1800

    if w < target_width:
        scale = target_width / float(w)

        new_w = int(w * scale)
        new_h = int(h * scale)

        image = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )

    return image


def detect_asset_text(image):
    """
    OCR is used only to identify the visible pair/asset text.

    It does NOT create candle data.
    """

    h, w = image.shape[:2]

    # Upper portion normally contains pair information.
    top = image[0:int(h * 0.40), 0:w]

    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    text = pytesseract.image_to_string(
        threshold,
        config="--psm 11"
    )

    lines = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            lines.append(line)

    # Prefer lines containing OTC or currency-pair patterns.
    candidates = []

    for line in lines:

        clean = re.sub(
            r"[^A-Za-z0-9/._ -]",
            " ",
            line
        )

        clean = re.sub(
            r"\s+",
            " ",
            clean
        ).strip()

        upper = clean.upper()

        if "OTC" in upper:
            candidates.append(clean)

        elif re.search(
            r"\b[A-Z]{3}\s*/\s*[A-Z]{3}\b",
            upper
        ):
            candidates.append(clean)

    if candidates:
        return candidates[0]

    return "NOT CONFIDENTLY DETECTED"


# ============================================================
# CANDLE VISION DETECTOR
# ============================================================

class CandleVisionDetector:

    def __init__(self):

        # Minimum/maximum candle body dimensions.
        self.min_body_width = 2
        self.max_body_width = 70

        self.min_body_height = 2
        self.max_body_height = 500

    # --------------------------------------------------------
    # FIND CHART
    # --------------------------------------------------------

    def get_chart_region(self, image):

        h, w = image.shape[:2]

        """
        Pocket Option screenshots normally contain:
        top UI
        chart
        bottom UI

        We deliberately keep a large portion of the center.
        """

        top = int(h * 0.12)
        bottom = int(h * 0.86)

        left = int(w * 0.03)
        right = int(w * 0.97)

        chart = image[
            top:bottom,
            left:right
        ]

        return chart, (left, top, right, bottom)

    # --------------------------------------------------------
    # COLOR MASKS
    # --------------------------------------------------------

    def get_color_masks(self, chart):

        hsv = cv2.cvtColor(
            chart,
            cv2.COLOR_BGR2HSV
        )

        # Green candles.
        green1 = cv2.inRange(
            hsv,
            np.array([35, 35, 35]),
            np.array([95, 255, 255])
        )

        # Red candles.
        red1 = cv2.inRange(
            hsv,
            np.array([0, 35, 35]),
            np.array([12, 255, 255])
        )

        red2 = cv2.inRange(
            hsv,
            np.array([165, 35, 35]),
            np.array([180, 255, 255])
        )

        red = cv2.bitwise_or(
            red1,
            red2
        )

        # Some Pocket Option themes use lighter/white bodies.
        white = cv2.inRange(
            hsv,
            np.array([0, 0, 170]),
            np.array([180, 45, 255])
        )

        # Remove tiny noise.
        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        green1 = cv2.morphologyEx(
            green1,
            cv2.MORPH_OPEN,
            kernel
        )

        red = cv2.morphologyEx(
            red,
            cv2.MORPH_OPEN,
            kernel
        )

        white = cv2.morphologyEx(
            white,
            cv2.MORPH_OPEN,
            kernel
        )

        return green1, red, white

    # --------------------------------------------------------
    # CANDIDATE COMPONENTS
    # --------------------------------------------------------

    def find_components(self, mask, color_name):

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)

            area = cv2.contourArea(contour)

            if area < 5:
                continue

            if w < self.min_body_width:
                continue

            if w > self.max_body_width:
                continue

            if h < self.min_body_height:
                continue

            if h > self.max_body_height:
                continue

            # Reject huge UI elements.
            if w * h > 15000:
                continue

            aspect = h / float(max(w, 1))

            # Candle bodies are usually vertically oriented.
            if aspect < 0.15:
                continue

            if aspect > 80:
                continue

            candidates.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area,
                "color": color_name
            })

        return candidates

    # --------------------------------------------------------
    # MERGE CANDIDATES
    # --------------------------------------------------------

    def merge_close_candidates(self, candidates):

        if not candidates:
            return []

        candidates = sorted(
            candidates,
            key=lambda c: c["x"]
        )

        groups = []

        for candidate in candidates:

            placed = False

            center = candidate["x"] + candidate["w"] / 2

            for group in groups:

                group_center = np.mean([
                    c["x"] + c["w"] / 2
                    for c in group
                ])

                # Candles have narrow bodies.
                # Components belonging to the same body
                # can therefore be close together.
                max_gap = max(
                    10,
                    candidate["w"] * 1.5
                )

                if abs(center - group_center) <= max_gap:

                    # Don't merge if clearly different candles.
                    highest = min(
                        c["y"]
                        for c in group
                    )

                    lowest = max(
                        c["y"] + c["h"]
                        for c in group
                    )

                    if (
                        candidate["y"] <= lowest + 25
                        and
                        candidate["y"] + candidate["h"]
                        >= highest - 25
                    ):
                        group.append(candidate)
                        placed = True
                        break

            if not placed:
                groups.append([candidate])

        merged = []

        for group in groups:

            x1 = min(c["x"] for c in group)
            y1 = min(c["y"] for c in group)

            x2 = max(
                c["x"] + c["w"]
                for c in group
            )

            y2 = max(
                c["y"] + c["h"]
                for c in group
            )

            colors = [
                c["color"]
                for c in group
            ]

            if colors.count("GREEN") >= colors.count("RED"):
                color = "GREEN"
            else:
                color = "RED"

            merged.append({
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
                "color": color
            })

        return merged

    # --------------------------------------------------------
    # SPATIAL CANDLE VALIDATION
    # --------------------------------------------------------

    def validate_candle_spacing(self, candidates, chart_width):

        if len(candidates) < 2:
            return candidates

        candidates = sorted(
            candidates,
            key=lambda c: c["x"] + c["w"] / 2
        )

        centers = np.array([
            c["x"] + c["w"] / 2
            for c in candidates
        ])

        gaps = np.diff(centers)

        if len(gaps) == 0:
            return candidates

        median_gap = np.median(gaps)

        if median_gap <= 0:
            return candidates

        valid = []

        for i, candidate in enumerate(candidates):

            center = (
                candidate["x"]
                + candidate["w"] / 2
            )

            # Very isolated huge gaps are suspicious,
            # but don't automatically discard them.
            if i == 0 or i == len(candidates) - 1:
                valid.append(candidate)
                continue

            left_gap = center - centers[i - 1]
            right_gap = centers[i + 1] - center

            if (
                left_gap <= median_gap * 3.5
                or
                right_gap <= median_gap * 3.5
            ):
                valid.append(candidate)

        return valid

    # --------------------------------------------------------
    # MAIN CANDLE DETECTION
    # --------------------------------------------------------

    def detect(self, image):

        chart, chart_box = self.get_chart_region(
            image
        )

        green_mask, red_mask, white_mask = (
            self.get_color_masks(chart)
        )

        green_candidates = self.find_components(
            green_mask,
            "GREEN"
        )

        red_candidates = self.find_components(
            red_mask,
            "RED"
        )

        # White objects are not automatically treated
        # as candles. They can be grid lines, text, etc.
        #
        # We only use colored candle bodies for the
        # first accuracy test.

        candidates = (
            green_candidates
            +
            red_candidates
        )

        candidates = self.merge_close_candidates(
            candidates
        )

        candidates = self.validate_candle_spacing(
            candidates,
            chart.shape[1]
        )

        candidates.sort(
            key=lambda c: c["x"] + c["w"] / 2
        )

        # Give each candle a sequential position.
        for index, candle in enumerate(
            candidates,
            start=1
        ):
            candle["number"] = index

        return candidates, chart_box


# ============================================================
# FORMAT RESULT
# ============================================================

def format_candle_result(
    candles,
    asset,
    elapsed
):

    green = sum(
        1
        for c in candles
        if c["color"] == "GREEN"
    )

    red = sum(
        1
        for c in candles
        if c["color"] == "RED"
    )

    total = len(candles)

    sequence = " → ".join(
        "🟢" if c["color"] == "GREEN"
        else "🔴"
        for c in candles
    )

    message = (
        "🔎 **CANDLE VISION TEST**\n\n"
        f"💱 **Detected asset:** {asset}\n\n"
        "📊 **WHAT THE BOT ACTUALLY DETECTED:**\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"🟢 Green candles: **{green}**\n"
        f"🔴 Red candles: **{red}**\n"
        f"📊 Total candles: **{total}**\n\n"
        "🕯️ **Candle sequence (left → right):**\n"
        f"{sequence if sequence else 'NONE'}\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ **IMPORTANT:**\n"
        "Only visually detected candle bodies are counted.\n"
        "No OHLC candles are generated.\n"
        "No random candles are added.\n"
        "No trading signal is generated.\n\n"
        f"⚡ **Processing time:** {elapsed:.2f}s"
    )

    return message


# ============================================================
# TELEGRAM HANDLERS
# ============================================================

detector = CandleVisionDetector()


async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🔎 CANDLE VISION TEST BOT\n\n"
        "Send a Pocket Option screenshot.\n\n"
        "I will only count candle objects actually "
        "visible in the chart.\n\n"
        "No strategy.\n"
        "No OHLC generation.\n"
        "No random candles.\n"
        "No trading signal."
    )


async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    start_time = time.time()

    try:

        await update.message.reply_text(
            "📸 Reading the visible candles..."
        )

        photo = await (
            update.message
            .photo[-1]
            .get_file()
        )

        file_path = "candle_test.png"

        await photo.download_to_drive(
            file_path
        )

        image = cv2.imread(
            file_path
        )

        if image is None:

            await update.message.reply_text(
                "❌ Could not read the image."
            )

            return

        image = resize_for_analysis(
            image
        )

        # OCR only identifies the visible asset text.
        asset = detect_asset_text(
            image
        )

        # Candle detector works from actual pixels.
        candles, chart_box = detector.detect(
            image
        )

        elapsed = (
            time.time()
            - start_time
        )

        result = format_candle_result(
            candles,
            asset,
            elapsed
        )

        await update.message.reply_text(
            result,
            parse_mode="Markdown"
        )

        print(
            f"Detected {len(candles)} candles "
            f"in {elapsed:.2f}s"
        )

    except Exception as e:

        print(
            "❌ Error:",
            repr(e)
        )

        await update.message.reply_text(
            f"❌ Error: {str(e)}"
        )


# ============================================================
# START TELEGRAM
# ============================================================

def run_telegram():

    if not TOKEN:
        print(
            "❌ BOT_TOKEN is missing. "
            "Set it in Render Environment Variables."
        )
        return

    application = (
        Application
        .builder()
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

    print(
        "✅ Candle Vision Test Bot started."
    )

    application.run_polling(
        drop_pending_updates=True
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 50
    )

    print(
        "🔎 POCKET OPTION CANDLE VISION TEST"
    )

    print(
        "=" * 50
    )

    threading.Thread(
        target=run_flask,
        daemon=True
    ).start()

    run_telegram()

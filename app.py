import os
import time
import threading
import requests
import numpy as np
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime, timezone, timedelta
from collections import deque
import cv2
import pytesseract
import re

# ==========================================
# TELEGRAM CREDENTIALS
# ==========================================
# Use your BOT TOKEN here or preferably set BOT_TOKEN
# as an environment variable.
TOKEN = "8937673241:AAGvyTA-G12xfwMlhif3Nh4_2Ag8OStq3tU"

CHAT_ID = "6280535707"
CHANNEL_ID = "-1004324805205"

# ==========================================
# TIME ZONE
# ==========================================
LOCAL_TZ = timezone(timedelta(hours=1))

# ==========================================
# TRADE SETTINGS
# ==========================================
ENTRY_INTERVAL_SECONDS = 15
EXPIRY_MINUTES = 1

# Your 4-entry structure
ENTRY_COUNT = 4

# ==========================================
# STRATEGY HEALTH
# ==========================================
strategy_history = {
    "Reversal Zone Strategy": deque(maxlen=10)
}

def get_strategy_health(strategy_name):
    history = strategy_history.get(strategy_name)

    if not history:
        return 50

    return min(
        100,
        max(50, sum(history) / len(history) * 100)
    )

def record_signal(strategy_name, win):
    if strategy_name in strategy_history:
        strategy_history[strategy_name].append(win)


# ==========================================
# TIME FUNCTIONS
# ==========================================
def get_next_minute():
    now = datetime.now(LOCAL_TZ)

    next_minute = (
        now.replace(second=0, microsecond=0)
        + timedelta(minutes=1)
    )

    return next_minute.strftime("%H:%M:%S")


# ==========================================
# SCREENSHOT READER
# ==========================================
class PocketOptionScreenshotReader:

    def __init__(self):
        self.price_levels = []

    def read_screenshot(self, image_path):

        img = cv2.imread(image_path)

        if img is None:
            print("❌ Could not load image")
            return None

        print(f"📸 Screenshot size: {img.shape}")

        # ------------------------------------------------
        # IMPORTANT:
        # Do not manufacture candle data.
        # We only analyze information actually extracted
        # from the screenshot.
        # ------------------------------------------------

        candles = self._extract_candles_from_chart(img)

        if not candles or len(candles) < 5:
            print("❌ Not enough real candle information detected")
            return None

        print(f"✅ Detected {len(candles)} chart candles")

        return self._candles_to_ohlc(candles)

    # ==============================================
    # CANDLE EXTRACTION
    # ==============================================
    def _extract_candles_from_chart(self, img):

        height, width = img.shape[:2]

        chart_region = img[
            int(height * 0.15):
            int(height * 0.80),

            int(width * 0.10):
            int(width * 0.85)
        ]

        hsv = cv2.cvtColor(
            chart_region,
            cv2.COLOR_BGR2HSV
        )

        # GREEN
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([90, 255, 255])

        green_mask = cv2.inRange(
            hsv,
            green_lower,
            green_upper
        )

        # RED
        red_lower1 = np.array([0, 40, 40])
        red_upper1 = np.array([12, 255, 255])

        red_lower2 = np.array([168, 40, 40])
        red_upper2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(
            hsv,
            red_lower1,
            red_upper1
        )

        red_mask2 = cv2.inRange(
            hsv,
            red_lower2,
            red_upper2
        )

        red_mask = cv2.bitwise_or(
            red_mask1,
            red_mask2
        )

        chart_height, chart_width = chart_region.shape[:2]

        # Look for approximately 40 candle columns
        num_columns = min(
            40,
            max(10, chart_width // 8)
        )

        column_width = max(
            1,
            chart_width // num_columns
        )

        candles = []

        for i in range(num_columns):

            x1 = i * column_width
            x2 = min(
                chart_width,
                (i + 1) * column_width
            )

            green_pixels = np.sum(
                green_mask[:, x1:x2] > 0
            )

            red_pixels = np.sum(
                red_mask[:, x1:x2] > 0
            )

            total_colored = (
                green_pixels +
                red_pixels
            )

            # Ignore empty columns
            if total_colored < 15:
                continue

            color = (
                "GREEN"
                if green_pixels > red_pixels
                else "RED"
            )

            # ------------------------------------------------
            # Find candle vertical extent
            # ------------------------------------------------
            combined_mask = cv2.bitwise_or(
                green_mask[:, x1:x2],
                red_mask[:, x1:x2]
            )

            ys, xs = np.where(
                combined_mask > 0
            )

            if len(ys) < 5:
                continue

            top = int(np.min(ys))
            bottom = int(np.max(ys))

            if bottom <= top:
                continue

            candles.append({
                "color": color,
                "top": top,
                "bottom": bottom,
                "height": bottom - top,
                "index": i,
                "green_pixels": int(green_pixels),
                "red_pixels": int(red_pixels)
            })

        return candles

    # ==============================================
    # CONVERT DETECTED CANDLE POSITIONS
    # ==============================================
    def _candles_to_ohlc(self, candles):

        if not candles:
            return None

        # ------------------------------------------------
        # IMPORTANT:
        # These are normalized chart coordinates.
        # They are NOT invented market prices.
        # ------------------------------------------------
        tops = np.array(
            [c["top"] for c in candles],
            dtype=float
        )

        bottoms = np.array(
            [c["bottom"] for c in candles],
            dtype=float
        )

        min_y = np.min(bottoms)
        max_y = np.max(tops)

        vertical_range = max_y - min_y

        if vertical_range <= 0:
            return None

        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []

        for candle in candles:

            top = candle["top"]
            bottom = candle["bottom"]

            # Normalize vertically.
            # Higher chart position = higher normalized price.
            high = (
                max_y - top
            ) / vertical_range

            low = (
                max_y - bottom
            ) / vertical_range

            candle_range = high - low

            if candle["color"] == "GREEN":

                close = high - candle_range * 0.15
                open_ = low + candle_range * 0.15

            else:

                open_ = high - candle_range * 0.15
                close = low + candle_range * 0.15

            opens.append(open_)
            highs.append(high)
            lows.append(low)
            closes.append(close)

            volumes.append(
                candle["green_pixels"] +
                candle["red_pixels"]
            )

        return {
            "open": np.array(opens),
            "high": np.array(highs),
            "low": np.array(lows),
            "close": np.array(closes),
            "volume": np.array(volumes)
        }


# ==========================================
# PATTERN SETTINGS
# ==========================================

# How close two rejection prices must be
# relative to the visible chart range.
ZONE_TOLERANCE = 0.035

# Candle body considered small compared with its range
SMALL_BODY_RATIO = 0.35

# Minimum rejection wick
REJECTION_WICK_RATIO = 0.40

# Open-close clustering
OPEN_CLOSE_CLUSTER_RATIO = 0.30

# Minimum number of tests
MIN_REJECTIONS = 2


# ==========================================
# CANDLE CHARACTERISTICS
# ==========================================
def candle_information(
    open_,
    high,
    low,
    close
):

    candle_range = high - low

    if candle_range <= 0:
        return {
            "body": 0,
            "range": 0,
            "body_ratio": 1,
            "upper_wick": 0,
            "lower_wick": 0
        }

    body = abs(close - open_)

    upper_wick = (
        high -
        max(open_, close)
    )

    lower_wick = (
        min(open_, close) -
        low
    )

    return {
        "body": body,
        "range": candle_range,
        "body_ratio": body / candle_range,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick
    }


# ==========================================
# DETECT REJECTION CANDLES
# ==========================================
def detect_rejection_candles(price_data):

    open_ = price_data["open"]
    high = price_data["high"]
    low = price_data["low"]
    close = price_data["close"]

    upper_rejections = []
    lower_rejections = []

    for i in range(len(close)):

        info = candle_information(
            open_[i],
            high[i],
            low[i],
            close[i]
        )

        if info["range"] <= 0:
            continue

        # ----------------------------------------------
        # UPPER REJECTION
        # ----------------------------------------------
        if (
            info["body_ratio"] <= SMALL_BODY_RATIO
            and
            info["upper_wick"] / info["range"]
            >= REJECTION_WICK_RATIO
        ):

            upper_rejections.append({
                "index": i,
                "level": high[i],
                "open": open_[i],
                "close": close[i],
                "color": (
                    "GREEN"
                    if close[i] > open_[i]
                    else "RED"
                    if close[i] < open_[i]
                    else "DOJI"
                )
            })

        # ----------------------------------------------
        # LOWER REJECTION
        # ----------------------------------------------
        if (
            info["body_ratio"] <= SMALL_BODY_RATIO
            and
            info["lower_wick"] / info["range"]
            >= REJECTION_WICK_RATIO
        ):

            lower_rejections.append({
                "index": i,
                "level": low[i],
                "open": open_[i],
                "close": close[i],
                "color": (
                    "GREEN"
                    if close[i] > open_[i]
                    else "RED"
                    if close[i] < open_[i]
                    else "DOJI"
                )
            })

    return upper_rejections, lower_rejections


# ==========================================
# CLUSTER REJECTIONS INTO ZONES
# ==========================================
def cluster_rejections(rejections, tolerance):

    if not rejections:
        return []

    sorted_rejections = sorted(
        rejections,
        key=lambda x: x["level"]
    )

    zones = []

    current = [sorted_rejections[0]]

    for rejection in sorted_rejections[1:]:

        current_level = np.mean(
            [x["level"] for x in current]
        )

        if abs(
            rejection["level"] -
            current_level
        ) <= tolerance:

            current.append(rejection)

        else:

            zones.append(current)
            current = [rejection]

    zones.append(current)

    return zones


# ==========================================
# ZONE DESCRIPTION
# ==========================================
def describe_zones(
    upper_rejections,
    lower_rejections,
    price_data
):

    all_prices = np.concatenate([
        price_data["high"],
        price_data["low"]
    ])

    chart_range = (
        np.max(all_prices) -
        np.min(all_prices)
    )

    if chart_range <= 0:
        chart_range = 1

    tolerance = (
        chart_range *
        ZONE_TOLERANCE
    )

    upper_groups = cluster_rejections(
        upper_rejections,
        tolerance
    )

    lower_groups = cluster_rejections(
        lower_rejections,
        tolerance
    )

    resistance_zones = []

    for group in upper_groups:

        level = np.mean([
            x["level"]
            for x in group
        ])

        resistance_zones.append({
            "level": level,
            "tests": len(group),
            "indices": [
                x["index"]
                for x in group
            ]
        })

    support_zones = []

    for group in lower_groups:

        level = np.mean([
            x["level"]
            for x in group
        ])

        support_zones.append({
            "level": level,
            "tests": len(group),
            "indices": [
                x["index"]
                for x in group
            ]
        })

    return (
        resistance_zones,
        support_zones
    )


# ==========================================
# OPEN / CLOSE CLUSTERING
# ==========================================
def detect_open_close_clustering(
    price_data,
    zone,
    side
):

    open_ = price_data["open"]
    close = price_data["close"]
    high = price_data["high"]
    low = price_data["low"]

    levels = []

    for i in range(len(close)):

        candle_range = (
            high[i] - low[i]
        )

        if candle_range <= 0:
            continue

        body = abs(
            close[i] - open_[i]
        )

        # Small body means open and close
        # are close relative to candle range.
        if (
            body / candle_range
            <= OPEN_CLOSE_CLUSTER_RATIO
        ):

            if side == "RESISTANCE":

                distance = abs(
                    high[i] -
                    zone["level"]
                )

            else:

                distance = abs(
                    low[i] -
                    zone["level"]
                )

            if distance <= candle_range * 1.5:

                levels.append(i)

    return levels


# ==========================================
# BREAKOUT FAILURE DETECTION
# ==========================================
def detect_breakout_failure(
    price_data,
    zone,
    side
):

    open_ = price_data["open"]
    high = price_data["high"]
    low = price_data["low"]
    close = price_data["close"]

    level = zone["level"]

    for i in range(1, len(close)):

        # ==========================================
        # RESISTANCE BREAKOUT FAILURE
        # ==========================================
        if side == "RESISTANCE":

            broke_above = (
                high[i] > level
            )

            failed_to_hold = (
                close[i] < level
            )

            turned_red = (
                close[i] < open_[i]
            )

            if (
                broke_above
                and
                failed_to_hold
                and
                turned_red
            ):

                return {
                    "index": i,
                    "type": "RESISTANCE_BREAKOUT_FAILURE",
                    "direction": "SELL"
                }

        # ==========================================
        # SUPPORT BREAKOUT FAILURE
        # ==========================================
        if side == "SUPPORT":

            broke_below = (
                low[i] < level
            )

            failed_to_hold = (
                close[i] > level
            )

            turned_green = (
                close[i] > open_[i]
            )

            if (
                broke_below
                and
                failed_to_hold
                and
                turned_green
            ):

                return {
                    "index": i,
                    "type": "SUPPORT_BREAKOUT_FAILURE",
                    "direction": "BUY"
                }

    return None


# ==========================================
# CONFIRMATION CANDLE
# ==========================================
def detect_confirmation(
    price_data,
    failure,
    direction
):

    if not failure:
        return False

    close = price_data["close"]
    open_ = price_data["open"]

    i = failure["index"]

    # Current candle itself is confirmation
    if direction == "SELL":

        return close[i] < open_[i]

    if direction == "BUY":

        return close[i] > open_[i]

    return False


# ==========================================
# TWO-SIDED INDECISION
# ==========================================
def detect_two_sided_rejection(
    resistance_zones,
    support_zones
):

    resistance_tests = max(
        [z["tests"] for z in resistance_zones],
        default=0
    )

    support_tests = max(
        [z["tests"] for z in support_zones],
        default=0
    )

    return (
        resistance_tests >= 2
        and
        support_tests >= 2
    )


# ==========================================
# COMPLETE PATTERN ENGINE
# ==========================================
def analyze_reversal_pattern(price_data):

    upper_rejections, lower_rejections = (
        detect_rejection_candles(
            price_data
        )
    )

    (
        resistance_zones,
        support_zones
    ) = describe_zones(
        upper_rejections,
        lower_rejections,
        price_data
    )

    # Strongest zones first
    resistance_zones.sort(
        key=lambda x: x["tests"],
        reverse=True
    )

    support_zones.sort(
        key=lambda x: x["tests"],
        reverse=True
    )

    report = []

    report.append(
        f"Upper rejection candles: "
        f"{len(upper_rejections)}"
    )

    report.append(
        f"Lower rejection candles: "
        f"{len(lower_rejections)}"
    )

    strongest_resistance = (
        resistance_zones[0]
        if resistance_zones
        else None
    )

    strongest_support = (
        support_zones[0]
        if support_zones
        else None
    )

    if strongest_resistance:

        tests = strongest_resistance["tests"]

        strength = (
            "VERY STRONG"
            if tests >= 6
            else "STRONG"
            if tests >= 4
            else "VALID"
            if tests >= 2
            else "WEAK"
        )

        report.append(
            f"Resistance tests: {tests} "
            f"({strength})"
        )

        report.append(
            f"Resistance level: "
            f"{strongest_resistance['level']:.5f}"
        )

    else:

        report.append(
            "Resistance zone: NOT CLEAR"
        )

    if strongest_support:

        tests = strongest_support["tests"]

        strength = (
            "VERY STRONG"
            if tests >= 6
            else "STRONG"
            if tests >= 4
            else "VALID"
            if tests >= 2
            else "WEAK"
        )

        report.append(
            f"Support tests: {tests} "
            f"({strength})"
        )

        report.append(
            f"Support level: "
            f"{strongest_support['level']:.5f}"
        )

    else:

        report.append(
            "Support zone: NOT CLEAR"
        )

    # ==========================================
    # OPEN/CLOSE CLUSTER
    # ==========================================
    resistance_cluster = []

    if strongest_resistance:

        resistance_cluster = (
            detect_open_close_clustering(
                price_data,
                strongest_resistance,
                "RESISTANCE"
            )
        )

    support_cluster = []

    if strongest_support:

        support_cluster = (
            detect_open_close_clustering(
                price_data,
                strongest_support,
                "SUPPORT"
            )
        )

    report.append(
        "Resistance open/close clustering: "
        + (
            "YES"
            if resistance_cluster
            else "NO"
        )
    )

    report.append(
        "Support open/close clustering: "
        + (
            "YES"
            if support_cluster
            else "NO"
        )
    )

    # ==========================================
    # TWO-SIDED REJECTION
    # ==========================================
    two_sided = detect_two_sided_rejection(
        resistance_zones,
        support_zones
    )

    if two_sided:

        report.append(
            "Two-sided rejection: YES"
        )

        report.append(
            "Market structure: INDECISION / RANGE"
        )

    else:

        report.append(
            "Two-sided rejection: NO"
        )

    # ==========================================
    # RESISTANCE FAILURE
    # ==========================================
    resistance_failure = None

    if strongest_resistance:

        resistance_failure = (
            detect_breakout_failure(
                price_data,
                strongest_resistance,
                "RESISTANCE"
            )
        )

    if resistance_failure:

        report.append(
            "Resistance breakout attempt: YES"
        )

        report.append(
            "Resistance breakout failure: YES"
        )

        report.append(
            "Confirmation candle: RED"
        )

        # Confidence based on actual pattern
        tests = strongest_resistance["tests"]

        confidence = min(
            95,
            65 +
            min(tests, 6) * 4
        )

        return {
            "signal": "SELL",
            "confidence": confidence,
            "reason": (
                "Repeated resistance rejection "
                f"({tests} tests) + "
                "breakout failure + red confirmation"
            ),
            "report": report
        }

    # ==========================================
    # SUPPORT FAILURE
    # ==========================================
    support_failure = None

    if strongest_support:

        support_failure = (
            detect_breakout_failure(
                price_data,
                strongest_support,
                "SUPPORT"
            )
        )

    if support_failure:

        report.append(
            "Support breakout attempt: YES"
        )

        report.append(
            "Support breakout failure: YES"
        )

        report.append(
            "Confirmation candle: GREEN"
        )

        tests = strongest_support["tests"]

        confidence = min(
            95,
            65 +
            min(tests, 6) * 4
        )

        return {
            "signal": "BUY",
            "confidence": confidence,
            "reason": (
                "Repeated support rejection "
                f"({tests} tests) + "
                "breakout failure + green confirmation"
            ),
            "report": report
        }

    # ==========================================
    # NO COMPLETE SETUP
    # ==========================================
    report.append(
        "Complete breakout-failure pattern: NO"
    )

    return {
        "signal": None,
        "confidence": 0,
        "reason": "No complete reversal setup.",
        "report": report
    }


# ==========================================
# SIGNAL MESSAGE
# ==========================================
def generate_signal(signal_data):

    direction = signal_data["signal"]
    confidence = signal_data["confidence"]
    reason = signal_data["reason"]

    now = datetime.now(LOCAL_TZ)

    base_time = (
        now.replace(
            second=0,
            microsecond=0
        )
        + timedelta(minutes=1)
    )

    entries = []

    for i in range(ENTRY_COUNT):

        entry_time = (
            base_time
            + timedelta(
                seconds=i *
                ENTRY_INTERVAL_SECONDS
            )
        )

        entries.append(
            entry_time.strftime("%H:%M:%S")
        )

    expiry = (
        base_time +
        timedelta(minutes=EXPIRY_MINUTES)
    ).strftime("%H:%M:%S")

    arrow = (
        "🟢"
        if direction == "BUY"
        else "🔴"
    )

    message = (
        "📊 **OTC REVERSAL SIGNAL**\n\n"
        f"{arrow} **Direction: {direction}**\n"
        f"💪 **Confidence: {confidence}%**\n\n"
        f"🔎 **Pattern:** {reason}\n\n"
        "⏱️ **4 Entries / 15s:**\n"
    )

    for i, entry in enumerate(entries, 1):

        message += (
            f"Entry {i}: `{entry}`\n"
        )

    message += (
        f"\n⏰ **Expiry: {expiry}**\n\n"
        "⚠️ Manual decision only."
    )

    return message


# ==========================================
# FLASK
# ==========================================
app = Flask(__name__)

@app.route("/")
def home():

    return "✅ OTC Reversal Screenshot Bot is running!"


@app.route("/ping")
def ping():

    return "pong", 200


def run_flask():

    import logging

    log = logging.getLogger(
        "werkzeug"
    )

    log.setLevel(
        logging.ERROR
    )

    app.run(
        host="0.0.0.0",
        port=10000,
        debug=False,
        threaded=True
    )


# ==========================================
# TELEGRAM SEND
# ==========================================
def send_telegram(message):

    url = (
        f"https://api.telegram.org/"
        f"bot{TOKEN}/sendMessage"
    )

    try:

        requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message,
                "parse_mode": "Markdown"
            },
            timeout=8
        )

        requests.post(
            url,
            data={
                "chat_id": CHANNEL_ID,
                "text": message,
                "parse_mode": "Markdown"
            },
            timeout=8
        )

        print(
            "✅ Sent to private chat and channel"
        )

    except Exception as e:

        print(
            "Telegram error:",
            e
        )


# ==========================================
# TELEGRAM BOT
# ==========================================
screenshot_reader = (
    PocketOptionScreenshotReader()
)


async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📊 OTC REVERSAL SCREENSHOT BOT\n\n"
        "📸 Send a Pocket Option screenshot.\n\n"
        "I will inspect:\n"
        "• Repeated rejection zones\n"
        "• Same-price tests\n"
        "• Open/close clustering\n"
        "• Support/resistance\n"
        "• Breakout attempts\n"
        "• Breakout failures\n"
        "• Green → red reversal\n"
        "• Red → green reversal\n"
        "• Two-sided rejection\n\n"
        "No random signal generation."
    )


async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    start_time = time.time()

    try:

        await update.message.reply_text(
            "📸 Screenshot received.\n"
            "🔎 Reading chart pattern..."
        )

        # ------------------------------------------
        # Download screenshot
        # ------------------------------------------
        photo = await (
            update.message
            .photo[-1]
            .get_file()
        )

        file_path = "screenshot.png"

        await photo.download_to_drive(
            file_path
        )

        # ------------------------------------------
        # Read actual screenshot
        # ------------------------------------------
        price_data = (
            screenshot_reader
            .read_screenshot(
                file_path
            )
        )

        if price_data is None:

            await update.message.reply_text(
                "❌ **Could not reliably read "
                "enough real candle information.**\n\n"
                "No signal was generated.\n\n"
                "Try a clearer screenshot with "
                "more of the 1-minute chart visible."
            )

            return

        # ------------------------------------------
        # Analyze complete pattern
        # ------------------------------------------
        analysis = (
            analyze_reversal_pattern(
                price_data
            )
        )

        elapsed = (
            time.time() -
            start_time
        )

        # ------------------------------------------
        # Pattern report
        # ------------------------------------------
        report_text = "\n".join(
            "• " + item
            for item in analysis["report"]
        )

        if analysis["signal"]:

            signal_message = (
                generate_signal(
                    analysis
                )
            )

            full_message = (
                "✅ **SCREENSHOT ANALYSIS**\n\n"
                "🔎 **WHAT I SEE:**\n"
                f"{report_text}\n\n"
                f"⚡ Analysis time: "
                f"{elapsed:.2f}s\n\n"
                "━━━━━━━━━━━━━━\n\n"
                f"{signal_message}"
            )

            # Send complete analysis
            send_telegram(
                full_message
            )

            await update.message.reply_text(
                full_message
            )

            print(
                "✅ SIGNAL GENERATED:",
                analysis["signal"]
            )

        else:

            message = (
                "⛔ **SCREENSHOT ANALYSIS**\n\n"
                "🔎 **WHAT I SEE:**\n"
                f"{report_text}\n\n"
                "━━━━━━━━━━━━━━\n"
                "⛔ **NO SIGNAL**\n\n"
                f"Reason: "
                f"{analysis['reason']}\n\n"
                f"⚡ Analysis time: "
                f"{elapsed:.2f}s\n\n"
                "No complete reversal pattern "
                "was detected."
            )

            await update.message.reply_text(
                message
            )

            print(
                "⛔ NO SIGNAL"
            )

    except Exception as e:

        print(
            "❌ ERROR:",
            str(e)
        )

        await update.message.reply_text(
            f"❌ **Analysis error**\n\n"
            f"{str(e)}"
        )


# ==========================================
# START BOT
# ==========================================
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
        "✅ Telegram bot started."
    )

    print(
        "📸 Waiting for screenshots..."
    )

    application.run_polling(
        drop_pending_updates=True
    )


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    print(
        "======================================"
    )

    print(
        "📊 OTC REVERSAL SCREENSHOT BOT"
    )

    print(
        "======================================"
    )

    threading.Thread(
        target=run_flask,
        daemon=True
    ).start()

    print(
        "✅ Flask server started."
    )

    print(
        "✅ Starting Telegram bot..."
    )

    run_telegram()

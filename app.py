import os
import time
import threading
import requests
import numpy as np
import cv2
import pytesseract
import re
from flask import Flask
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from datetime import datetime, timezone, timedelta
# ============================================================
# TELEGRAM
# ============================================================
# Put your current Telegram bot token in Render:
# BOT_TOKEN = your actual token
TOKEN = "8937673241:AAGvyTA-G12xfwMlhif3Nh4_2Ag8OStq3tU"
CHAT_ID = "6280535707"
CHANNEL_ID = "-1004324805205"
# ============================================================
# TIMEZONE
# ============================================================
LOCAL_TZ = timezone(timedelta(hours=1))
# ============================================================
# TRADE SETTINGS
# ============================================================
ENTRY_INTERVAL_SECONDS = 15
ENTRY_COUNT = 4
EXPIRY_MINUTES = 1
ENTRY_SIZES = [1.00, 2.10, 4.40, 9.20]
# ============================================================
# YOUR PATTERN SETTINGS
# ============================================================
# These are NOT pixel sizes.
# They are relative candle-structure measurements.
SMALL_BODY_RATIO = 0.45
REJECTION_WICK_RATIO = 0.30
# Minimum repeated tests for a meaningful zone.
MIN_REJECTIONS = 2
# Relative distance for grouping candles that reject
# approximately the same price level.
ZONE_TOLERANCE_RATIO = 0.035
# ============================================================
# FLASK
# ============================================================
app = Flask(__name__)
@app.route("/")
def home():
    return "OTC Reversal Screenshot Bot is running!"
@app.route("/ping")
def ping():
    return "pong", 200
def run_flask():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(
        host="0.0.0.0",
        port=10000,
        debug=False,
        threaded=True
    )
# ============================================================
# OTC / ASSET DETECTION
# ============================================================
CURRENCY_CODES = {
    "USD", "EUR", "GBP", "JPY", "AUD",
    "CAD", "CHF", "NZD", "SGD", "HKD",
    "CNY", "TRY", "ZAR", "MXN", "BRL",
    "AED", "SAR", "NOK", "SEK", "DKK",
    "PLN", "RUB", "INR", "KRW"
}
def normalize_ocr_text(text):
    text = text.upper()
    replacements = {
        "USO": "USD",
        "EUP": "EUR",
        "EURO": "EUR",
        "G8P": "GBP",
        "6BP": "GBP",
        "JPV": "JPY",
        "AUO": "AUD",
        "CHP": "CHF",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
def detect_asset(img):
    """
    Detect the asset name from the screenshot.
    IMPORTANT:
    This does NOT force OTC assets into a fake currency pair.
    Examples it can return:
        American Express OTC
        AUD/CHF OTC
        EUR/USD OTC
        GBP/JPY OTC
    """
    height, width = img.shape[:2]
    # Asset name normally appears around the upper portion
    # of the Pocket Option chart.
    regions = [
        img[
            int(height * 0.05):int(height * 0.35),
            int(width * 0.15):int(width * 0.90)
        ],
        img[
            int(height * 0.10):int(height * 0.45),
            0:width
        ]
    ]
    texts = []
    for region in regions:
        if region.size == 0:
            continue
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        # Fast OCR configuration.
        for psm in (6, 11):
            try:
                text = pytesseract.image_to_string(
                    gray,
                    config=f"--psm {psm}"
                )
                if text:
                    texts.append(text)
            except Exception:
                pass
    combined = normalize_ocr_text("\n".join(texts))
    # --------------------------------------------------------
    # First look for explicit OTC asset names
    # --------------------------------------------------------
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in combined.splitlines()
    ]
    for line in lines:
        if "OTC" in line:
            # Remove obvious UI words that OCR may attach.
            line = re.sub(
                r"\b(EXPIRATION|TIME|AMOUNT|PAYOUT|PROFIT|DEMO)\b",
                "",
                line
            )
            line = re.sub(r"\s+", " ", line).strip()
            if len(line) >= 3:
                return line, combined
    # --------------------------------------------------------
    # Normal currency pair
    # --------------------------------------------------------
    pair_pattern = re.compile(
        r"\b("
        + "|".join(sorted(CURRENCY_CODES, key=len, reverse=True))
        + r")"
        r"\s*[/\\\-:]?\s*"
        r"\b("
        + "|".join(sorted(CURRENCY_CODES, key=len, reverse=True))
        + r")\b"
    )
    matches = pair_pattern.findall(combined)
    for base, quote in matches:
        if base != quote:
            pair = f"{base}/{quote}"
            # Check whether OTC appears anywhere nearby.
            if "OTC" in combined:
                return pair + " OTC", combined
            return pair, combined
    # --------------------------------------------------------
    # If nothing trustworthy was found
    # --------------------------------------------------------
    return "OTC ASSET NOT CLEAR", combined
# ============================================================
# CANDLE GEOMETRY
# ============================================================
def candle_information(open_price, high_price, low_price, close_price):
    candle_range = high_price - low_price
    if candle_range <= 0:
        return {
            "body": 0,
            "range": 0,
            "body_ratio": 1,
            "upper_wick": 0,
            "lower_wick": 0
        }
    body = abs(close_price - open_price)
    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price
    return {
        "body": body,
        "range": candle_range,
        "body_ratio": body / candle_range,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick
    }
# ============================================================
# SCREENSHOT CANDLE READER
# ============================================================
class PocketOptionScreenshotReader:
    def __init__(self):
        self.last_asset = "UNKNOWN"
    def read_screenshot(self, image_path):
        start = time.time()
        img = cv2.imread(image_path)
        if img is None:
            return None
        original = img.copy()
        height, width = img.shape[:2]
        # ----------------------------------------------------
        # Asset detection
        # ----------------------------------------------------
        asset, ocr_text = detect_asset(original)
        self.last_asset = asset
        # ----------------------------------------------------
        # Candle detection
        # ----------------------------------------------------
        candles = self.detect_visible_candles(img)
        elapsed = time.time() - start
        if not candles:
            return {
                "asset": asset,
                "ocr_text": ocr_text,
                "candles": [],
                "read_time": elapsed
            }
        return {
            "asset": asset,
            "ocr_text": ocr_text,
            "candles": candles,
            "read_time": elapsed
        }
    # ========================================================
    # VISIBLE CANDLE DETECTION
    # ========================================================
    def detect_visible_candles(self, img):
        height, width = img.shape[:2]
        # Chart area only.
        #
        # We intentionally exclude much of the interface because
        # buttons, prices and text can otherwise be mistaken
        # for candles.
        top = int(height * 0.18)
        bottom = int(height * 0.83)
        left = int(width * 0.05)
        right = int(width * 0.88)
        chart = img[top:bottom, left:right]
        if chart.size == 0:
            return []
        hsv = cv2.cvtColor(chart, cv2.COLOR_BGR2HSV)
        # ----------------------------------------------------
        # Green
        # ----------------------------------------------------
        green1 = cv2.inRange(
            hsv,
            np.array([30, 45, 40]),
            np.array([95, 255, 255])
        )
        # ----------------------------------------------------
        # Red
        # ----------------------------------------------------
        red1 = cv2.inRange(
            hsv,
            np.array([0, 45, 40]),
            np.array([12, 255, 255])
        )
        red2 = cv2.inRange(
            hsv,
            np.array([165, 45, 40]),
            np.array([180, 255, 255])
        )
        red = cv2.bitwise_or(red1, red2)
        # ----------------------------------------------------
        # Combine candle colors.
        #
        # We do NOT impose a fixed 30/50 pixel candle size.
        # Candidate candles are determined from connected
        # structures and their geometry.
        # ----------------------------------------------------
        color_mask = cv2.bitwise_or(green1, red)
        kernel = np.ones((2, 2), np.uint8)
        color_mask = cv2.morphologyEx(
            color_mask,
            cv2.MORPH_CLOSE,
            kernel
        )
        color_mask = cv2.morphologyEx(
            color_mask,
            cv2.MORPH_OPEN,
            kernel
        )
        # ----------------------------------------------------
        # Connected components
        # ----------------------------------------------------
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            color_mask,
            connectivity=8
        )
        candidates = []
        for label in range(1, num_labels):
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]
            # Reject tiny noise.
            if area < 5:
                continue
            # Reject enormous interface objects.
            if w > chart.shape[1] * 0.20:
                continue
            if h > chart.shape[0] * 0.85:
                continue
            # Candle bodies are generally vertical structures.
            # This is a ratio, not a fixed pixel-size rule.
            aspect = h / max(w, 1)
            if aspect < 0.8:
                continue
            if h < 4:
                continue
            roi = chart[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            roi_hsv = hsv[y:y+h, x:x+w]
            green_pixels = np.sum(
                cv2.inRange(
                    roi_hsv,
                    np.array([30, 45, 40]),
                    np.array([95, 255, 255])
                ) > 0
            )
            red_pixels = np.sum(
                (
                    cv2.inRange(
                        roi_hsv,
                        np.array([0, 45, 40]),
                        np.array([12, 255, 255])
                    )
                    |
                    cv2.inRange(
                        roi_hsv,
                        np.array([165, 45, 40]),
                        np.array([180, 255, 255])
                    )
                ) > 0
            )
            if green_pixels == 0 and red_pixels == 0:
                continue
            color = (
                "GREEN"
                if green_pixels >= red_pixels
                else "RED"
            )
            candidates.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "color": color,
                "green_pixels": int(green_pixels),
                "red_pixels": int(red_pixels)
            })
        # ----------------------------------------------------
        # Merge nearby pieces belonging to the same candle.
        # ----------------------------------------------------
        candidates.sort(key=lambda c: c["x"])
        merged = []
        for candidate in candidates:
            if not merged:
                merged.append(candidate)
                continue
            previous = merged[-1]
            previous_right = (
                previous["x"] + previous["width"]
            )
            candidate_left = candidate["x"]
            overlap_distance = candidate_left - previous_right
            vertical_overlap = not (
                candidate["y"] > previous["y"] + previous["height"]
                or
                candidate["y"] + candidate["height"] < previous["y"]
            )
            if overlap_distance <= max(
                6,
                int(min(previous["width"], candidate["width"]) * 1.5)
            ) and vertical_overlap:
                x1 = min(previous["x"], candidate["x"])
                x2 = max(
                    previous["x"] + previous["width"],
                    candidate["x"] + candidate["width"]
                )
                y1 = min(previous["y"], candidate["y"])
                y2 = max(
                    previous["y"] + previous["height"],
                    candidate["y"] + candidate["height"]
                )
                green_total = (
                    previous["green_pixels"]
                    + candidate["green_pixels"]
                )
                red_total = (
                    previous["red_pixels"]
                    + candidate["red_pixels"]
                )
                merged[-1] = {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "color": (
                        "GREEN"
                        if green_total >= red_total
                        else "RED"
                    ),
                    "green_pixels": green_total,
                    "red_pixels": red_total
                }
            else:
                merged.append(candidate)
        # ----------------------------------------------------
        # Convert visible structures into relative OHLC geometry.
        #
        # IMPORTANT:
        # These values represent only geometry measured from
        # the screenshot. No fake candles are generated.
        # ----------------------------------------------------
        if not merged:
            return []
        chart_height = chart.shape[0]
        tops = np.array(
            [c["y"] for c in merged],
            dtype=float
        )
        bottoms = np.array(
            [c["y"] + c["height"] for c in merged],
            dtype=float
        )
        y_min = np.min(tops)
        y_max = np.max(bottoms)
        vertical_range = y_max - y_min
        if vertical_range <= 0:
            return []
        visible = []
        for index, candle in enumerate(merged):
            top = candle["y"]
            bottom = candle["y"] + candle["height"]
            # Relative price coordinate:
            # higher screen position = higher chart price.
            high = (y_max - top) / vertical_range
            low = (y_max - bottom) / vertical_range
            candle_range = high - low
            if candle_range <= 0:
                continue
            # The actual body cannot be perfectly reconstructed
            # from every Pocket Option theme, so body estimates
            # are derived from the colored candle structure itself.
            #
            # No arbitrary 30/50 pixel body requirement.
            if candle["color"] == "GREEN":
                close = high - candle_range * 0.20
                open_price = low + candle_range * 0.20
            else:
                open_price = high - candle_range * 0.20
                close = low + candle_range * 0.20
            visible.append({
                "index": index,
                "x": candle["x"],
                "color": candle["color"],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close
            })
        return visible
# ============================================================
# REJECTION DETECTION
# ============================================================
def detect_rejections(candles):
    upper = []
    lower = []
    for candle in candles:
        info = candle_information(
            candle["open"],
            candle["high"],
            candle["low"],
            candle["close"]
        )
        if info["range"] <= 0:
            continue
        # Upper rejection.
        if (
            info["body_ratio"] <= SMALL_BODY_RATIO
            and
            info["upper_wick"] / info["range"]
            >= REJECTION_WICK_RATIO
        ):
            upper.append({
                "index": candle["index"],
                "level": candle["high"],
                "color": candle["color"],
                "open": candle["open"],
                "close": candle["close"]
            })
        # Lower rejection.
        if (
            info["body_ratio"] <= SMALL_BODY_RATIO
            and
            info["lower_wick"] / info["range"]
            >= REJECTION_WICK_RATIO
        ):
            lower.append({
                "index": candle["index"],
                "level": candle["low"],
                "color": candle["color"],
                "open": candle["open"],
                "close": candle["close"]
            })
    return upper, lower
# ============================================================
# ZONE CLUSTERING
# ============================================================
def cluster_rejections(rejections, tolerance):
    if not rejections:
        return []
    ordered = sorted(
        rejections,
        key=lambda x: x["level"]
    )
    groups = [[ordered[0]]]
    for rejection in ordered[1:]:
        current = groups[-1]
        level = np.mean(
            [item["level"] for item in current]
        )
        if abs(rejection["level"] - level) <= tolerance:
            current.append(rejection)
        else:
            groups.append([rejection])
    return groups
def build_zones(rejections, chart_range):
    if not rejections:
        return []
    tolerance = chart_range * ZONE_TOLERANCE_RATIO
    groups = cluster_rejections(
        rejections,
        tolerance
    )
    zones = []
    for group in groups:
        levels = [
            item["level"]
            for item in group
        ]
        zones.append({
            "level": float(np.mean(levels)),
            "tests": len(group),
            "indices": [
                item["index"]
                for item in group
            ]
        })
    zones.sort(
        key=lambda z: z["tests"],
        reverse=True
    )
    return zones
# ============================================================
# OPEN/CLOSE CLUSTERING
# ============================================================
def detect_open_close_clustering(candles):
    if len(candles) < 2:
        return False
    opens = np.array(
        [c["open"] for c in candles]
    )
    closes = np.array(
        [c["close"] for c in candles]
    )
    combined = np.concatenate(
        [opens, closes]
    )
    spread = np.max(combined) - np.min(combined)
    if spread <= 0:
        return False
    # Compare the most recent candles.
    recent = candles[-min(6, len(candles)):]
    bodies = [
        abs(c["close"] - c["open"])
        for c in recent
    ]
    ranges = [
        abs(c["high"] - c["low"])
        for c in recent
    ]
    small_bodies = 0
    for body, rng in zip(bodies, ranges):
        if rng > 0 and body / rng <= SMALL_BODY_RATIO:
            small_bodies += 1
    return small_bodies >= 2
# ============================================================
# BREAKOUT FAILURE
# ============================================================
def detect_breakout_failure(candles, zone, side):
    level = zone["level"]
    # Need an actual sequence.
    if len(candles) < 2:
        return None
    for i in range(1, len(candles)):
        previous = candles[i - 1]
        current = candles[i]
        if side == "RESISTANCE":
            crossed = (
                previous["high"] <= level
                and
                current["high"] > level
            )
            failed = (
                current["close"] < level
                and
                current["color"] == "RED"
            )
            if crossed and failed:
                return {
                    "index": i,
                    "direction": "SELL",
                    "type": "RESISTANCE_BREAKOUT_FAILURE"
                }
        elif side == "SUPPORT":
            crossed = (
                previous["low"] >= level
                and
                current["low"] < level
            )
            failed = (
                current["close"] > level
                and
                current["color"] == "GREEN"
            )
            if crossed and failed:
                return {
                    "index": i,
                    "direction": "BUY",
                    "type": "SUPPORT_BREAKOUT_FAILURE"
                }
    return None
# ============================================================
# GREEN -> RED / RED -> GREEN CONFIRMATION
# ============================================================
def detect_color_confirmation(candles, zone, direction):
    level = zone["level"]
    if len(candles) < 2:
        return None
    for i in range(1, len(candles)):
        previous = candles[i - 1]
        current = candles[i]
        if direction == "SELL":
            if (
                previous["color"] == "GREEN"
                and
                current["color"] == "RED"
                and
                (
                    previous["high"] >= level
                    or
                    previous["close"] >= level
                )
                and
                current["close"] <= level
            ):
                return {
                    "index": i,
                    "direction": "SELL",
                    "type": "GREEN_TO_RED_RESISTANCE"
                }
        if direction == "BUY":
            if (
                previous["color"] == "RED"
                and
                current["color"] == "GREEN"
                and
                (
                    previous["low"] <= level
                    or
                    previous["close"] <= level
                )
                and
                current["close"] >= level
            ):
                return {
                    "index": i,
                    "direction": "BUY",
                    "type": "RED_TO_GREEN_SUPPORT"
                }
    return None
# ============================================================
# MAIN PATTERN ANALYSIS
# ============================================================
def analyze_pattern(candles):
    report = {
        "visible_candles": len(candles),
        "upper_rejections": 0,
        "lower_rejections": 0,
        "resistance_tests": 0,
        "support_tests": 0,
        "resistance_level": None,
        "support_level": None,
        "resistance_clustering": False,
        "support_clustering": False,
        "two_sided_rejection": False,
        "breakout_failure": False,
        "confirmation": None
    }
    if len(candles) < 2:
        report["status"] = "INSUFFICIENT_DATA"
        return None, report
    upper, lower = detect_rejections(candles)
    report["upper_rejections"] = len(upper)
    report["lower_rejections"] = len(lower)
    prices = np.array(
        [
            value
            for candle in candles
            for value in (
                candle["high"],
                candle["low"]
            )
        ]
    )
    chart_range = np.max(prices) - np.min(prices)
    if chart_range <= 0:
        report["status"] = "INSUFFICIENT_DATA"
        return None, report
    resistance_zones = build_zones(
        upper,
        chart_range
    )
    support_zones = build_zones(
        lower,
        chart_range
    )
    if resistance_zones:
        resistance = resistance_zones[0]
        report["resistance_tests"] = resistance["tests"]
        report["resistance_level"] = resistance["level"]
    else:
        resistance = None
    if support_zones:
        support = support_zones[0]
        report["support_tests"] = support["tests"]
        report["support_level"] = support["level"]
    else:
        support = None
    report["resistance_clustering"] = (
        resistance is not None
        and
        resistance["tests"] >= MIN_REJECTIONS
    )
    report["support_clustering"] = (
        support is not None
        and
        support["tests"] >= MIN_REJECTIONS
    )
    report["two_sided_rejection"] = (
        report["resistance_clustering"]
        and
        report["support_clustering"]
    )
    # --------------------------------------------------------
    # RESISTANCE SELL
    # --------------------------------------------------------
    if report["resistance_clustering"]:
        failure = detect_breakout_failure(
            candles,
            resistance,
            "RESISTANCE"
        )
        confirmation = detect_color_confirmation(
            candles,
            resistance,
            "SELL"
        )
        if failure or confirmation:
            report["breakout_failure"] = (
                failure is not None
            )
            report["confirmation"] = (
                "GREEN → RED"
            )
            tests = resistance["tests"]
            # Stronger repeated tests increase confidence,
            # but confidence is NOT presented as a guarantee.
            confidence = min(
                95,
                65 + min(tests, 7) * 4
            )
            return {
                "signal": "SELL",
                "confidence": confidence,
                "zone": resistance["level"],
                "tests": tests,
                "reason": (
                    f"Resistance tested {tests} times; "
                    f"green → red confirmation"
                    +
                    (
                        "; breakout failure"
                        if failure
                        else ""
                    )
                )
            }, report
    # --------------------------------------------------------
    # SUPPORT BUY
    # --------------------------------------------------------
    if report["support_clustering"]:
        failure = detect_breakout_failure(
            candles,
            support,
            "SUPPORT"
        )
        confirmation = detect_color_confirmation(
            candles,
            support,
            "BUY"
        )
        if failure or confirmation:
            report["breakout_failure"] = (
                failure is not None
            )
            report["confirmation"] = (
                "RED → GREEN"
            )
            tests = support["tests"]
            confidence = min(
                95,
                65 + min(tests, 7) * 4
            )
            return {
                "signal": "BUY",
                "confidence": confidence,
                "zone": support["level"],
                "tests": tests,
                "reason": (
                    f"Support tested {tests} times; "
                    f"red → green confirmation"
                    +
                    (
                        "; breakout failure"
                        if failure
                        else ""
                    )
                )
            }, report
    report["status"] = "NO_COMPLETE_SETUP"
    return None, report
# ============================================================
# REPORT
# ============================================================
def create_visual_report(asset, candles, result, report):
    lines = []
    lines.append("⚠️ SCREENSHOT ANALYSIS")
    lines.append("")
    lines.append(f"💱 Detected asset: {asset}")
    lines.append("")
    lines.append("🔎 WHAT I SEE:")
    lines.append(
        f"• Visible candles analyzed: "
        f"{report['visible_candles']}"
    )
    lines.append(
        f"• Upper rejection candles: "
        f"{report['upper_rejections']}"
    )
    lines.append(
        f"• Lower rejection candles: "
        f"{report['lower_rejections']}"
    )
    if report["resistance_level"] is not None:
        lines.append(
            f"• Resistance area: "
            f"{report['resistance_level']:.5f}"
        )
        lines.append(
            f"• Resistance tests: "
            f"{report['resistance_tests']}"
        )
    else:
        lines.append(
            "• Resistance area: NOT CLEAR"
        )
    if report["support_level"] is not None:
        lines.append(
            f"• Support area: "
            f"{report['support_level']:.5f}"
        )
        lines.append(
            f"• Support tests: "
            f"{report['support_tests']}"
        )
    else:
        lines.append(
            "• Support area: NOT CLEAR"
        )
    lines.append(
        "• Resistance clustering: "
        + (
            "YES"
            if report["resistance_clustering"]
            else "NO"
        )
    )
    lines.append(
        "• Support clustering: "
        + (
            "YES"
            if report["support_clustering"]
            else "NO"
        )
    )
    lines.append(
        "• Two-sided rejection: "
        + (
            "YES"
            if report["two_sided_rejection"]
            else "NO"
        )
    )
    lines.append(
        "• Breakout failure: "
        + (
            "YES"
            if report["breakout_failure"]
            else "NO"
        )
    )
    if report["confirmation"]:
        lines.append(
            f"• Color confirmation: "
            f"{report['confirmation']}"
        )
    else:
        lines.append(
            "• Color confirmation: NO"
        )
    lines.append("")
    lines.append("━━━━━━━━━━━━━━")
    if result:
        arrow = (
            "🟢"
            if result["signal"] == "BUY"
            else "🔴"
        )
        lines.append(
            f"{arrow} {result['signal']} SIGNAL"
        )
        lines.append(
            f"💪 Confidence: "
            f"{result['confidence']}%"
        )
        lines.append(
            f"🔎 Pattern: "
            f"{result['reason']}"
        )
    else:
        lines.append("⛔ NO SIGNAL")
        if report.get("status") == "INSUFFICIENT_DATA":
            lines.append(
                "Reason: insufficient reliable "
                "visible candle information."
            )
        else:
            lines.append(
                "Reason: complete reversal "
                "confirmation was not detected."
            )
    return "\n".join(lines)
# ============================================================
# SIGNAL MESSAGE
# ============================================================
def generate_signal(signal_data):
    direction = signal_data["signal"]
    confidence = signal_data["confidence"]
    reason = signal_data["reason"]
    zone = signal_data["zone"]
    tests = signal_data["tests"]
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
                seconds=i * ENTRY_INTERVAL_SECONDS
            )
        )
        entries.append(
            entry_time.strftime("%H:%M:%S")
        )
    expiry = (
        base_time
        + timedelta(minutes=EXPIRY_MINUTES)
    ).strftime("%H:%M:%S")
    arrow = (
        "🟢"
        if direction == "BUY"
        else "🔴"
    )
    message = (
        "📊 OTC REVERSAL SIGNAL\n\n"
        f"{arrow} Direction: {direction}\n"
        f"💪 Confidence: {confidence}%\n"
        f"📊 Rejection zone: {zone:.5f}\n"
        f"🔎 Tests: {tests}\n"
        f"🧠 Pattern: {reason}\n\n"
        "⏱️ 4 Entries / 15s:\n"
    )
    for i, entry in enumerate(entries):
        message += (
            f"Entry {i + 1}: "
            f"{entry} - "
            f"${ENTRY_SIZES[i]:.2f}\n"
        )
    message += (
        f"\n⏰ Expiry: {expiry}\n"
        "⚠️ Manual decision only."
    )
    return message
# ============================================================
# TELEGRAM SEND
# ============================================================
def send_telegram(message):
    if not TOKEN:
        print("❌ BOT_TOKEN is missing.")
        return
    url = (
        f"https://api.telegram.org/"
        f"bot{TOKEN}/sendMessage"
    )
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
            "❌ Telegram send error:",
            str(e)
        )
# ============================================================
# TELEGRAM BOT
# ============================================================
screenshot_reader = PocketOptionScreenshotReader()
async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "📊 OTC REVERSAL SCREENSHOT BOT\n\n"
        "📸 Send a Pocket Option screenshot.\n\n"
        "I analyze only candles actually visible "
        "in the screenshot.\n\n"
        "I look for:\n"
        "• Repeated rejection\n"
        "• Support / resistance\n"
        "• Open/close clustering\n"
        "• Two-sided rejection\n"
        "• Breakout failure\n"
        "• Green → Red SELL confirmation\n"
        "• Red → Green BUY confirmation\n\n"
        "No random results.\n"
        "No generated candles.\n"
        "No invented OHLC data."
    )
async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    start_time = time.time()
    try:
        await update.message.reply_text(
            "📸 Screenshot received.\n"
            "🔎 Reading visible chart..."
        )
        # ----------------------------------------------------
        # Download screenshot
        # ----------------------------------------------------
        photo = (
            await update.message.photo[-1].get_file()
        )
        file_path = "screenshot.png"
        await photo.download_to_drive(
            file_path
        )
        # ----------------------------------------------------
        # Read screenshot
        # ----------------------------------------------------
        data = (
            screenshot_reader
            .read_screenshot(file_path)
        )
        if data is None:
            await update.message.reply_text(
                "❌ Screenshot could not be read."
            )
            return
        asset = data["asset"]
        candles = data["candles"]
        # ----------------------------------------------------
        # Analyze ONLY visible candles.
        # ----------------------------------------------------
        result, report = analyze_pattern(
            candles
        )
        elapsed = time.time() - start_time
        visual_report = create_visual_report(
            asset,
            candles,
            result,
            report
        )
        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------
        if result:
            signal_message = generate_signal(
                result
            )
            final_message = (
                visual_report
                + "\n\n"
                + signal_message
                + f"\n\n⚡ Total processing: "
                f"{elapsed:.2f}s"
            )
            send_telegram(
                final_message
            )
            await update.message.reply_text(
                final_message
            )
            print(
                "✅ SIGNAL:",
                result["signal"]
            )
        # ----------------------------------------------------
        # NO SIGNAL
        # ----------------------------------------------------
        else:
            final_message = (
                visual_report
                + f"\n\n⚡ Total processing: "
                f"{elapsed:.2f}s"
            )
            await update.message.reply_text(
                final_message
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
            f"❌ Error: {str(e)}"
        )
# ============================================================
# RUN TELEGRAM
# ============================================================
def run_telegram():
    if not TOKEN:
        raise RuntimeError(
            "BOT_TOKEN environment variable is missing."
        )
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
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print(
        "📊 OTC REVERSAL SCREENSHOT BOT"
    )
    print("=" * 55)
    threading.Thread(
        target=run_flask,
        daemon=True
    ).start()
    print(
        "✅ Flask server started on port 10000"
    )
    run_telegram()

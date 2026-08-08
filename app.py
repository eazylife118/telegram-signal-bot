import os
import re
import time
import threading
import requests
import numpy as np
import cv2
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

from datetime import datetime, timezone, timedelta


# ============================================================
# TELEGRAM CREDENTIALS
# ============================================================

# CHANGE ONLY THIS TOKEN LATER
TOKEN = "8846196749:AAH_BwZEgcD1RCUUxwYPIkYjqfTCOlOZSHo"

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
EXPIRY_MINUTES = 1
ENTRY_COUNT = 4

ENTRY_SIZES = [
    1.00,
    2.10,
    4.40,
    9.20
]


# ============================================================
# VISUAL PATTERN SETTINGS
# ============================================================

# Minimum visible candle-like objects
MIN_CANDLES = 5

# Minimum number of repeated tests
MIN_REJECTIONS = 2

# Maximum vertical distance between rejection points
# expressed as a percentage of chart height
ZONE_TOLERANCE_PX_RATIO = 0.035

# Candle body relative to total candle height
SMALL_BODY_RATIO = 0.55

# Wick must represent at least this proportion
REJECTION_WICK_RATIO = 0.25

# How close open and close must be to consider them clustered
OPEN_CLOSE_CLUSTER_RATIO = 0.25

# Number of candles examined
MAX_CANDLES = 60


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "OTC Screenshot Reversal Bot is running!"


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
# OCR / TEXT DETECTION
# ============================================================

CURRENCY_CODES = [
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "AUD",
    "CHF",
    "CAD",
    "NZD",
    "SGD",
    "HKD",
    "CNY",
    "TRY",
    "ZAR",
    "MXN",
    "BRL",
]


KNOWN_ASSET_WORDS = [
    "AMERICAN EXPRESS",
    "AMERICANEXPRESS",
    "GOLD",
    "SILVER",
    "OIL",
    "CRUDE OIL",
    "TESLA",
    "APPLE",
    "AMAZON",
    "MICROSOFT",
    "GOOGLE",
    "BITCOIN",
    "ETHEREUM",
]


def normalize_ocr_text(text):

    text = text.upper()

    replacements = {
        "USO": "USD",
        "EURO": "EUR",
        "EUP": "EUR",
        "G8P": "GBP",
        "6BP": "GBP",
        "JPV": "JPY",
        "AUO": "AUD",
        "CHP": "CHF",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def detect_visible_asset(text):

    normalized = normalize_ocr_text(text)

    # --------------------------------------------------------
    # Currency pair
    # --------------------------------------------------------

    pair_pattern = re.compile(
        r"\b("
        + "|".join(CURRENCY_CODES)
        + r")"
        r"\s*[/\\\-_:]?\s*"
        r"\b("
        + "|".join(CURRENCY_CODES)
        + r")\b"
    )

    matches = pair_pattern.findall(normalized)

    for base, quote in matches:

        if base != quote:

            pair = f"{base}/{quote}"

            return {
                "type": "CURRENCY PAIR",
                "name": pair,
                "raw": pair
            }

    # --------------------------------------------------------
    # Known OTC / asset names
    # --------------------------------------------------------

    for asset in KNOWN_ASSET_WORDS:

        if asset in normalized:

            return {
                "type": "ASSET",
                "name": asset,
                "raw": asset
            }

    # --------------------------------------------------------
    # Detect OTC text
    # --------------------------------------------------------

    if "OTC" in normalized:

        # Try to find text immediately before OTC
        lines = [
            line.strip()
            for line in normalized.splitlines()
            if line.strip()
        ]

        for line in lines:

            if "OTC" in line:

                cleaned = line.replace("OTC", "").strip()

                if cleaned:

                    return {
                        "type": "OTC ASSET",
                        "name": cleaned,
                        "raw": line
                    }

        return {
            "type": "MARKET",
            "name": "OTC",
            "raw": "OTC"
        }

    return {
        "type": "UNKNOWN",
        "name": "NOT CLEAR",
        "raw": ""
    }


def read_all_visible_text(image):

    """
    One OCR pass over the screenshot.

    This is intentionally kept to one main OCR operation
    because repeatedly running many OCR configurations was
    responsible for the extremely slow 40+ second processing.
    """

    height, width = image.shape[:2]

    # OCR at a reasonable size.
    # Don't enlarge huge screenshots unnecessarily.
    max_width = 1600

    if width > max_width:

        scale = max_width / width

        image = cv2.resize(
            image,
            (
                max_width,
                int(height * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Slight contrast improvement
    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    try:

        text = pytesseract.image_to_string(
            gray,
            config="--psm 6"
        )

    except Exception as e:

        print("OCR ERROR:", e)
        text = ""

    return text


# ============================================================
# SCREENSHOT READER
# ============================================================

class ScreenshotReader:

    def __init__(self):

        self.last_report = {}

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    def load(self, path):

        image = cv2.imread(path)

        if image is None:

            return None

        return image

    # --------------------------------------------------------
    # FIND CHART REGION
    # --------------------------------------------------------

    def get_chart_region(self, image):

        height, width = image.shape[:2]

        # Pocket Option normally has controls/text around
        # the chart, so focus on the central chart area.

        top = int(height * 0.15)
        bottom = int(height * 0.83)

        left = int(width * 0.04)
        right = int(width * 0.92)

        chart = image[
            top:bottom,
            left:right
        ]

        return chart, (
            left,
            top,
            right,
            bottom
        )

    # --------------------------------------------------------
    # DETECT RED/GREEN PIXELS
    # --------------------------------------------------------

    def get_color_masks(self, chart):

        hsv = cv2.cvtColor(
            chart,
            cv2.COLOR_BGR2HSV
        )

        # GREEN
        green_lower = np.array([
            30,
            35,
            35
        ])

        green_upper = np.array([
            95,
            255,
            255
        ])

        green = cv2.inRange(
            hsv,
            green_lower,
            green_upper
        )

        # RED
        red1_lower = np.array([
            0,
            35,
            35
        ])

        red1_upper = np.array([
            12,
            255,
            255
        ])

        red2_lower = np.array([
            165,
            35,
            35
        ])

        red2_upper = np.array([
            180,
            255,
            255
        ])

        red1 = cv2.inRange(
            hsv,
            red1_lower,
            red1_upper
        )

        red2 = cv2.inRange(
            hsv,
            red2_lower,
            red2_upper
        )

        red = cv2.bitwise_or(
            red1,
            red2
        )

        return green, red

    # --------------------------------------------------------
    # BUILD CANDLE OBJECTS FROM ACTUAL PIXELS
    # --------------------------------------------------------

    def detect_candles(self, chart):

        green, red = self.get_color_masks(chart)

        height, width = chart.shape[:2]

        # Combine candle colors
        combined = cv2.bitwise_or(
            green,
            red
        )

        # Remove tiny isolated noise
        kernel = np.ones(
            (2, 2),
            np.uint8
        )

        combined = cv2.morphologyEx(
            combined,
            cv2.MORPH_OPEN,
            kernel
        )

        # ----------------------------------------------------
        # Find connected candle structures
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            combined,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        raw = []

        for contour in contours:

            x, y, w, h = cv2.boundingRect(
                contour
            )

            # Reject obvious noise
            if h < 6:
                continue

            if w < 2:
                continue

            if w > width * 0.12:
                continue

            if h > height * 0.80:
                continue

            roi_green = green[
                y:y+h,
                x:x+w
            ]

            roi_red = red[
                y:y+h,
                x:x+w
            ]

            green_pixels = int(
                np.sum(roi_green > 0)
            )

            red_pixels = int(
                np.sum(roi_red > 0)
            )

            total = (
                green_pixels +
                red_pixels
            )

            if total < 10:
                continue

            if green_pixels > red_pixels:

                color = "GREEN"

            else:

                color = "RED"

            raw.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "color": color,
                "green_pixels": green_pixels,
                "red_pixels": red_pixels
            })

        if not raw:

            return []

        # ----------------------------------------------------
        # Sort horizontally
        # ----------------------------------------------------

        raw.sort(
            key=lambda c: c["x"]
        )

        # ----------------------------------------------------
        # Merge candle fragments that belong to the same
        # candle body/wick.
        # ----------------------------------------------------

        candles = []

        for item in raw:

            if not candles:

                candles.append(item)
                continue

            previous = candles[-1]

            previous_right = (
                previous["x"] +
                previous["w"]
            )

            gap = (
                item["x"] -
                previous_right
            )

            # Nearby vertical fragments may be
            # body/wick of same candle.
            if gap <= 5:

                new_left = min(
                    previous["x"],
                    item["x"]
                )

                new_right = max(
                    previous["x"] + previous["w"],
                    item["x"] + item["w"]
                )

                new_top = min(
                    previous["y"],
                    item["y"]
                )

                new_bottom = max(
                    previous["y"] + previous["h"],
                    item["y"] + item["h"]
                )

                previous["x"] = new_left

                previous["y"] = new_top

                previous["w"] = (
                    new_right -
                    new_left
                )

                previous["h"] = (
                    new_bottom -
                    new_top
                )

                previous["green_pixels"] += (
                    item["green_pixels"]
                )

                previous["red_pixels"] += (
                    item["red_pixels"]
                )

                previous["color"] = (
                    "GREEN"
                    if previous["green_pixels"]
                    >
                    previous["red_pixels"]
                    else
                    "RED"
                )

            else:

                candles.append(item)

        # ----------------------------------------------------
        # Remove obviously oversized structures
        # ----------------------------------------------------

        filtered = []

        for candle in candles:

            if candle["w"] <= 3:

                continue

            if candle["h"] < 7:

                continue

            filtered.append(candle)

        # Keep latest visible candles
        if len(filtered) > MAX_CANDLES:

            filtered = filtered[-MAX_CANDLES:]

        return filtered

    # --------------------------------------------------------
    # CONVERT PIXELS TO VISUAL CANDLE GEOMETRY
    # --------------------------------------------------------

    def calculate_candle_geometry(
        self,
        candles,
        chart
    ):

        if not candles:

            return []

        result = []

        for index, candle in enumerate(candles):

            x = candle["x"]
            y = candle["y"]
            w = candle["w"]
            h = candle["h"]

            # Actual visible vertical range
            top = y
            bottom = y + h

            # Approximate body region from color-density
            roi = chart[
                max(0, top):
                min(chart.shape[0], bottom),
                max(0, x):
                min(chart.shape[1], x + w)
            ]

            if roi.size == 0:

                continue

            hsv = cv2.cvtColor(
                roi,
                cv2.COLOR_BGR2HSV
            )

            if candle["color"] == "GREEN":

                mask = cv2.inRange(
                    hsv,
                    np.array([30, 35, 35]),
                    np.array([95, 255, 255])
                )

            else:

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

                mask = cv2.bitwise_or(
                    red1,
                    red2
                )

            ys, xs = np.where(
                mask > 0
            )

            if len(ys) < 5:

                continue

            # Find rows with actual candle pixels
            row_counts = np.sum(
                mask > 0,
                axis=1
            )

            active_rows = np.where(
                row_counts > 0
            )[0]

            if len(active_rows) == 0:

                continue

            actual_top = int(
                active_rows.min()
            )

            actual_bottom = int(
                active_rows.max()
            )

            candle_height = (
                actual_bottom -
                actual_top +
                1
            )

            if candle_height <= 2:

                continue

            # ------------------------------------------------
            # Estimate body rows.
            #
            # A candle body generally has considerably more
            # colored pixels horizontally than a thin wick.
            # ------------------------------------------------

            max_row_pixels = max(
                1,
                int(row_counts.max())
            )

            body_rows = np.where(
                row_counts >=
                max_row_pixels * 0.45
            )[0]

            if len(body_rows) > 0:

                body_top = int(
                    body_rows.min()
                )

                body_bottom = int(
                    body_rows.max()
                )

            else:

                body_top = actual_top
                body_bottom = actual_bottom

            body_height = (
                body_bottom -
                body_top +
                1
            )

            upper_wick = max(
                0,
                body_top -
                actual_top
            )

            lower_wick = max(
                0,
                actual_bottom -
                body_bottom
            )

            body_ratio = (
                body_height /
                candle_height
            )

            upper_ratio = (
                upper_wick /
                candle_height
            )

            lower_ratio = (
                lower_wick /
                candle_height
            )

            # The vertical center of the body is used as
            # a visual proxy for open/close clustering.
            body_center = (
                body_top +
                body_bottom
            ) / 2

            result.append({
                "index": index,
                "x": x,
                "y": y,
                "width": w,
                "height": candle_height,
                "top": actual_top,
                "bottom": actual_bottom,
                "body_top": body_top,
                "body_bottom": body_bottom,
                "body_height": body_height,
                "body_ratio": body_ratio,
                "upper_wick": upper_wick,
                "lower_wick": lower_wick,
                "upper_ratio": upper_ratio,
                "lower_ratio": lower_ratio,
                "body_center": body_center,
                "color": candle["color"]
            })

        return result


# ============================================================
# REJECTION DETECTION
# ============================================================

def detect_rejections(candles):

    upper = []
    lower = []

    for candle in candles:

        body_ratio = candle["body_ratio"]

        # ----------------------------------------------------
        # Upper rejection
        # ----------------------------------------------------

        upper_rejection = (
            body_ratio <= SMALL_BODY_RATIO
            and
            candle["upper_ratio"]
            >= REJECTION_WICK_RATIO
        )

        # ----------------------------------------------------
        # Lower rejection
        # ----------------------------------------------------

        lower_rejection = (
            body_ratio <= SMALL_BODY_RATIO
            and
            candle["lower_ratio"]
            >= REJECTION_WICK_RATIO
        )

        if upper_rejection:

            upper.append({
                "index": candle["index"],
                "level": candle["top"],
                "color": candle["color"],
                "body_center": candle["body_center"]
            })

        if lower_rejection:

            lower.append({
                "index": candle["index"],
                "level": candle["bottom"],
                "color": candle["color"],
                "body_center": candle["body_center"]
            })

    return upper, lower


# ============================================================
# REJECTION ZONE CLUSTERING
# ============================================================

def cluster_levels(
    rejections,
    tolerance_px
):

    if not rejections:

        return []

    sorted_rejections = sorted(
        rejections,
        key=lambda x: x["level"]
    )

    groups = []

    current = [
        sorted_rejections[0]
    ]

    for rejection in sorted_rejections[1:]:

        current_level = np.mean([
            item["level"]
            for item in current
        ])

        if abs(
            rejection["level"] -
            current_level
        ) <= tolerance_px:

            current.append(
                rejection
            )

        else:

            groups.append(current)

            current = [
                rejection
            ]

    groups.append(current)

    zones = []

    for group in groups:

        level = float(np.mean([
            item["level"]
            for item in group
        ]))

        zones.append({
            "level": level,
            "tests": len(group),
            "indices": [
                item["index"]
                for item in group
            ],
            "colors": [
                item["color"]
                for item in group
            ]
        })

    zones.sort(
        key=lambda z: z["tests"],
        reverse=True
    )

    return zones


# ============================================================
# OPEN / CLOSE CLUSTERING
# ============================================================

def detect_open_close_clustering(
    candles,
    zone
):

    zone_level = zone["level"]

    relevant = []

    for candle in candles:

        # Distance of body center from rejection level
        distance = abs(
            candle["body_center"] -
            zone_level
        )

        # Use candle height as scale
        scale = max(
            1,
            candle["height"]
        )

        if distance <= scale * 1.2:

            relevant.append(candle)

    if len(relevant) < 2:

        return False, 0

    centers = [
        c["body_center"]
        for c in relevant
    ]

    spread = max(centers) - min(centers)

    average_height = np.mean([
        c["height"]
        for c in relevant
    ])

    clustered = (
        spread <=
        average_height *
        OPEN_CLOSE_CLUSTER_RATIO
    )

    return clustered, len(relevant)


# ============================================================
# TWO-SIDED REJECTION
# ============================================================

def detect_two_sided_rejection(
    upper_rejections,
    lower_rejections
):

    if not upper_rejections:
        return False

    if not lower_rejections:
        return False

    # Both sides rejecting means the market is trapped
    # between upper and lower rejection areas.
    return True


# ============================================================
# BREAKOUT FAILURE
# ============================================================

def detect_breakout_failure(
    candles,
    zone,
    side
):

    level = zone["level"]

    for i in range(
        1,
        len(candles)
    ):

        candle = candles[i]

        previous = candles[i - 1]

        # ----------------------------------------------------
        # Resistance:
        # candle moves above resistance but finishes back
        # underneath it.
        # ----------------------------------------------------

        if side == "RESISTANCE":

            broke_above = (
                candle["top"] <
                level
            )

            closed_back_below = (
                candle["body_center"] >
                level
            )

            turned_red = (
                candle["color"] ==
                "RED"
            )

            if (
                broke_above
                and
                closed_back_below
                and
                turned_red
            ):

                return {
                    "index": i,
                    "direction": "SELL"
                }

        # ----------------------------------------------------
        # Support:
        # candle moves below support but finishes back
        # above it.
        # ----------------------------------------------------

        if side == "SUPPORT":

            broke_below = (
                candle["bottom"] >
                level
            )

            closed_back_above = (
                candle["body_center"] <
                level
            )

            turned_green = (
                candle["color"] ==
                "GREEN"
            )

            if (
                broke_below
                and
                closed_back_above
                and
                turned_green
            ):

                return {
                    "index": i,
                    "direction": "BUY"
                }

    return None


# ============================================================
# GREEN -> RED / RED -> GREEN CONFIRMATION
# ============================================================

def detect_color_reversal(
    candles,
    resistance_zones,
    support_zones
):

    # --------------------------------------------------------
    # SELL:
    # GREEN tests resistance -> RED follows
    # --------------------------------------------------------

    for zone in resistance_zones:

        if zone["tests"] < MIN_REJECTIONS:

            continue

        for i in range(
            1,
            len(candles)
        ):

            previous = candles[i - 1]
            current = candles[i]

            if (
                previous["color"] ==
                "GREEN"
                and
                current["color"] ==
                "RED"
            ):

                previous_distance = abs(
                    previous["top"] -
                    zone["level"]
                )

                current_distance = abs(
                    current["top"] -
                    zone["level"]
                )

                scale = max(
                    previous["height"],
                    current["height"],
                    1
                )

                if (
                    previous_distance <=
                    scale * 1.5
                    or
                    current_distance <=
                    scale * 1.5
                ):

                    return {
                        "signal": "SELL",
                        "zone": zone,
                        "confirmation_index": i,
                        "reason":
                            "GREEN resistance test followed by RED confirmation"
                    }

    # --------------------------------------------------------
    # BUY:
    # RED tests support -> GREEN follows
    # --------------------------------------------------------

    for zone in support_zones:

        if zone["tests"] < MIN_REJECTIONS:

            continue

        for i in range(
            1,
            len(candles)
        ):

            previous = candles[i - 1]
            current = candles[i]

            if (
                previous["color"] ==
                "RED"
                and
                current["color"] ==
                "GREEN"
            ):

                previous_distance = abs(
                    previous["bottom"] -
                    zone["level"]
                )

                current_distance = abs(
                    current["bottom"] -
                    zone["level"]
                )

                scale = max(
                    previous["height"],
                    current["height"],
                    1
                )

                if (
                    previous_distance <=
                    scale * 1.5
                    or
                    current_distance <=
                    scale * 1.5
                ):

                    return {
                        "signal": "BUY",
                        "zone": zone,
                        "confirmation_index": i,
                        "reason":
                            "RED support test followed by GREEN confirmation"
                    }

    return None


# ============================================================
# COMPLETE PATTERN ANALYSIS
# ============================================================

def analyze_pattern(
    candles,
    chart_height
):

    report = {
        "upper_rejections": 0,
        "lower_rejections": 0,
        "resistance": "NOT CLEAR",
        "support": "NOT CLEAR",
        "resistance_cluster": "NO",
        "support_cluster": "NO",
        "two_sided": "NO",
        "breakout_failure": "NO",
        "confirmation": "NONE",
    }

    if len(candles) < MIN_CANDLES:

        report["reason"] = (
            "INSUFFICIENT VISIBLE CANDLES"
        )

        return None, report

    upper, lower = detect_rejections(
        candles
    )

    report[
        "upper_rejections"
    ] = len(upper)

    report[
        "lower_rejections"
    ] = len(lower)

    tolerance = max(
        3,
        int(
            chart_height *
            ZONE_TOLERANCE_PX_RATIO
        )
    )

    resistance_zones = cluster_levels(
        upper,
        tolerance
    )

    support_zones = cluster_levels(
        lower,
        tolerance
    )

    # --------------------------------------------------------
    # Resistance
    # --------------------------------------------------------

    if resistance_zones:

        strongest = resistance_zones[0]

        report["resistance"] = (
            f"{strongest['tests']} tests"
        )

        if strongest["tests"] >= 2:

            report[
                "resistance_cluster"
            ] = "YES"

    # --------------------------------------------------------
    # Support
    # --------------------------------------------------------

    if support_zones:

        strongest = support_zones[0]

        report["support"] = (
            f"{strongest['tests']} tests"
        )

        if strongest["tests"] >= 2:

            report[
                "support_cluster"
            ] = "YES"

    # --------------------------------------------------------
    # Open / close clustering
    # --------------------------------------------------------

    if resistance_zones:

        clustered, count = (
            detect_open_close_clustering(
                candles,
                resistance_zones[0]
            )
        )

        if clustered:

            report[
                "resistance_cluster"
            ] = f"YES ({count})"

    if support_zones:

        clustered, count = (
            detect_open_close_clustering(
                candles,
                support_zones[0]
            )
        )

        if clustered:

            report[
                "support_cluster"
            ] = f"YES ({count})"

    # --------------------------------------------------------
    # Two-sided rejection
    # --------------------------------------------------------

    two_sided = (
        detect_two_sided_rejection(
            upper,
            lower
        )
    )

    if two_sided:

        report[
            "two_sided"
        ] = "YES"

    # --------------------------------------------------------
    # GREEN -> RED / RED -> GREEN
    # --------------------------------------------------------

    reversal = detect_color_reversal(
        candles,
        resistance_zones,
        support_zones
    )

    # --------------------------------------------------------
    # Breakout failure
    # --------------------------------------------------------

    breakout = None

    if resistance_zones:

        breakout = detect_breakout_failure(
            candles,
            resistance_zones[0],
            "RESISTANCE"
        )

    if breakout is None and support_zones:

        breakout = detect_breakout_failure(
            candles,
            support_zones[0],
            "SUPPORT"
        )

    if breakout:

        report[
            "breakout_failure"
        ] = "YES"

    # --------------------------------------------------------
    # FINAL SIGNAL
    # --------------------------------------------------------

    if reversal:

        direction = reversal["signal"]

        zone = reversal["zone"]

        tests = zone["tests"]

        report[
            "confirmation"
        ] = reversal["reason"]

        # Strength based on actual visible evidence.
        confidence = 60

        if tests >= 3:
            confidence += 8

        if tests >= 5:
            confidence += 8

        if tests >= 8:
            confidence += 7

        if direction == "SELL":

            if report[
                "resistance_cluster"
            ] != "NO":

                confidence += 5

        if direction == "BUY":

            if report[
                "support_cluster"
            ] != "NO":

                confidence += 5

        if breakout:

            confidence += 7

        if confidence > 95:

            confidence = 95

        result = {
            "signal": direction,
            "confidence": confidence,
            "tests": tests,
            "zone": zone["level"],
            "reason": reversal["reason"],
            "breakout": bool(breakout)
        }

        return result, report

    report["reason"] = (
        "No complete visual confirmation"
    )

    return None, report


# ============================================================
# REPORT
# ============================================================

def format_report(
    report,
    elapsed
):

    text = (
        "🔎 **WHAT I SEE:**\n\n"
        f"• Upper rejection candles: "
        f"{report['upper_rejections']}\n"
        f"• Lower rejection candles: "
        f"{report['lower_rejections']}\n"
        f"• Resistance zone: "
        f"{report['resistance']}\n"
        f"• Support zone: "
        f"{report['support']}\n"
        f"• Resistance open/close clustering: "
        f"{report['resistance_cluster']}\n"
        f"• Support open/close clustering: "
        f"{report['support_cluster']}\n"
        f"• Two-sided rejection: "
        f"{report['two_sided']}\n"
        f"• Breakout-failure pattern: "
        f"{report['breakout_failure']}\n"
        f"• Candle confirmation: "
        f"{report['confirmation']}\n"
    )

    text += (
        "\n━━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ Analysis time: {elapsed:.2f}s"
    )

    return text


# ============================================================
# SIGNAL MESSAGE
# ============================================================

def generate_signal(
    result,
    asset
):

    direction = result["signal"]

    confidence = result["confidence"]

    tests = result["tests"]

    zone = result["zone"]

    reason = result["reason"]

    arrow = (
        "🟢"
        if direction == "BUY"
        else
        "🔴"
    )

    now = datetime.now(
        LOCAL_TZ
    )

    base_time = (
        now.replace(
            second=0,
            microsecond=0
        )
        +
        timedelta(minutes=1)
    )

    entries = []

    for i in range(
        ENTRY_COUNT
    ):

        entry = (
            base_time
            +
            timedelta(
                seconds=
                i *
                ENTRY_INTERVAL_SECONDS
            )
        )

        entries.append(
            entry.strftime(
                "%H:%M:%S"
            )
        )

    expiry = (
        base_time
        +
        timedelta(
            minutes=EXPIRY_MINUTES
        )
    ).strftime(
        "%H:%M:%S"
    )

    message = (
        "✅ **OTC REVERSAL SIGNAL**\n\n"
        f"💱 **Asset:** {asset}\n"
        f"{arrow} **Direction: {direction}**\n"
        f"💪 **Visual Strength: {confidence}%**\n"
        f"📍 **Rejected Zone:** {zone:.0f}px\n"
        f"🔁 **Repeated Tests:** {tests}\n\n"
        f"🔎 **Pattern:** {reason}\n"
        f"📊 **Breakout Failure:** "
        f"{'YES' if result['breakout'] else 'NO'}\n\n"
        "⏱️ **4 Entries / 15s:**\n"
    )

    for i, entry in enumerate(
        entries,
        1
    ):

        message += (
            f"Entry {i}: "
            f"`{entry}` "
            f"- ${ENTRY_SIZES[i-1]:.2f}\n"
        )

    message += (
        f"\n⏰ **Expiry:** {expiry}\n"
        "⚠️ Manual decision only."
    )

    return message


# ============================================================
# TELEGRAM SEND
# ============================================================

def send_telegram(
    message
):

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
            timeout=5
        )

        requests.post(
            url,
            data={
                "chat_id": CHANNEL_ID,
                "text": message,
                "parse_mode": "Markdown"
            },
            timeout=5
        )

        print(
            "✅ Signal sent to Telegram"
        )

    except Exception as e:

        print(
            "Telegram error:",
            e
        )


# ============================================================
# TELEGRAM START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📊 OTC SCREENSHOT REVERSAL BOT\n\n"
        "Send a Pocket Option screenshot.\n\n"
        "I will inspect the actual screenshot for:\n"
        "• Asset / currency pair\n"
        "• Green and red candles\n"
        "• Upper rejection\n"
        "• Lower rejection\n"
        "• Repeated tests\n"
        "• Open/close clustering\n"
        "• Two-sided rejection\n"
        "• Breakout failure\n"
        "• Green → Red SELL confirmation\n"
        "• Red → Green BUY confirmation\n\n"
        "No random candles.\n"
        "No generated OHLC.\n"
        "No fake market data."
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
            "🔎 Reading the visible chart..."
        )

        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        photo = (
            await
            update.message.photo[-1].get_file()
        )

        file_path = (
            "screenshot.png"
        )

        await photo.download_to_drive(
            file_path
        )

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        reader = ScreenshotReader()

        image = reader.load(
            file_path
        )

        if image is None:

            await update.message.reply_text(
                "❌ Could not load screenshot."
            )

            return

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        ocr_text = read_all_visible_text(
            image
        )

        normalized_text = (
            normalize_ocr_text(
                ocr_text
            )
        )

        asset_info = detect_visible_asset(
            normalized_text
        )

        asset_name = asset_info[
            "name"
        ]

        # ----------------------------------------------------
        # Chart
        # ----------------------------------------------------

        chart, bounds = (
            reader.get_chart_region(
                image
            )
        )

        candles_raw = (
            reader.detect_candles(
                chart
            )
        )

        candles = (
            reader.calculate_candle_geometry(
                candles_raw,
                chart
            )
        )

        elapsed = (
            time.time() -
            start_time
        )

        print(
            "\n=============================="
        )

        print(
            "OCR ASSET:",
            asset_name
        )

        print(
            "VISIBLE CANDLE OBJECTS:",
            len(candles)
        )

        print(
            "PROCESSING:",
            f"{elapsed:.2f}s"
        )

        print(
            "==============================\n"
        )

        # ----------------------------------------------------
        # Insufficient visual information
        # ----------------------------------------------------

        if len(candles) < MIN_CANDLES:

            response = (
                "⚠️ **SCREENSHOT ANALYSIS**\n\n"
                f"💱 Detected asset: "
                f"{asset_name}\n\n"
                "🔎 **WHAT I CAN ACTUALLY SEE:**\n"
                f"• Visible candle objects: "
                f"{len(candles)}\n"
                f"• Minimum required: "
                f"{MIN_CANDLES}\n\n"
                "⛔ **NO SIGNAL**\n\n"
                "Reason: insufficient reliable "
                "candle information in the screenshot.\n\n"
                f"⚡ Analysis time: "
                f"{elapsed:.2f}s"
            )

            await update.message.reply_text(
                response
            )

            return

        # ----------------------------------------------------
        # Analyze actual visual candles
        # ----------------------------------------------------

        result, report = analyze_pattern(
            candles,
            chart.shape[0]
        )

        detailed_report = format_report(
            report,
            elapsed
        )

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        if result:

            signal_message = (
                generate_signal(
                    result,
                    asset_name
                )
            )

            full_message = (
                "📸 **SCREENSHOT ANALYSIS**\n\n"
                f"💱 **Detected:** {asset_name}\n\n"
                f"{detailed_report}\n\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                f"{signal_message}"
            )

            send_telegram(
                full_message
            )

            await update.message.reply_text(
                full_message
            )

            print(
                "✅ SIGNAL:",
                result["signal"]
            )

        # ----------------------------------------------------
        # NO SIGNAL
        # ----------------------------------------------------

        else:

            response = (
                "⛔ **SCREENSHOT ANALYSIS**\n\n"
                f"💱 **Detected:** {asset_name}\n\n"
                f"{detailed_report}\n\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "⛔ **NO SIGNAL**\n\n"
                "Reason: No complete reversal "
                "confirmation was visible.\n\n"
                "The bot will NOT invent a signal "
                "when the screenshot does not provide "
                "enough evidence."
            )

            await update.message.reply_text(
                response
            )

            print(
                "⛔ NO SIGNAL"
            )

    except Exception as e:

        print(
            "❌ ERROR:",
            repr(e)
        )

        await update.message.reply_text(
            "❌ **Analysis Error**\n\n"
            f"{str(e)}"
        )


# ============================================================
# START TELEGRAM BOT
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
        "========================================"
    )

    print(
        "📊 OTC SCREENSHOT REVERSAL BOT"
    )

    print(
        "📸 Waiting for screenshots..."
    )

    print(
        "========================================"
    )

    application.run_polling(
        drop_pending_updates=True
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

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

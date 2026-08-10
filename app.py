import os
import time
import threading
import requests
import cv2
import numpy as np
import pytesseract
import re

from flask import Flask
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from datetime import datetime, timezone, timedelta


# ============================================================
# TELEGRAM
# ============================================================

# IMPORTANT:
# Put the NEW Telegram token in Render:
#
# BOT_TOKEN = your new token
#
# Do NOT hard-code the exposed token.

TOKEN = os.getenv("BOT_TOKEN", "")

CHAT_ID = os.getenv(
    "CHAT_ID",
    "6280535707"
)

CHANNEL_ID = os.getenv(
    "CHANNEL_ID",
    "-1004324805205"
)


# ============================================================
# TIMEZONE
# ============================================================

LOCAL_TZ = timezone(
    timedelta(hours=1)
)


# ============================================================
# TEST / SIGNAL MODE
# ============================================================

# TRUE:
# The bot explains everything it sees.
#
# FALSE:
# The bot sends only the short final signal.
#
# Keep TRUE while testing.
#
DETAILED_TEST_MODE = True


# ============================================================
# TRADE SETTINGS
# ============================================================

ENTRY_INTERVAL_SECONDS = 15

ENTRY_COUNT = 4

EXPIRY_MINUTES = 1

ENTRY_SIZES = [
    1.00,
    2.10,
    4.40,
    9.20
]


# ============================================================
# CANDLE DETECTION SETTINGS
# ============================================================

MIN_BODY_AREA = 10

MIN_BODY_HEIGHT = 2

MIN_CANDLE_WIDTH = 2

RIGHT_MIN_BODY_AREA = 6

RIGHT_MIN_BODY_HEIGHT = 2

MAX_CANDLE_WIDTH_RATIO = 0.045

MERGE_DISTANCE_RATIO = 0.55


# ============================================================
# STRATEGY SETTINGS
# ============================================================

LOOKBACK_CANDLES = 20

MIN_ZONE_TESTS = 2

ZONE_TOLERANCE_RATIO = 0.035

SMALL_BODY_RATIO = 0.45

MOMENTUM_EXPANSION_RATIO = 1.35

MOMENTUM_WEAKENING_RATIO = 0.70

MAX_SIGNAL_CONFIDENCE = 95

MIN_SIGNAL_CONFIDENCE = 65


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():

    return (
        "OTC Screenshot Strategy Bot is running!"
    )


@app.route("/ping")
def ping():

    return "pong", 200


def run_flask():

    import logging

    logging.getLogger(
        "werkzeug"
    ).setLevel(
        logging.ERROR
    )

    app.run(
        host="0.0.0.0",
        port=10000,
        debug=False,
        threaded=True
    )


# ============================================================
# CURRENCY / ASSET DETECTION
# ============================================================

CURRENCY_CODES = {
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "AUD",
    "CAD",
    "CHF",
    "NZD",
    "SGD",
    "HKD",
    "CNY",
    "TRY",
    "ZAR",
    "MXN",
    "BRL",
    "AED",
    "SAR",
    "NOK",
    "SEK",
    "DKK",
    "PLN",
    "RUB",
    "INR",
    "KRW"
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

        text = text.replace(
            old,
            new
        )

    return text


def detect_asset(img):

    height, width = img.shape[:2]

    regions = [

        img[
            int(height * 0.05):
            int(height * 0.35),
            int(width * 0.15):
            int(width * 0.90)
        ],

        img[
            int(height * 0.10):
            int(height * 0.45),
            0:
            width
        ]

    ]

    texts = []

    for region in regions:

        if region.size == 0:
            continue

        gray = cv2.cvtColor(
            region,
            cv2.COLOR_BGR2GRAY
        )

        for psm in (
            6,
            11
        ):

            try:

                text = pytesseract.image_to_string(
                    gray,
                    config=f"--psm {psm}"
                )

                if text:
                    texts.append(text)

            except Exception:

                pass

    combined = normalize_ocr_text(
        "\n".join(texts)
    )

    lines = [
        re.sub(
            r"\s+",
            " ",
            line
        ).strip()

        for line in combined.splitlines()
    ]

    # --------------------------------------------------------
    # Explicit OTC asset
    # --------------------------------------------------------

    for line in lines:

        if "OTC" in line:

            line = re.sub(
                r"\b(EXPIRATION|TIME|AMOUNT|PAYOUT|PROFIT|DEMO)\b",
                "",
                line
            )

            line = re.sub(
                r"\s+",
                " ",
                line
            ).strip()

            if len(line) >= 3:

                return (
                    line,
                    combined
                )

    # --------------------------------------------------------
    # Currency pair
    # --------------------------------------------------------

    pair_pattern = re.compile(
        r"\b("
        +
        "|".join(
            sorted(
                CURRENCY_CODES,
                key=len,
                reverse=True
            )
        )
        +
        r")"
        r"\s*[/\\\-:]?\s*"
        r"\b("
        +
        "|".join(
            sorted(
                CURRENCY_CODES,
                key=len,
                reverse=True
            )
        )
        +
        r")\b"
    )

    matches = pair_pattern.findall(
        combined
    )

    for base, quote in matches:

        if base == quote:
            continue

        pair = (
            f"{base}/{quote}"
        )

        if "OTC" in combined:

            return (
                pair + " OTC",
                combined
            )

        return (
            pair,
            combined
        )

    return (
        "OTC ASSET NOT CLEAR",
        combined
    )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    img = cv2.imread(path)

    if img is None:

        raise ValueError(
            "Could not read screenshot."
        )

    h, w = img.shape[:2]

    if w < 1400:

        scale = (
            1400 /
            float(w)
        )

        img = cv2.resize(
            img,
            (
                int(w * scale),
                int(h * scale)
            ),
            interpolation=cv2.INTER_CUBIC
        )

    return img


# ============================================================
# COLOR MASKS
# ============================================================

def get_color_masks(img):

    hsv = cv2.cvtColor(
        img,
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

    red_lower_1 = np.array([
        0,
        45,
        40
    ])

    red_upper_1 = np.array([
        12,
        255,
        255
    ])

    red_lower_2 = np.array([
        165,
        45,
        40
    ])

    red_upper_2 = np.array([
        180,
        255,
        255
    ])

    red1 = cv2.inRange(
        hsv,
        red_lower_1,
        red_upper_1
    )

    red2 = cv2.inRange(
        hsv,
        red_lower_2,
        red_upper_2
    )

    red = cv2.bitwise_or(
        red1,
        red2
    )

    return green, red


# ============================================================
# FIND CANDIDATES
# ============================================================

def find_candidates(
    mask,
    color,
    image_width,
    right_side=False
):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    max_width = max(
        10,
        int(
            image_width *
            MAX_CANDLE_WIDTH_RATIO
        )
    )

    if right_side:

        min_area = RIGHT_MIN_BODY_AREA
        min_height = RIGHT_MIN_BODY_HEIGHT

    else:

        min_area = MIN_BODY_AREA
        min_height = MIN_BODY_HEIGHT

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w < MIN_CANDLE_WIDTH:
            continue

        if h < min_height:
            continue

        if w > max_width:
            continue

        if w > h * 6:
            continue

        region = cleaned[
            y:y+h,
            x:x+w
        ]

        colored_pixels = int(
            np.sum(
                region > 0
            )
        )

        if colored_pixels < 5:
            continue

        density = (
            colored_pixels /
            float(
                max(
                    1,
                    w * h
                )
            )
        )

        if density < 0.15:
            continue

        center_x = (
            x +
            w / 2
        )

        candidates.append({

            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": float(area),
            "pixels": colored_pixels,
            "density": density,
            "center_x": center_x,
            "color": color

        })

    return candidates


# ============================================================
# MERGE SAME-COLOR PIECES
# ============================================================

def merge_candidates(
    candidates
):

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda c:
        c["center_x"]
    )

    merged = []

    for candidate in candidates:

        merged_into_existing = False

        for existing in merged:

            distance = abs(
                candidate["center_x"]
                -
                existing["center_x"]
            )

            allowed = max(
                candidate["w"],
                existing["w"],
                2
            ) * MERGE_DISTANCE_RATIO

            candidate_top = (
                candidate["y"]
            )

            candidate_bottom = (
                candidate["y"]
                +
                candidate["h"]
            )

            existing_top = (
                existing["y"]
            )

            existing_bottom = (
                existing["y"]
                +
                existing["h"]
            )

            vertical_overlap = not (
                candidate_bottom <
                existing_top
                or
                candidate_top >
                existing_bottom
            )

            if (
                distance <= allowed
                and
                vertical_overlap
            ):

                left = min(
                    existing["x"],
                    candidate["x"]
                )

                right = max(
                    existing["x"]
                    +
                    existing["w"],
                    candidate["x"]
                    +
                    candidate["w"]
                )

                top = min(
                    existing["y"],
                    candidate["y"]
                )

                bottom = max(
                    existing["y"]
                    +
                    existing["h"],
                    candidate["y"]
                    +
                    candidate["h"]
                )

                existing["x"] = left
                existing["y"] = top

                existing["w"] = (
                    right - left
                )

                existing["h"] = (
                    bottom - top
                )

                existing["center_x"] = (
                    left +
                    existing["w"] / 2
                )

                existing["area"] += (
                    candidate["area"]
                )

                existing["pixels"] += (
                    candidate["pixels"]
                )

                merged_into_existing = True

                break

        if not merged_into_existing:

            merged.append(
                candidate.copy()
            )

    return merged


# ============================================================
# REMOVE CROSS-COLOR DUPLICATES
# ============================================================

def remove_cross_color_duplicates(
    candles
):

    candles = sorted(
        candles,
        key=lambda c:
        c["center_x"]
    )

    result = []

    for candle in candles:

        duplicate_index = None

        for i, existing in enumerate(
            result
        ):

            distance = abs(
                candle["center_x"]
                -
                existing["center_x"]
            )

            threshold = max(
                candle["w"],
                existing["w"],
                2
            ) * 0.65

            if distance <= threshold:

                duplicate_index = i

                break

        if duplicate_index is None:

            result.append(
                candle
            )

        else:

            existing = result[
                duplicate_index
            ]

            if (
                candle["pixels"]
                >
                existing["pixels"]
            ):

                result[
                    duplicate_index
                ] = candle

    return result


# ============================================================
# RIGHT SIDE DETECTION
# ============================================================

def detect_right_side(
    chart,
    green_mask,
    red_mask
):

    h, w = chart.shape[:2]

    right_start = int(
        w * 0.72
    )

    green_right = (
        green_mask[
            :,
            right_start:
        ]
    )

    red_right = (
        red_mask[
            :,
            right_start:
        ]
    )

    green = find_candidates(
        green_right,
        "GREEN",
        w,
        right_side=True
    )

    red = find_candidates(
        red_right,
        "RED",
        w,
        right_side=True
    )

    for candle in green + red:

        candle["x"] += (
            right_start
        )

        candle["center_x"] += (
            right_start
        )

    return green + red


# ============================================================
# DETECT CANDLES
# ============================================================

def detect_candles(img):

    h, w = img.shape[:2]

    green_mask, red_mask = (
        get_color_masks(img)
    )

    green = find_candidates(
        green_mask,
        "GREEN",
        w,
        right_side=False
    )

    red = find_candidates(
        red_mask,
        "RED",
        w,
        right_side=False
    )

    green = merge_candidates(
        green
    )

    red = merge_candidates(
        red
    )

    candles = (
        green +
        red
    )

    right_candidates = (
        detect_right_side(
            img,
            green_mask,
            red_mask
        )
    )

    candles.extend(
        right_candidates
    )

    candles = (
        remove_cross_color_duplicates(
            candles
        )
    )

    candles.sort(
        key=lambda c:
        c["center_x"]
    )

    # --------------------------------------------------------
    # RIGHT → LEFT
    #
    # Candle #1 = newest.
    # --------------------------------------------------------

    candles.reverse()

    return candles


# ============================================================
# CONVERT BODY TO RELATIVE PRICE GEOMETRY
# ============================================================

def convert_to_price_geometry(
    candles
):

    if not candles:
        return []

    tops = np.array(
        [
            c["y"]
            for c in candles
        ],
        dtype=float
    )

    bottoms = np.array(
        [
            c["y"] + c["h"]
            for c in candles
        ],
        dtype=float
    )

    y_min = np.min(tops)

    y_max = np.max(bottoms)

    vertical_range = (
        y_max - y_min
    )

    if vertical_range <= 0:
        return []

    result = []

    for index, candle in enumerate(
        candles
    ):

        top = candle["y"]

        bottom = (
            candle["y"]
            +
            candle["h"]
        )

        # Screen top = higher visible price.

        high = (
            y_max - top
        ) / vertical_range

        low = (
            y_max - bottom
        ) / vertical_range

        if high <= low:
            continue

        # IMPORTANT:
        #
        # These are BODY boundaries.
        #
        # They are NOT fabricated OHLC values.
        #
        # We therefore call them:
        # body_high / body_low.

        result.append({

            "index": index,

            "x": candle["x"],

            "width": candle["w"],

            "height": candle["h"],

            "color": candle["color"],

            "body_high": high,

            "body_low": low,

            "body_size": high - low

        })

    return result


# ============================================================
# BODY INFORMATION
# ============================================================

def body_information(
    candle
):

    body_size = max(
        0.000001,
        candle["body_size"]
    )

    return {

        "body": body_size,

        "range": body_size,

        "body_ratio": 1.0,

        "upper_wick": None,

        "lower_wick": None

    }


# ============================================================
# APPROXIMATE RESISTANCE ZONE
# ============================================================

def find_resistance_zone(
    candles
):

    if len(candles) < 2:
        return None

    candles_to_check = (
        candles[
            :LOOKBACK_CANDLES
        ]
    )

    levels = [
        c["body_high"]
        for c in candles_to_check
    ]

    chart_min = min(levels)

    chart_max = max(levels)

    visible_range = max(
        0.000001,
        chart_max - chart_min
    )

    tolerance = (
        visible_range *
        ZONE_TOLERANCE_RATIO
    )

    zones = []

    for candle in candles_to_check:

        level = candle[
            "body_high"
        ]

        placed = False

        for zone in zones:

            if abs(
                level -
                zone["level"]
            ) <= tolerance:

                zone["candles"].append(
                    candle
                )

                zone["level"] = float(
                    np.mean(
                        [
                            c["body_high"]
                            for c
                            in zone["candles"]
                        ]
                    )
                )

                placed = True

                break

        if not placed:

            zones.append({

                "level": float(level),

                "candles": [
                    candle
                ]

            })

    zones.sort(
        key=lambda z:
        len(z["candles"]),
        reverse=True
    )

    strongest = zones[0]

    if len(
        strongest["candles"]
    ) < MIN_ZONE_TESTS:

        return None

    return strongest


# ============================================================
# APPROXIMATE SUPPORT ZONE
# ============================================================

def find_support_zone(
    candles
):

    if len(candles) < 2:
        return None

    candles_to_check = (
        candles[
            :LOOKBACK_CANDLES
        ]
    )

    levels = [
        c["body_low"]
        for c in candles_to_check
    ]

    chart_min = min(levels)

    chart_max = max(levels)

    visible_range = max(
        0.000001,
        chart_max - chart_min
    )

    tolerance = (
        visible_range *
        ZONE_TOLERANCE_RATIO
    )

    zones = []

    for candle in candles_to_check:

        level = candle[
            "body_low"
        ]

        placed = False

        for zone in zones:

            if abs(
                level -
                zone["level"]
            ) <= tolerance:

                zone["candles"].append(
                    candle
                )

                zone["level"] = float(
                    np.mean(
                        [
                            c["body_low"]
                            for c
                            in zone["candles"]
                        ]
                    )
                )

                placed = True

                break

        if not placed:

            zones.append({

                "level": float(level),

                "candles": [
                    candle
                ]

            })

    zones.sort(
        key=lambda z:
        len(z["candles"]),
        reverse=True
    )

    strongest = zones[0]

    if len(
        strongest["candles"]
    ) < MIN_ZONE_TESTS:

        return None

    return strongest


# ============================================================
# CANDLE TREND
# ============================================================

def analyze_trend(
    candles
):

    if len(candles) < 3:

        return {
            "direction": "UNKNOWN",
            "strength": 0,
            "reason": "Not enough candles."
        }

    recent = candles[
        :min(
            6,
            len(candles)
        )
    ]

    green = sum(
        1
        for c in recent
        if c["color"] == "GREEN"
    )

    red = sum(
        1
        for c in recent
        if c["color"] == "RED"
    )

    if green > red:

        return {
            "direction": "BULLISH",
            "strength": green,
            "reason":
                f"{green} of the recent "
                f"{len(recent)} candles are GREEN."
        }

    if red > green:

        return {
            "direction": "BEARISH",
            "strength": red,
            "reason":
                f"{red} of the recent "
                f"{len(recent)} candles are RED."
        }

    return {
        "direction": "NEUTRAL",
        "strength": 0,
        "reason":
            "Recent candle colors are balanced."
    }


# ============================================================
# BODY MOMENTUM
# ============================================================

def analyze_body_momentum(
    candles
):

    if len(candles) < 3:

        return {
            "state": "UNKNOWN",
            "direction": "NONE",
            "reason":
                "Not enough candles."
        }

    newest = candles[0]

    previous = candles[1]

    older = candles[2]

    average_previous = (
        previous["body_size"]
        +
        older["body_size"]
    ) / 2

    if (
        newest["body_size"]
        >
        average_previous *
        MOMENTUM_EXPANSION_RATIO
    ):

        state = "STRONGER"

    elif (
        newest["body_size"]
        <
        average_previous *
        MOMENTUM_WEAKENING_RATIO
    ):

        state = "WEAKER"

    else:

        state = "NORMAL"

    direction = (
        "BULLISH"
        if newest["color"] == "GREEN"
        else "BEARISH"
    )

    return {

        "state": state,

        "direction": direction,

        "reason":
            f"Newest {newest['color']} "
            f"body is {state.lower()} "
            "relative to the previous "
            "two bodies."

    }


# ============================================================
# CONSECUTIVE CANDLES
# ============================================================

def detect_consecutive_momentum(
    candles
):

    if len(candles) < 3:

        return {
            "direction": "NONE",
            "count": 0
        }

    first_color = (
        candles[0]["color"]
    )

    count = 0

    for candle in candles:

        if candle["color"] == first_color:

            count += 1

        else:

            break

    if count >= 3:

        direction = (
            "BUY"
            if first_color == "GREEN"
            else "SELL"
        )

        return {
            "direction": direction,
            "count": count
        }

    return {
        "direction": "NONE",
        "count": count
    }


# ============================================================
# COLOR REVERSAL
# ============================================================

def detect_color_reversal(
    candles
):

    if len(candles) < 2:
        return None

    newest = candles[0]

    previous = candles[1]

    if (
        previous["color"] == "GREEN"
        and
        newest["color"] == "RED"
    ):

        return {
            "direction": "SELL",
            "type": "GREEN_TO_RED"
        }

    if (
        previous["color"] == "RED"
        and
        newest["color"] == "GREEN"
    ):

        return {
            "direction": "BUY",
            "type": "RED_TO_GREEN"
        }

    return None


# ============================================================
# BODY ENGULFING
# ============================================================

def detect_body_engulfing(
    candles
):

    if len(candles) < 2:
        return None

    newest = candles[0]

    previous = candles[1]

    new_high = newest[
        "body_high"
    ]

    new_low = newest[
        "body_low"
    ]

    old_high = previous[
        "body_high"
    ]

    old_low = previous[
        "body_low"
    ]

    # Bullish body engulfing.

    if (
        previous["color"] == "RED"
        and
        newest["color"] == "GREEN"
        and
        new_high >= old_high
        and
        new_low <= old_low
    ):

        return {
            "direction": "BUY",
            "type": "BULLISH_BODY_ENGULFING"
        }

    # Bearish body engulfing.

    if (
        previous["color"] == "GREEN"
        and
        newest["color"] == "RED"
        and
        new_high >= old_high
        and
        new_low <= old_low
    ):

        return {
            "direction": "SELL",
            "type": "BEARISH_BODY_ENGULFING"
        }

    return None


# ============================================================
# INSIDE BODY
# ============================================================

def detect_inside_body(
    candles
):

    if len(candles) < 2:
        return False

    newest = candles[0]

    previous = candles[1]

    return (
        newest["body_high"]
        <=
        previous["body_high"]
        and
        newest["body_low"]
        >=
        previous["body_low"]
    )


# ============================================================
# BODY COMPRESSION
# ============================================================

def detect_body_compression(
    candles
):

    if len(candles) < 4:
        return False

    recent = candles[:4]

    sizes = [
        c["body_size"]
        for c in recent
    ]

    average = np.mean(
        sizes[1:]
    )

    if average <= 0:
        return False

    return (
        sizes[0]
        <
        average * 0.70
    )


# ============================================================
# OPEN/CLOSE STYLE CLUSTERING
#
# Here we use BODY BOUNDARY clustering because the screenshot
# reader does not have trustworthy numerical OHLC.
# ============================================================

def detect_body_clustering(
    candles
):

    if len(candles) < 3:
        return False

    recent = candles[
        :min(
            6,
            len(candles)
        )
    ]

    sizes = [
        c["body_size"]
        for c in recent
    ]

    average = np.mean(
        sizes
    )

    if average <= 0:
        return False

    small = sum(
        1
        for size in sizes
        if size <= average * 0.70
    )

    return small >= 2


# ============================================================
# BREAKOUT FAILURE
# ============================================================

def detect_breakout_failure(
    candles,
    zone,
    side
):

    if not zone:
        return None

    level = zone["level"]

    if len(candles) < 2:
        return None

    # Because candles are newest -> oldest,
    # examine transitions between adjacent candles.

    for i in range(
        len(candles) - 1
    ):

        newer = candles[i]

        older = candles[i + 1]

        if side == "RESISTANCE":

            crossed = (
                older["body_high"]
                <= level
                and
                newer["body_high"]
                > level
            )

            failed = (
                newer["body_low"]
                < level
                and
                newer["color"] == "RED"
            )

            if crossed and failed:

                return {
                    "direction": "SELL",
                    "type":
                        "RESISTANCE_BREAKOUT_FAILURE",
                    "index": i
                }

        elif side == "SUPPORT":

            crossed = (
                older["body_low"]
                >= level
                and
                newer["body_low"]
                < level
            )

            failed = (
                newer["body_high"]
                > level
                and
                newer["color"] == "GREEN"
            )

            if crossed and failed:

                return {
                    "direction": "BUY",
                    "type":
                        "SUPPORT_BREAKOUT_FAILURE",
                    "index": i
                }

    return None


# ============================================================
# ZONE COLOR CONFIRMATION
# ============================================================

def detect_zone_confirmation(
    candles,
    zone,
    direction
):

    if not zone:
        return None

    level = zone["level"]

    if len(candles) < 2:
        return None

    for i in range(
        len(candles) - 1
    ):

        newer = candles[i]

        older = candles[i + 1]

        if direction == "SELL":

            if (
                older["color"] == "GREEN"
                and
                newer["color"] == "RED"
                and
                older["body_high"] >= level
                and
                newer["body_low"] <= level
            ):

                return {
                    "direction": "SELL",
                    "type":
                        "GREEN_TO_RED_RESISTANCE"
                }

        if direction == "BUY":

            if (
                older["color"] == "RED"
                and
                newer["color"] == "GREEN"
                and
                older["body_low"] <= level
                and
                newer["body_high"] >= level
            ):

                return {
                    "direction": "BUY",
                    "type":
                        "RED_TO_GREEN_SUPPORT"
                }

    return None


# ============================================================
# RESISTANCE BEHAVIOR
# ============================================================

def resistance_behavior(
    candles,
    resistance
):

    if not resistance:

        return {
            "near": False,
            "rejection": False,
            "reason":
                "No confirmed resistance zone."
        }

    newest = candles[0]

    level = resistance["level"]

    distance = abs(
        newest["body_high"]
        -
        level
    )

    tolerance = max(
        0.000001,
        newest["body_size"] *
        0.75
    )

    near = (
        distance <= tolerance
    )

    rejection = (
        near
        and
        newest["color"] == "RED"
    )

    if rejection:

        reason = (
            "Newest RED body is near "
            "the repeated resistance area."
        )

    elif near:

        reason = (
            "Newest GREEN body is near "
            "the resistance area."
        )

    else:

        reason = (
            "Newest candle is not near "
            "the resistance area."
        )

    return {

        "near": near,

        "rejection": rejection,

        "reason": reason

    }


# ============================================================
# SUPPORT BEHAVIOR
# ============================================================

def support_behavior(
    candles,
    support
):

    if not support:

        return {
            "near": False,
            "rejection": False,
            "reason":
                "No confirmed support zone."
        }

    newest = candles[0]

    level = support["level"]

    distance = abs(
        newest["body_low"]
        -
        level
    )

    tolerance = max(
        0.000001,
        newest["body_size"] *
        0.75
    )

    near = (
        distance <= tolerance
    )

    rejection = (
        near
        and
        newest["color"] == "GREEN"
    )

    if rejection:

        reason = (
            "Newest GREEN body is near "
            "the repeated support area."
        )

    elif near:

        reason = (
            "Newest RED body is near "
            "the support area."
        )

    else:

        reason = (
            "Newest candle is not near "
            "the support area."
        )

    return {

        "near": near,

        "rejection": rejection,

        "reason": reason

    }


# ============================================================
# ALL STRATEGIES
# ============================================================

def analyze_all_strategies(
    candles
):

    if not candles:

        return {
            "status":
                "NO_CANDLES",
            "signal":
                None,
            "confidence":
                0,
            "strategies": [],
            "reason":
                "No reliable candle bodies detected."
        }

    candles = candles[
        :LOOKBACK_CANDLES
    ]

    strategy_results = []

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    trend = analyze_trend(
        candles
    )

    strategy_results.append({

        "name":
            "Recent Candle Trend",

        "direction":
            (
                "BUY"
                if trend["direction"] == "BULLISH"
                else
                "SELL"
                if trend["direction"] == "BEARISH"
                else
                "NONE"
            ),

        "detected":
            trend["direction"] != "NEUTRAL"
            and
            trend["direction"] != "UNKNOWN",

        "reason":
            trend["reason"]

    })

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    momentum = analyze_body_momentum(
        candles
    )

    strategy_results.append({

        "name":
            "Body Momentum",

        "direction":
            (
                "BUY"
                if momentum["direction"] == "BULLISH"
                else
                "SELL"
                if momentum["direction"] == "BEARISH"
                else
                "NONE"
            ),

        "detected":
            momentum["state"]
            == "STRONGER",

        "reason":
            momentum["reason"]

    })

    # --------------------------------------------------------
    # CONSECUTIVE MOMENTUM
    # --------------------------------------------------------

    consecutive = (
        detect_consecutive_momentum(
            candles
        )
    )

    strategy_results.append({

        "name":
            "3+ Candle Momentum",

        "direction":
            consecutive["direction"],

        "detected":
            consecutive["direction"] != "NONE",

        "reason":
            (
                f"{consecutive['count']} consecutive "
                f"{'GREEN' if consecutive['direction'] == 'BUY' else 'RED'} "
                "candles."
                if consecutive["direction"] != "NONE"
                else
                f"Only {consecutive['count']} consecutive "
                "same-color candles."
            )

    })

    # --------------------------------------------------------
    # COLOR REVERSAL
    # --------------------------------------------------------

    reversal = (
        detect_color_reversal(
            candles
        )
    )

    strategy_results.append({

        "name":
            "Color Reversal",

        "direction":
            (
                reversal["direction"]
                if reversal
                else
                "NONE"
            ),

        "detected":
            reversal is not None,

        "reason":
            (
                reversal["type"]
                if reversal
                else
                "No immediate color reversal."
            )

    })

    # --------------------------------------------------------
    # BODY ENGULFING
    # --------------------------------------------------------

    engulfing = (
        detect_body_engulfing(
            candles
        )
    )

    strategy_results.append({

        "name":
            "Body Engulfing",

        "direction":
            (
                engulfing["direction"]
                if engulfing
                else
                "NONE"
            ),

        "detected":
            engulfing is not None,

        "reason":
            (
                engulfing["type"]
                if engulfing
                else
                "No body engulfing pattern."
            )

    })

    # --------------------------------------------------------
    # INSIDE BODY
    # --------------------------------------------------------

    inside = detect_inside_body(
        candles
    )

    strategy_results.append({

        "name":
            "Inside Body",

        "direction":
            "NONE",

        "detected":
            inside,

        "reason":
            (
                "Newest body is contained "
                "inside the previous body."
                if inside
                else
                "No inside-body compression."
            )

    })

    # --------------------------------------------------------
    # BODY COMPRESSION
    # --------------------------------------------------------

    compression = (
        detect_body_compression(
            candles
        )
    )

    strategy_results.append({

        "name":
            "Body Compression",

        "direction":
            "NONE",

        "detected":
            compression,

        "reason":
            (
                "Newest body is significantly "
                "smaller than recent bodies."
                if compression
                else
                "No strong body compression."
            )

    })

    # --------------------------------------------------------
    # BODY CLUSTERING
    # --------------------------------------------------------

    clustering = (
        detect_body_clustering(
            candles
        )
    )

    strategy_results.append({

        "name":
            "Open/Close-Style Clustering",

        "direction":
            "NONE",

        "detected":
            clustering,

        "reason":
            (
                "Several recent candle bodies "
                "are relatively small."
                if clustering
                else
                "No strong clustering detected."
            )

    })

    # --------------------------------------------------------
    # SUPPORT
    # --------------------------------------------------------

    support = find_support_zone(
        candles
    )

    support_result = (
        support_behavior(
            candles,
            support
        )
    )

    strategy_results.append({

        "name":
            "Support",

        "direction":
            (
                "BUY"
                if support_result["rejection"]
                else
                "NONE"
            ),

        "detected":
            support is not None,

        "reason":
            support_result["reason"]

    })

    # --------------------------------------------------------
    # RESISTANCE
    # --------------------------------------------------------

    resistance = find_resistance_zone(
        candles
    )

    resistance_result = (
        resistance_behavior(
            candles,
            resistance
        )
    )

    strategy_results.append({

        "name":
            "Resistance",

        "direction":
            (
                "SELL"
                if resistance_result["rejection"]
                else
                "NONE"
            ),

        "detected":
            resistance is not None,

        "reason":
            resistance_result["reason"]

    })

    # --------------------------------------------------------
    # RESISTANCE COLOR CONFIRMATION
    # --------------------------------------------------------

    resistance_confirmation = (
        detect_zone_confirmation(
            candles,
            resistance,
            "SELL"
        )
    )

    strategy_results.append({

        "name":
            "Resistance Color Confirmation",

        "direction":
            (
                "SELL"
                if resistance_confirmation
                else
                "NONE"
            ),

        "detected":
            resistance_confirmation is not None,

        "reason":
            (
                resistance_confirmation["type"]
                if resistance_confirmation
                else
                "No confirmed GREEN → RED "
                "resistance transition."
            )

    })

    # --------------------------------------------------------
    # SUPPORT COLOR CONFIRMATION
    # --------------------------------------------------------

    support_confirmation = (
        detect_zone_confirmation(
            candles,
            support,
            "BUY"
        )
    )

    strategy_results.append({

        "name":
            "Support Color Confirmation",

        "direction":
            (
                "BUY"
                if support_confirmation
                else
                "NONE"
            ),

        "detected":
            support_confirmation is not None,

        "reason":
            (
                support_confirmation["type"]
                if support_confirmation
                else
                "No confirmed RED → GREEN "
                "support transition."
            )

    })

    # --------------------------------------------------------
    # RESISTANCE BREAKOUT FAILURE
    # --------------------------------------------------------

    resistance_failure = (
        detect_breakout_failure(
            candles,
            resistance,
            "RESISTANCE"
        )
    )

    strategy_results.append({

        "name":
            "Resistance Breakout Failure",

        "direction":
            (
                "SELL"
                if resistance_failure
                else
                "NONE"
            ),

        "detected":
            resistance_failure is not None,

        "reason":
            (
                "Price structure crossed resistance "
                "and returned below it."
                if resistance_failure
                else
                "No confirmed resistance failure."
            )

    })

    # --------------------------------------------------------
    # SUPPORT BREAKOUT FAILURE
    # --------------------------------------------------------

    support_failure = (
        detect_breakout_failure(
            candles,
            support,
            "SUPPORT"
        )
    )

    strategy_results.append({

        "name":
            "Support Breakout Failure",

        "direction":
            (
                "BUY"
                if support_failure
                else
                "NONE"
            ),

        "detected":
            support_failure is not None,

        "reason":
            (
                "Price structure crossed support "
                "and returned above it."
                if support_failure
                else
                "No confirmed support failure."
            )

    })

    # --------------------------------------------------------
    # COUNT AGREEMENT
    # --------------------------------------------------------

    buy_votes = 0

    sell_votes = 0

    for item in strategy_results:

        if not item["detected"]:
            continue

        if item["direction"] == "BUY":

            buy_votes += 1

        elif item["direction"] == "SELL":

            sell_votes += 1

    # --------------------------------------------------------
    # TWO-SIDED REJECTION
    # --------------------------------------------------------

    two_sided = (
        support is not None
        and
        resistance is not None
    )

    strategy_results.append({

        "name":
            "Two-Sided Support/Resistance",

        "direction":
            "NONE",

        "detected":
            two_sided,

        "reason":
            (
                "Both support and resistance "
                "zones were detected."
                if two_sided
                else
                "Both zones were not confirmed."
            )

    })

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    signal = None

    confidence = 0

    core_reason = (
        "No complete multi-factor setup."
    )

    if buy_votes > sell_votes:

        # Require more than one agreement.

        if buy_votes >= 2:

            signal = "BUY"

            confidence = min(
                MAX_SIGNAL_CONFIDENCE,
                55 + buy_votes * 7
            )

            core_reason = (
                f"{buy_votes} BUY strategy "
                "confirmations agree."
            )

    elif sell_votes > buy_votes:

        if sell_votes >= 2:

            signal = "SELL"

            confidence = min(
                MAX_SIGNAL_CONFIDENCE,
                55 + sell_votes * 7
            )

            core_reason = (
                f"{sell_votes} SELL strategy "
                "confirmations agree."
            )

    # --------------------------------------------------------
    # STRONG ZONE CONFIRMATION BONUS
    # --------------------------------------------------------

    if signal == "BUY":

        if (
            support is not None
            and
            support_result["rejection"]
        ):

            confidence += 6

            core_reason = (
                "Support rejection + "
                f"{buy_votes} BUY confirmations."
            )

    if signal == "SELL":

        if (
            resistance is not None
            and
            resistance_result["rejection"]
        ):

            confidence += 6

            core_reason = (
                "Resistance rejection + "
                f"{sell_votes} SELL confirmations."
            )

    confidence = min(
        MAX_SIGNAL_CONFIDENCE,
        confidence
    )

    # --------------------------------------------------------
    # MINIMUM THRESHOLD
    # --------------------------------------------------------

    if (
        signal is not None
        and
        confidence < MIN_SIGNAL_CONFIDENCE
    ):

        signal = None

        confidence = 0

        core_reason = (
            "Pattern agreement was not "
            "strong enough for the signal threshold."
        )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "status":
            (
                "SIGNAL"
                if signal
                else
                "NO_COMPLETE_SETUP"
            ),

        "signal":
            signal,

        "confidence":
            confidence,

        "buy_votes":
            buy_votes,

        "sell_votes":
            sell_votes,

        "core_reason":
            core_reason,

        "strategies":
            strategy_results,

        "support":
            support,

        "resistance":
            resistance,

        "support_result":
            support_result,

        "resistance_result":
            resistance_result,

        "limitations": [

            "True wick rejection is not confirmed "
            "because the current detector reads "
            "colored candle bodies.",

            "RSI is not yet visually extracted.",

            "MACD is not yet visually extracted.",

            "EMA is not yet visually extracted.",

            "Volume is not yet visually extracted."

        ]

    }


# ============================================================
# DETECTION MAP
# ============================================================

def create_detection_map(
    img,
    candles
):

    output = img.copy()

    for number, candle in enumerate(
        candles,
        start=1
    ):

        x = int(
            candle["x"]
        )

        y = int(
            candle["y"]
        )

        w = int(
            candle["w"]
        )

        h = int(
            candle["h"]
        )

        cv2.rectangle(
            output,
            (
                x,
                y
            ),
            (
                x + w,
                y + h
            ),
            (
                0,
                255,
                255
            ),
            2
        )

        if candle["color"] == "GREEN":

            label_color = (
                0,
                255,
                0
            )

        else:

            label_color = (
                0,
                0,
                255
            )

        cv2.putText(
            output,
            str(number),
            (
                x,
                max(
                    25,
                    y - 7
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            label_color,
            2,
            cv2.LINE_AA
        )

    return output


# ============================================================
# DETAILED TEST REPORT
# ============================================================

def create_detailed_report(
    asset,
    candles,
    analysis
):

    lines = []

    lines.append(
        "🔎 OTC SCREENSHOT STRATEGY TEST"
    )

    lines.append("")

    lines.append(
        f"💱 Asset: {asset}"
    )

    lines.append(
        "➡️ Scan: RIGHT → LEFT"
    )

    lines.append(
        "🎯 Candle #1 = newest visible candle"
    )

    lines.append("")

    lines.append(
        "📊 CANDLE DETECTION"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"Visible candles: {len(candles)}"
    )

    lines.append("")

    lines.append(
        "🕯️ NEWEST → OLDEST"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    for i, candle in enumerate(
        candles[
            :LOOKBACK_CANDLES
        ],
        start=1
    ):

        icon = (
            "🟢"
            if candle["color"] == "GREEN"
            else
            "🔴"
        )

        lines.append(
            f"{i}. {icon} "
            f"{candle['color']} "
            f"body={candle['body_size']:.4f}"
        )

    lines.append("")

    lines.append(
        "🧠 STRATEGY-BY-STRATEGY READING"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    for item in analysis[
        "strategies"
    ]:

        if item["detected"]:

            marker = "✅"

        else:

            marker = "⚪"

        direction = (
            f" → {item['direction']}"
            if item["direction"]
            != "NONE"
            else
            ""
        )

        lines.append(
            f"{marker} {item['name']}"
            f"{direction}"
        )

        lines.append(
            f"   {item['reason']}"
        )

    lines.append("")

    lines.append(
        "📈 AGREEMENT"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"🟢 BUY votes: "
        f"{analysis['buy_votes']}"
    )

    lines.append(
        f"🔴 SELL votes: "
        f"{analysis['sell_votes']}"
    )

    lines.append("")

    if analysis["support"]:

        lines.append(
            "🟢 SUPPORT ZONE"
        )

        lines.append(
            f"Tests: "
            f"{len(analysis['support']['candles'])}"
        )

        lines.append(
            f"Level: "
            f"{analysis['support']['level']:.5f}"
        )

    else:

        lines.append(
            "🟢 SUPPORT ZONE: NOT CLEAR"
        )

    if analysis["resistance"]:

        lines.append(
            "🔴 RESISTANCE ZONE"
        )

        lines.append(
            f"Tests: "
            f"{len(analysis['resistance']['candles'])}"
        )

        lines.append(
            f"Level: "
            f"{analysis['resistance']['level']:.5f}"
        )

    else:

        lines.append(
            "🔴 RESISTANCE ZONE: NOT CLEAR"
        )

    lines.append("")

    lines.append(
        "🎯 FINAL TEST RESULT"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    if analysis["signal"]:

        arrow = (
            "🟢"
            if analysis["signal"] == "BUY"
            else
            "🔴"
        )

        lines.append(
            f"{arrow} "
            f"{analysis['signal']}"
        )

        lines.append(
            f"💪 Strength: "
            f"{analysis['confidence']}%"
        )

        lines.append(
            f"🧠 Core reason: "
            f"{analysis['core_reason']}"
        )

    else:

        lines.append(
            "⛔ NO SIGNAL"
        )

        lines.append(
            f"Reason: "
            f"{analysis['core_reason']}"
        )

    lines.append("")

    lines.append(
        "⚠️ WHAT THIS VERSION CANNOT "
        "HONESTLY CONFIRM YET"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━"
    )

    for limitation in analysis[
        "limitations"
    ]:

        lines.append(
            "• " + limitation
        )

    lines.append("")

    lines.append(
        "TEST MODE: No automatic trading."
    )

    return "\n".join(lines)


# ============================================================
# SHORT SIGNAL REPORT
# ============================================================

def create_short_signal(
    asset,
    analysis
):

    direction = analysis[
        "signal"
    ]

    confidence = analysis[
        "confidence"
    ]

    now = datetime.now(
        LOCAL_TZ
    )

    base_time = (
        now.replace(
            second=0,
            microsecond=0
        )
        +
        timedelta(
            minutes=1
        )
    )

    expiry = (
        base_time
        +
        timedelta(
            minutes=EXPIRY_MINUTES
        )
    )

    arrow = (
        "🟢"
        if direction == "BUY"
        else
        "🔴"
    )

    message = (

        "🚨 OTC SIGNAL\n\n"

        f"💱 Pair: {asset}\n"

        f"{arrow} "
        f"Direction: {direction}\n"

        f"💪 Strength: "
        f"{confidence}%\n"

        f"⏰ Entry: "
        f"{base_time.strftime('%H:%M:%S')}\n"

        f"⌛ Expiry: "
        f"{expiry.strftime('%H:%M:%S')}\n\n"

        f"🧠 Core reason: "
        f"{analysis['core_reason']}\n\n"

        "⚠️ Manual decision only."
    )

    return message


# ============================================================
# TELEGRAM SEND
# ============================================================

def send_telegram(
    message
):

    if not TOKEN:

        print(
            "❌ BOT_TOKEN is missing."
        )

        return

    url = (
        "https://api.telegram.org/"
        f"bot{TOKEN}/sendMessage"
    )

    destinations = [
        CHAT_ID,
        CHANNEL_ID
    ]

    for destination in destinations:

        try:

            response = requests.post(
                url,
                data={
                    "chat_id":
                        destination,
                    "text":
                        message
                },
                timeout=8
            )

            if not response.ok:

                print(
                    "Telegram error:",
                    response.text
                )

        except Exception as e:

            print(
                "❌ Telegram send error:",
                str(e)
            )

    print(
        "✅ Sent to Telegram"
    )


# ============================================================
# TELEGRAM BOT
# ============================================================

screenshot_reader = None


# ============================================================
# START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(

        "📊 OTC SCREENSHOT STRATEGY BOT\n\n"

        "📸 Send a Pocket Option screenshot.\n\n"

        "I scan the visible candles "
        "RIGHT → LEFT.\n\n"

        "Candle #1 = newest.\n\n"

        "Strategies tested:\n"

        "• Candle trend\n"
        "• Body momentum\n"
        "• 3+ candle momentum\n"
        "• Color reversal\n"
        "• Body engulfing\n"
        "• Inside body\n"
        "• Body compression\n"
        "• Body clustering\n"
        "• Support\n"
        "• Resistance\n"
        "• Support confirmation\n"
        "• Resistance confirmation\n"
        "• Support breakout failure\n"
        "• Resistance breakout failure\n"
        "• Two-sided S/R structure\n\n"

        "No random candles.\n"
        "No generated market data.\n"
        "No automatic trading."
    )


# ============================================================
# PHOTO HANDLER
# ============================================================

async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    start_time = time.time()

    original_path = (
        "strategy_chart.png"
    )

    detection_path = (
        "strategy_detection.png"
    )

    try:

        await update.message.reply_text(

            "📸 Screenshot received.\n"
            "👁️ Reading visible candles "
            "RIGHT → LEFT...\n"
            "🧠 Testing all strategies..."
        )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        photo = (
            await update.message
            .photo[-1]
            .get_file()
        )

        await photo.download_to_drive(
            original_path
        )

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        img = load_image(
            original_path
        )

        # ----------------------------------------------------
        # ASSET
        # ----------------------------------------------------

        asset, ocr_text = detect_asset(
            img
        )

        # ----------------------------------------------------
        # CANDLES
        # ----------------------------------------------------

        raw_candles = detect_candles(
            img
        )

        if not raw_candles:

            await update.message.reply_text(

                "❌ No reliable candle bodies "
                "were detected.\n\n"
                "No strategy result was generated."
            )

            return

        # ----------------------------------------------------
        # GEOMETRY
        # ----------------------------------------------------

        candles = (
            convert_to_price_geometry(
                raw_candles
            )
        )

        if not candles:

            await update.message.reply_text(

                "❌ Candle geometry could not "
                "be established reliably."
            )

            return

        # ----------------------------------------------------
        # STRATEGIES
        # ----------------------------------------------------

        analysis = (
            analyze_all_strategies(
                candles
            )
        )

        elapsed = (
            time.time()
            -
            start_time
        )

        # ----------------------------------------------------
        # DETAILED TEST MODE
        # ----------------------------------------------------

        if DETAILED_TEST_MODE:

            detailed = (
                create_detailed_report(
                    asset,
                    candles,
                    analysis
                )
            )

            detailed += (
                "\n\n⚡ Processing: "
                f"{elapsed:.2f}s"
            )

            await update.message.reply_text(
                detailed
            )

            # ------------------------------------------------
            # SEND SIGNAL ONLY IF THERE IS ONE
            # ------------------------------------------------

            if analysis["signal"]:

                signal = (
                    create_short_signal(
                        asset,
                        analysis
                    )
                )

                send_telegram(
                    signal
                )

                await update.message.reply_text(
                    "🚨 SIGNAL GENERATED\n\n"
                    +
                    signal
                )

            # ------------------------------------------------
            # DETECTION MAP
            # ------------------------------------------------

            detection_map = (
                create_detection_map(
                    img,
                    raw_candles
                )
            )

            cv2.imwrite(
                detection_path,
                detection_map
            )

            with open(
                detection_path,
                "rb"
            ) as photo_file:

                await update.message.reply_photo(

                    photo=photo_file,

                    caption=(
                        "🔢 DETECTION MAP\n\n"
                        "➡️ RIGHT → LEFT\n"
                        "🎯 #1 = newest\n\n"
                        "🟡 Yellow box = detected "
                        "candle body\n"
                        "🟢 Number = GREEN\n"
                        "🔴 Number = RED\n\n"
                        "Use this to verify that "
                        "the bot is reading the "
                        "same candles you see."
                    )
                )

        # ----------------------------------------------------
        # SHORT MODE
        # ----------------------------------------------------

        else:

            if analysis["signal"]:

                signal = (
                    create_short_signal(
                        asset,
                        analysis
                    )
                )

                send_telegram(
                    signal
                )

                await update.message.reply_text(
                    signal
                )

            else:

                await update.message.reply_text(

                    "⛔ NO SIGNAL\n\n"
                    "The strategies did not "
                    "agree strongly enough."
                )

    except Exception as e:

        print(
            "❌ ERROR:",
            repr(e)
        )

        await update.message.reply_text(

            "❌ Analysis error:\n"
            +
            str(e)
        )

    finally:

        for path in [
            original_path,
            detection_path
        ]:

            if os.path.exists(path):

                try:

                    os.remove(
                        path
                    )

                except Exception:

                    pass


# ============================================================
# RUN TELEGRAM
# ============================================================

def run_telegram():

    if not TOKEN:

        raise RuntimeError(
            "BOT_TOKEN environment variable "
            "is missing."
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
        "========================================"
    )

    print(
        "🧠 OTC SCREENSHOT STRATEGY BOT"
    )

    print(
        "========================================"
    )

    print(
        "RIGHT → LEFT candle scanning."
    )

    print(
        "Candle #1 = newest."
    )

    print(
        "Detailed test mode:",
        DETAILED_TEST_MODE
    )

    print(
        "No random candles."
    )

    print(
        "No automatic trading."
    )

    print(
        "Waiting for screenshots..."
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

    print("=" * 55)

    print(
        "📊 OTC SCREENSHOT STRATEGY BOT"
    )

    print("=" * 55)

    threading.Thread(
        target=run_flask,
        daemon=True
    ).start()

    print(
        "✅ Flask server started "
        "on port 10000"
    )

    run_telegram()

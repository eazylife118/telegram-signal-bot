import os
import cv2
import numpy as np
import time
import asyncio
import threading
from flask import Flask
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ============================================================
# TELEGRAM SETTINGS
# ============================================================

TELEGRAM_TOKEN = os.getenv(
    "BOT_TOKEN",
    "PASTE_YOUR_BOT_TOKEN_HERE"
)

CHANNEL_ID = os.getenv(
    "CHANNEL_ID",
    "-1004324805205"
)

# ============================================================
# FLASK KEEP-ALIVE
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "OTC Candle Strategy Bot is running."


@app.route("/ping")
def ping():
    return "OK"


def run_flask():
    port = int(os.getenv("PORT", "10000"))

    app.run(
        host="0.0.0.0",
        port=port,
        use_reloader=False
    )


# ============================================================
# TIMEZONE
# ============================================================

NIGERIA_TZ = ZoneInfo("Africa/Lagos")


def nigeria_now():
    return datetime.now(NIGERIA_TZ)


def signal_and_entry_times():
    now = nigeria_now()

    signal_time = now.replace(
        second=0,
        microsecond=0
    )

    entry_time = signal_time + timedelta(
        minutes=1
    )

    return (
        signal_time.strftime("%H:%M"),
        entry_time.strftime("%H:%M")
    )


# ============================================================
# DETECTION SETTINGS
# ============================================================

MIN_BODY_AREA = 10
MIN_BODY_HEIGHT = 2
MIN_CANDLE_WIDTH = 2

RIGHT_MIN_BODY_AREA = 6
RIGHT_MIN_BODY_HEIGHT = 2

MAX_CANDLE_WIDTH_RATIO = 0.045
MERGE_DISTANCE_RATIO = 0.55

LOOKBACK_CANDLES = 20
MIN_ZONE_TESTS = 2
ZONE_TOLERANCE_RATIO = 0.035

MIN_SIGNAL_AGREEMENT = 65

SMALL_BODY_RATIO = 0.35

# ============================================================
# NEW PRICE-ACTION LAYER SETTINGS
# ============================================================

STRUCTURE_LOOKBACK = 12
TREND_LOOKBACK = 10
SEQUENCE_LOOKBACK = 8
CHOP_LOOKBACK = 8

MIN_SWING_DISTANCE = 2
SWING_EQUAL_TOLERANCE = 0.015

TREND_STRENGTH_THRESHOLD = 0.55
CHOP_ALTERNATION_THRESHOLD = 0.65

# ============================================================
# IMAGE LOADING
# ============================================================


def load_image(path):

    img = cv2.imread(path)

    if img is None:
        raise ValueError(
            "Could not read screenshot."
        )

    height, width = img.shape[:2]

    if width < 1400:

        scale = 1400 / width

        img = cv2.resize(
            img,
            (
                int(width * scale),
                int(height * scale)
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
        90,
        255,
        255
    ])

    green = cv2.inRange(
        hsv,
        green_lower,
        green_upper
    )

    # RED RANGE 1

    red_lower_1 = np.array([
        0,
        55,
        55
    ])

    red_upper_1 = np.array([
        12,
        255,
        255
    ])

    red1 = cv2.inRange(
        hsv,
        red_lower_1,
        red_upper_1
    )

    # RED RANGE 2

    red_lower_2 = np.array([
        168,
        55,
        55
    ])

    red_upper_2 = np.array([
        180,
        255,
        255
    ])

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

        x, y, width, height = cv2.boundingRect(
            contour
        )

        if width < MIN_CANDLE_WIDTH:
            continue

        if height < min_height:
            continue

        if width > max_width:
            continue

        if width > height * 6:
            continue

        region = cleaned[
            y:y + height,
            x:x + width
        ]

        colored_pixels = int(
            np.sum(region > 0)
        )

        if colored_pixels < 5:
            continue

        density = (
            colored_pixels /
            float(max(1, width * height))
        )

        if density < 0.15:
            continue

        center_x = (
            x +
            width / 2
        )

        candidates.append({
            "x": x,
            "y": y,
            "w": width,
            "h": height,
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


def merge_candidates(candidates):

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda c: c["center_x"]
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

            candidate_top = candidate["y"]

            candidate_bottom = (
                candidate["y"] +
                candidate["h"]
            )

            existing_top = existing["y"]

            existing_bottom = (
                existing["y"] +
                existing["h"]
            )

            vertical_overlap = not (
                candidate_bottom < existing_top
                or
                candidate_top > existing_bottom
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
                    existing["x"] + existing["w"],
                    candidate["x"] + candidate["w"]
                )

                top = min(
                    existing["y"],
                    candidate["y"]
                )

                bottom = max(
                    existing["y"] + existing["h"],
                    candidate["y"] + candidate["h"]
                )

                existing["x"] = left
                existing["y"] = top
                existing["w"] = right - left
                existing["h"] = bottom - top

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


def remove_cross_color_duplicates(candles):

    candles = sorted(
        candles,
        key=lambda c: c["center_x"]
    )

    result = []

    for candle in candles:

        duplicate_index = None

        for index, existing in enumerate(result):

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

                duplicate_index = index

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

    height, width = chart.shape[:2]

    right_start = int(
        width * 0.72
    )

    green_right = green_mask[
        :,
        right_start:
    ]

    red_right = red_mask[
        :,
        right_start:
    ]

    green = find_candidates(
        green_right,
        "GREEN",
        width,
        right_side=True
    )

    red = find_candidates(
        red_right,
        "RED",
        width,
        right_side=True
    )

    for candle in green + red:

        candle["x"] += right_start

        candle["center_x"] += (
            right_start
        )

    return green + red


# ============================================================
# DETECT CANDLES
# ============================================================


def detect_candles(img):

    height, width = img.shape[:2]

    green_mask, red_mask = (
        get_color_masks(img)
    )

    green = find_candidates(
        green_mask,
        "GREEN",
        width,
        right_side=False
    )

    red = find_candidates(
        red_mask,
        "RED",
        width,
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

    right_candidates = detect_right_side(
        img,
        green_mask,
        red_mask
    )

    candles.extend(
        right_candidates
    )

    candles = remove_cross_color_duplicates(
        candles
    )

    candles.sort(
        key=lambda c: c["center_x"]
    )

    candles.reverse()

    return candles


# ============================================================
# BODY POSITION
# ============================================================


def body_position(candle):

    top = candle["y"]

    bottom = (
        candle["y"] +
        candle["h"]
    )

    return (
        top,
        bottom
    )


# ============================================================
# PRICE-ACTION HELPERS
# ============================================================


def candle_midpoint(candle):

    top, bottom = body_position(
        candle
    )

    return (
        top +
        bottom
    ) / 2


def candle_range(candle):

    return max(
        1,
        candle["h"]
    )


def directional_value(candle):

    midpoint = candle_midpoint(
        candle
    )

    return midpoint


def price_higher(
    candle_a,
    candle_b
):

    return candle_midpoint(
        candle_a
    ) < candle_midpoint(
        candle_b
    )


def price_lower(
    candle_a,
    candle_b
):

    return candle_midpoint(
        candle_a
    ) > candle_midpoint(
        candle_b
    )


def body_strength(candle):

    return min(
        1.0,
        candle["h"] /
        max(
            1.0,
            candle["w"] * 3.0
        )
    )


# ============================================================
# RESISTANCE
# ============================================================


def find_resistance_zone(candles):

    if not candles:
        return None

    candles_to_check = candles[
        :LOOKBACK_CANDLES
    ]

    if len(candles_to_check) < 2:
        return None

    highest_points = []

    for candle in candles_to_check:

        top, bottom = body_position(
            candle
        )

        highest_points.append(
            top
        )

    chart_min = min(
        highest_points
    )

    chart_max = max(
        highest_points
    )

    visible_range = max(
        1,
        chart_max - chart_min
    )

    tolerance = (
        visible_range *
        ZONE_TOLERANCE_RATIO
    )

    zones = []

    for candle in candles_to_check:

        top, bottom = body_position(
            candle
        )

        placed = False

        for zone in zones:

            if abs(
                top -
                zone["level"]
            ) <= tolerance:

                zone["candles"].append(
                    candle
                )

                zone["level"] = np.mean(
                    [
                        body_position(c)[0]
                        for c in zone["candles"]
                    ]
                )

                placed = True

                break

        if not placed:

            zones.append({
                "level": float(top),
                "candles": [candle]
            })

    zones.sort(
        key=lambda z: len(
            z["candles"]
        ),
        reverse=True
    )

    if not zones:
        return None

    strongest = zones[0]

    if len(
        strongest["candles"]
    ) < MIN_ZONE_TESTS:

        return None

    return strongest


# ============================================================
# SUPPORT
# ============================================================


def find_support_zone(candles):

    if not candles:
        return None

    candles_to_check = candles[
        :LOOKBACK_CANDLES
    ]

    if len(candles_to_check) < 2:
        return None

    lowest_points = []

    for candle in candles_to_check:

        top, bottom = body_position(
            candle
        )

        lowest_points.append(
            bottom
        )

    chart_min = min(
        lowest_points
    )

    chart_max = max(
        lowest_points
    )

    visible_range = max(
        1,
        chart_max - chart_min
    )

    tolerance = (
        visible_range *
        ZONE_TOLERANCE_RATIO
    )

    zones = []

    for candle in candles_to_check:

        top, bottom = body_position(
            candle
        )

        placed = False

        for zone in zones:

            if abs(
                bottom -
                zone["level"]
            ) <= tolerance:

                zone["candles"].append(
                    candle
                )

                zone["level"] = np.mean(
                    [
                        body_position(c)[1]
                        for c in zone["candles"]
                    ]
                )

                placed = True

                break

        if not placed:

            zones.append({
                "level": float(bottom),
                "candles": [candle]
            })

    zones.sort(
        key=lambda z: len(
            z["candles"]
        ),
        reverse=True
    )

    if not zones:
        return None

    strongest = zones[0]

    if len(
        strongest["candles"]
    ) < MIN_ZONE_TESTS:

        return None

    return strongest


# ============================================================
# RESISTANCE REACTION
# ============================================================


def resistance_behavior(
    candles,
    resistance
):

    if not resistance:

        return {
            "near": False,
            "rejection": False
        }

    newest = candles[0]

    top, bottom = body_position(
        newest
    )

    level = resistance["level"]

    distance = abs(
        top -
        level
    )

    tolerance = max(
        3,
        newest["h"] * 0.75
    )

    near = (
        distance <= tolerance
    )

    rejection = (
        near
        and
        newest["color"] == "RED"
    )

    return {
        "near": near,
        "rejection": rejection
    }


# ============================================================
# SUPPORT REACTION
# ============================================================


def support_behavior(
    candles,
    support
):

    if not support:

        return {
            "near": False,
            "rejection": False
        }

    newest = candles[0]

    top, bottom = body_position(
        newest
    )

    level = support["level"]

    distance = abs(
        bottom -
        level
    )

    tolerance = max(
        3,
        newest["h"] * 0.75
    )

    near = (
        distance <= tolerance
    )

    rejection = (
        near
        and
        newest["color"] == "GREEN"
    )

    return {
        "near": near,
        "rejection": rejection
    }


# ============================================================
# REACTION STRENGTH
# ============================================================


def reaction_strength(
    zone,
    candles,
    resistance=True
):

    if not zone:
        return 0

    tests = len(
        zone["candles"]
    )

    if tests >= 8:
        return 3

    if tests >= 5:
        return 2

    if tests >= 2:
        return 1

    return 0


# ============================================================
# BODY MOMENTUM
# ============================================================


def analyze_body_momentum(candles):

    if len(candles) < 3:
        return "UNKNOWN"

    newest = candles[0]["h"]

    previous = candles[1]["h"]

    older = candles[2]["h"]

    average_previous = (
        previous +
        older
    ) / 2

    if newest > average_previous * 1.35:
        return "STRONGER"

    if newest < average_previous * 0.70:
        return "WEAKER"

    return "NORMAL"


# ============================================================
# CONSECUTIVE MOMENTUM
# ============================================================


def consecutive_momentum(candles):

    if len(candles) < 3:
        return None

    color = candles[0]["color"]

    count = 0

    for candle in candles:

        if candle["color"] == color:

            count += 1

        else:

            break

    if count >= 3:

        if color == "GREEN":
            return "BUY"

        return "SELL"

    return None


# ============================================================
# BODY REVERSAL
# ============================================================


def body_reversal(candles):

    if len(candles) < 2:
        return None

    current = candles[0]

    previous = candles[1]

    if current["color"] == previous["color"]:
        return None

    if current["h"] >= previous["h"] * 1.20:

        if current["color"] == "GREEN":
            return "BUY"

        return "SELL"

    return None


# ============================================================
# TWO-CANDLE REVERSAL
# ============================================================


def two_candle_reversal(candles):

    if len(candles) < 2:
        return None

    a = candles[0]
    b = candles[1]

    if (
        a["color"] == "GREEN"
        and
        b["color"] == "RED"
        and
        a["h"] >= b["h"] * 0.80
    ):

        return "BUY"

    if (
        a["color"] == "RED"
        and
        b["color"] == "GREEN"
        and
        a["h"] >= b["h"] * 0.80
    ):

        return "SELL"

    return None


# ============================================================
# THREE-CANDLE REVERSAL
# ============================================================


def three_candle_reversal(candles):

    if len(candles) < 3:
        return None

    a = candles[0]["color"]
    b = candles[1]["color"]
    c = candles[2]["color"]

    if (
        a == "GREEN"
        and
        b == "GREEN"
        and
        c == "RED"
    ):

        return "BUY"

    if (
        a == "RED"
        and
        b == "RED"
        and
        c == "GREEN"
    ):

        return "SELL"

    return None


# ============================================================
# PULLBACK
# ============================================================


def detect_pullback(candles):

    if len(candles) < 4:
        return None

    a = candles[0]["color"]
    b = candles[1]["color"]
    c = candles[2]["color"]
    d = candles[3]["color"]

    if (
        a == b
        and
        b == c
        and
        d != c
    ):

        if a == "GREEN":
            return "BUY"

        return "SELL"

    return None


# ============================================================
# BREAKOUT FAILURE
# ============================================================


def breakout_failure(
    candles,
    resistance,
    support
):

    resistance_failure = False
    support_failure = False

    if resistance:

        level = resistance["level"]

        for i in range(
            1,
            min(
                len(candles),
                LOOKBACK_CANDLES
            )
        ):

            newer = candles[i - 1]
            older = candles[i]

            newer_top = body_position(
                newer
            )[0]

            older_top = body_position(
                older
            )[0]

            if (
                older_top <= level
                and
                newer_top > level
                and
                newer["color"] == "RED"
            ):

                resistance_failure = True

                break

    if support:

        level = support["level"]

        for i in range(
            1,
            min(
                len(candles),
                LOOKBACK_CANDLES
            )
        ):

            newer = candles[i - 1]
            older = candles[i]

            newer_bottom = body_position(
                newer
            )[1]

            older_bottom = body_position(
                older
            )[1]

            if (
                older_bottom >= level
                and
                newer_bottom < level
                and
                newer["color"] == "GREEN"
            ):

                support_failure = True

                break

    return (
        resistance_failure,
        support_failure
    )


# ============================================================
# RESISTANCE BREAKOUT HOLD
# ============================================================


def resistance_breakout_hold(
    candles,
    resistance
):

    if not resistance:
        return False

    level = resistance["level"]

    maximum = min(
        len(candles) - 1,
        LOOKBACK_CANDLES - 1
    )

    for i in range(maximum):

        newer = candles[i]
        older = candles[i + 1]

        older_top, older_bottom = (
            body_position(older)
        )

        newer_top, newer_bottom = (
            body_position(newer)
        )

        older_crosses = (
            older_top <= level
            and
            older_bottom >= level
        )

        older_above = (
            older_bottom < level
        )

        older_break = (
            older_crosses
            or
            older_above
        )

        newer_remains_above = (
            newer_bottom < level
        )

        if (
            older["color"] == "GREEN"
            and
            newer["color"] == "RED"
            and
            older_break
            and
            newer_remains_above
        ):

            return True

    return False


# ============================================================
# SUPPORT BREAKDOWN HOLD
# ============================================================


def support_breakdown_hold(
    candles,
    support
):

    if not support:
        return False

    level = support["level"]

    maximum = min(
        len(candles) - 1,
        LOOKBACK_CANDLES - 1
    )

    for i in range(maximum):

        newer = candles[i]
        older = candles[i + 1]

        older_top, older_bottom = (
            body_position(older)
        )

        newer_top, newer_bottom = (
            body_position(newer)
        )

        older_crosses = (
            older_top <= level
            and
            older_bottom >= level
        )

        older_below = (
            older_top > level
        )

        older_break = (
            older_crosses
            or
            older_below
        )

        newer_remains_below = (
            newer_top > level
        )

        if (
            older["color"] == "RED"
            and
            newer["color"] == "GREEN"
            and
            older_break
            and
            newer_remains_below
        ):

            return True

    return False


# ============================================================
# COLOR CONFIRMATION
# ============================================================


def color_confirmation(
    candles,
    resistance,
    support
):

    green_red = False
    red_green = False

    if resistance:

        level = resistance["level"]

        for i in range(
            1,
            min(
                len(candles),
                LOOKBACK_CANDLES
            )
        ):

            newer = candles[i - 1]
            older = candles[i]

            older_top = body_position(
                older
            )[0]

            newer_top = body_position(
                newer
            )[0]

            if (
                older["color"] == "GREEN"
                and
                newer["color"] == "RED"
                and
                older_top <= level
                and
                newer_top >= level
            ):

                green_red = True

                break

    if support:

        level = support["level"]

        for i in range(
            1,
            min(
                len(candles),
                LOOKBACK_CANDLES
            )
        ):

            newer = candles[i - 1]
            older = candles[i]

            older_bottom = body_position(
                older
            )[1]

            newer_bottom = body_position(
                newer
            )[1]

            if (
                older["color"] == "RED"
                and
                newer["color"] == "GREEN"
                and
                older_bottom >= level
                and
                newer_bottom <= level
            ):

                red_green = True

                break

    return (
        green_red,
        red_green
    )


# ============================================================
# CONTINUATION
# ============================================================


def continuation_structure(
    candles,
    resistance,
    support
):

    if len(candles) < 3:
        return None

    colors = [
        c["color"]
        for c in candles[:3]
    ]

    if colors == [
        "GREEN",
        "GREEN",
        "GREEN"
    ]:

        if resistance:

            level = resistance["level"]

            above = True

            for candle in candles[:3]:

                if body_position(candle)[1] >= level:

                    above = False

                    break

            if above:
                return "BUY"

        return "BUY"

    if colors == [
        "RED",
        "RED",
        "RED"
    ]:

        if support:

            level = support["level"]

            below = True

            for candle in candles[:3]:

                if body_position(candle)[0] <= level:

                    below = False

                    break

            if below:
                return "SELL"

        return "SELL"

    return None


# ============================================================
# NEW LAYER 1
# HIGHER HIGH / HIGHER LOW
# ============================================================


def higher_high_higher_low(candles):

    if len(candles) < 5:
        return None

    recent = candles[
        :min(
            len(candles),
            STRUCTURE_LOOKBACK
        )
    ]

    bullish_moves = 0

    for i in range(
        len(recent) - 2
    ):

        newest = recent[i]
        older = recent[i + 1]
        older2 = recent[i + 2]

        newest_mid = candle_midpoint(
            newest
        )

        older_mid = candle_midpoint(
            older
        )

        older2_mid = candle_midpoint(
            older2
        )

        # Smaller screen Y = higher price.

        if (
            newest_mid < older_mid
            and
            older_mid < older2_mid
        ):

            bullish_moves += 1

    if bullish_moves >= 2:
        return "BUY"

    return None


# ============================================================
# NEW LAYER 2
# LOWER HIGH / LOWER LOW
# ============================================================


def lower_high_lower_low(candles):

    if len(candles) < 5:
        return None

    recent = candles[
        :min(
            len(candles),
            STRUCTURE_LOOKBACK
        )
    ]

    bearish_moves = 0

    for i in range(
        len(recent) - 2
    ):

        newest = recent[i]
        older = recent[i + 1]
        older2 = recent[i + 2]

        newest_mid = candle_midpoint(
            newest
        )

        older_mid = candle_midpoint(
            older
        )

        older2_mid = candle_midpoint(
            older2
        )

        if (
            newest_mid > older_mid
            and
            older_mid > older2_mid
        ):

            bearish_moves += 1

    if bearish_moves >= 2:
        return "SELL"

    return None


# ============================================================
# NEW LAYER 3
# OVERALL TREND DIRECTION
# ============================================================


def overall_trend_direction(candles):

    if len(candles) < 6:
        return "SIDEWAYS"

    sample = candles[
        :min(
            len(candles),
            TREND_LOOKBACK
        )
    ]

    bullish = 0
    bearish = 0

    for i in range(
        len(sample) - 1
    ):

        current = candle_midpoint(
            sample[i]
        )

        previous = candle_midpoint(
            sample[i + 1]
        )

        if current < previous:
            bullish += 1

        elif current > previous:
            bearish += 1

    total = bullish + bearish

    if total == 0:
        return "SIDEWAYS"

    bullish_ratio = (
        bullish /
        total
    )

    bearish_ratio = (
        bearish /
        total
    )

    if bullish_ratio >= 0.60:
        return "BULLISH"

    if bearish_ratio >= 0.60:
        return "BEARISH"

    return "SIDEWAYS"


# ============================================================
# NEW LAYER 4
# TREND STRENGTH
# ============================================================


def trend_strength(candles):

    if len(candles) < 6:
        return 0.0

    sample = candles[
        :min(
            len(candles),
            TREND_LOOKBACK
        )
    ]

    directional = 0
    total = 0

    for i in range(
        len(sample) - 1
    ):

        current = candle_midpoint(
            sample[i]
        )

        previous = candle_midpoint(
            sample[i + 1]
        )

        total += 1

        if current != previous:
            directional += 1

    if total == 0:
        return 0.0

    consistency = (
        directional /
        total
    )

    return round(
        consistency * 100,
        1
    )


# ============================================================
# NEW LAYER 5
# PULLBACK QUALITY
# ============================================================


def pullback_quality(candles):

    if len(candles) < 5:
        return None

    colors = [
        c["color"]
        for c in candles[:5]
    ]

    first_three = colors[2:5]

    recent = colors[:2]

    bullish_base = (
        first_three.count("GREEN") >= 2
    )

    bearish_base = (
        first_three.count("RED") >= 2
    )

    if (
        bullish_base
        and
        recent.count("RED") >= 1
    ):

        # Small counter-move followed by
        # bullish pressure.

        if (
            candles[0]["h"]
            <=
            candles[2]["h"] * 1.35
        ):

            return "BUY"

    if (
        bearish_base
        and
        recent.count("GREEN") >= 1
    ):

        if (
            candles[0]["h"]
            <=
            candles[2]["h"] * 1.35
        ):

            return "SELL"

    return None


# ============================================================
# NEW LAYER 6
# BREAKOUT + RETEST
# ============================================================


def breakout_retest(
    candles,
    resistance,
    support
):

    if len(candles) < 4:
        return None

    # Bullish breakout and retest.

    if resistance:

        level = resistance["level"]

        broke = False

        for candle in candles[2:5]:

            top, bottom = body_position(
                candle
            )

            if (
                candle["color"] == "GREEN"
                and
                bottom < level
            ):

                broke = True

                break

        if broke:

            for candle in candles[:2]:

                top, bottom = body_position(
                    candle
                )

                near_level = (
                    abs(
                        bottom -
                        level
                    )
                    <=
                    max(
                        5,
                        candle["h"] * 1.5
                    )
                )

                if (
                    near_level
                    and
                    candle["color"] == "GREEN"
                ):

                    return "BUY"

    # Bearish breakout and retest.

    if support:

        level = support["level"]

        broke = False

        for candle in candles[2:5]:

            top, bottom = body_position(
                candle
            )

            if (
                candle["color"] == "RED"
                and
                top > level
            ):

                broke = True

                break

        if broke:

            for candle in candles[:2]:

                top, bottom = body_position(
                    candle
                )

                near_level = (
                    abs(
                        top -
                        level
                    )
                    <=
                    max(
                        5,
                        candle["h"] * 1.5
                    )
                )

                if (
                    near_level
                    and
                    candle["color"] == "RED"
                ):

                    return "SELL"

    return None


# ============================================================
# NEW LAYER 7
# SWING REJECTION
# ============================================================


def swing_rejection(
    candles,
    resistance,
    support
):

    if not candles:
        return None

    newest = candles[0]

    if resistance:

        level = resistance["level"]

        top, bottom = body_position(
            newest
        )

        if (
            abs(
                top -
                level
            )
            <=
            max(
                6,
                newest["h"] * 1.5
            )
            and
            newest["color"] == "RED"
        ):

            return "SELL"

    if support:

        level = support["level"]

        top, bottom = body_position(
            newest
        )

        if (
            abs(
                bottom -
                level
            )
            <=
            max(
                6,
                newest["h"] * 1.5
            )
            and
            newest["color"] == "GREEN"
        ):

            return "BUY"

    return None


# ============================================================
# NEW LAYER 8
# TREND EXHAUSTION
# ============================================================


def trend_exhaustion(candles):

    if len(candles) < 5:
        return None

    colors = [
        c["color"]
        for c in candles[:5]
    ]

    if colors.count("GREEN") >= 4:

        newest = candles[0]
        previous = candles[1]

        if (
            newest["h"]
            <
            previous["h"]
        ):

            return "SELL"

    if colors.count("RED") >= 4:

        newest = candles[0]
        previous = candles[1]

        if (
            newest["h"]
            <
            previous["h"]
        ):

            return "BUY"

    return None


# ============================================================
# NEW LAYER 9
# CANDLE SEQUENCE QUALITY
# ============================================================


def candle_sequence_quality(candles):

    if len(candles) < 5:
        return None

    sample = candles[
        :min(
            len(candles),
            SEQUENCE_LOOKBACK
        )
    ]

    green_count = sum(
        1
        for c in sample
        if c["color"] == "GREEN"
    )

    red_count = sum(
        1
        for c in sample
        if c["color"] == "RED"
    )

    if green_count >= 6:

        recent_heights = [
            c["h"]
            for c in sample[:4]
        ]

        if (
            recent_heights[0]
            >=
            recent_heights[-1] * 0.80
        ):

            return "BUY"

    if red_count >= 6:

        recent_heights = [
            c["h"]
            for c in sample[:4]
        ]

        if (
            recent_heights[0]
            >=
            recent_heights[-1] * 0.80
        ):

            return "SELL"

    return None


# ============================================================
# NEW LAYER 10
# IMPULSE VS CORRECTION
# ============================================================


def impulse_vs_correction(candles):

    if len(candles) < 5:
        return None

    main = candles[2]

    recent = candles[:2]

    recent_counter = 0

    for candle in recent:

        if (
            candle["color"]
            !=
            main["color"]
        ):

            recent_counter += 1

    if recent_counter == 0:
        return None

    counter_size = sum(
        c["h"]
        for c in recent
        if c["color"] != main["color"]
    )

    impulse_size = sum(
        c["h"]
        for c in candles[2:5]
        if c["color"] == main["color"]
    )

    if impulse_size <= 0:
        return None

    ratio = (
        counter_size /
        impulse_size
    )

    if (
        ratio <= 0.65
        and
        main["color"] == "GREEN"
    ):

        return "BUY"

    if (
        ratio <= 0.65
        and
        main["color"] == "RED"
    ):

        return "SELL"

    return None


# ============================================================
# NEW LAYER 11
# STRUCTURE-BREAK CONFIRMATION
# ============================================================


def structure_break_confirmation(candles):

    if len(candles) < 5:
        return None

    recent = candles[:4]

    newest = recent[0]

    previous_high = min(
        candle_midpoint(c)
        for c in recent[1:]
    )

    previous_low = max(
        candle_midpoint(c)
        for c in recent[1:]
    )

    current_price = candle_midpoint(
        newest
    )

    if (
        newest["color"] == "GREEN"
        and
        current_price < previous_high
    ):

        return "BUY"

    if (
        newest["color"] == "RED"
        and
        current_price > previous_low
    ):

        return "SELL"

    return None


# ============================================================
# NEW LAYER 12
# FAKE BREAKOUT DETECTION
# ============================================================


def fake_breakout_detection(
    candles,
    resistance,
    support
):

    if len(candles) < 3:
        return None

    newest = candles[0]
    previous = candles[1]

    if resistance:

        level = resistance["level"]

        previous_top, previous_bottom = (
            body_position(previous)
        )

        newest_top, newest_bottom = (
            body_position(newest)
        )

        previous_above = (
            previous_bottom < level
        )

        newest_back_inside = (
            newest_top >= level
        )

        if (
            previous_above
            and
            newest_back_inside
            and
            newest["color"] == "RED"
        ):

            return "SELL"

    if support:

        level = support["level"]

        previous_top, previous_bottom = (
            body_position(previous)
        )

        newest_top, newest_bottom = (
            body_position(newest)
        )

        previous_below = (
            previous_top > level
        )

        newest_back_inside = (
            newest_bottom <= level
        )

        if (
            previous_below
            and
            newest_back_inside
            and
            newest["color"] == "GREEN"
        ):

            return "BUY"

    return None


# ============================================================
# NEW LAYER 13
# TREND TRANSITION DETECTION
# ============================================================


def trend_transition_detection(candles):

    if len(candles) < 7:
        return None

    older = candles[4:7]
    newer = candles[:3]

    older_green = sum(
        1
        for c in older
        if c["color"] == "GREEN"
    )

    older_red = sum(
        1
        for c in older
        if c["color"] == "RED"
    )

    newer_green = sum(
        1
        for c in newer
        if c["color"] == "GREEN"
    )

    newer_red = sum(
        1
        for c in newer
        if c["color"] == "RED"
    )

    if (
        older_red >= 2
        and
        newer_green >= 2
        and
        candle_midpoint(candles[0])
        <
        candle_midpoint(candles[3])
    ):

        return "BUY"

    if (
        older_green >= 2
        and
        newer_red >= 2
        and
        candle_midpoint(candles[0])
        >
        candle_midpoint(candles[3])
    ):

        return "SELL"

    return None


# ============================================================
# NEW LAYER 14
# MULTI-CANDLE DIRECTIONAL AGREEMENT
# ============================================================


def multi_candle_directional_agreement(candles):

    if len(candles) < 6:
        return None

    sample = candles[:6]

    green = sum(
        1
        for c in sample
        if c["color"] == "GREEN"
    )

    red = sum(
        1
        for c in sample
        if c["color"] == "RED"
    )

    first_price = candle_midpoint(
        sample[-1]
    )

    latest_price = candle_midpoint(
        sample[0]
    )

    if (
        green >= 4
        and
        latest_price < first_price
    ):

        return "BUY"

    if (
        red >= 4
        and
        latest_price > first_price
    ):

        return "SELL"

    return None


# ============================================================
# NEW LAYER 15
# SIDEWAYS / CHOPPY MARKET FILTER
# ============================================================


def sideways_choppy_filter(candles):

    if len(candles) < CHOP_LOOKBACK:
        return False

    sample = candles[
        :CHOP_LOOKBACK
    ]

    changes = 0

    previous_color = None

    for candle in reversed(sample):

        color = candle["color"]

        if (
            previous_color is not None
            and
            color != previous_color
        ):

            changes += 1

        previous_color = color

    possible_changes = (
        len(sample) - 1
    )

    if possible_changes <= 0:
        return False

    alternation_ratio = (
        changes /
        possible_changes
    )

    # Frequent alternation without
    # structural progress = choppy.

    if (
        alternation_ratio
        >=
        CHOP_ALTERNATION_THRESHOLD
    ):

        first_price = candle_midpoint(
            sample[-1]
        )

        latest_price = candle_midpoint(
            sample[0]
        )

        total_range = max(
            1,
            max(
                candle_midpoint(c)
                for c in sample
            )
            -
            min(
                candle_midpoint(c)
                for c in sample
            )
        )

        net_move = abs(
            latest_price -
            first_price
        )

        progress_ratio = (
            net_move /
            total_range
        )

        if progress_ratio < 0.35:
            return True

    return False


# ============================================================
# NEW LAYER SCORING
# ============================================================


def apply_new_price_action_layers(
    candles,
    resistance,
    support,
    buy_score,
    sell_score,
    buy_reasons,
    sell_reasons
):

    layer_results = {}

    # --------------------------------------------------------
    # 1. HIGHER HIGH / HIGHER LOW
    # --------------------------------------------------------

    result = higher_high_higher_low(
        candles
    )

    layer_results[
        "higher_high_lower"
    ] = result

    if result == "BUY":

        buy_score += 2

        buy_reasons.append(
            "Higher-high / higher-low bullish structure"
        )

    # --------------------------------------------------------
    # 2. LOWER HIGH / LOWER LOW
    # --------------------------------------------------------

    result = lower_high_lower_low(
        candles
    )

    layer_results[
        "lower_high_lower"
    ] = result

    if result == "SELL":

        sell_score += 2

        sell_reasons.append(
            "Lower-high / lower-low bearish structure"
        )

    # --------------------------------------------------------
    # 3. OVERALL TREND
    # --------------------------------------------------------

    trend = overall_trend_direction(
        candles
    )

    layer_results[
        "overall_trend"
    ] = trend

    if trend == "BULLISH":

        buy_score += 2

        buy_reasons.append(
            "Overall candle structure is bullish"
        )

    elif trend == "BEARISH":

        sell_score += 2

        sell_reasons.append(
            "Overall candle structure is bearish"
        )

    # --------------------------------------------------------
    # 4. TREND STRENGTH
    # --------------------------------------------------------

    strength = trend_strength(
        candles
    )

    layer_results[
        "trend_strength"
    ] = strength

    if (
        strength >= 70
        and
        trend == "BULLISH"
    ):

        buy_score += 2

        buy_reasons.append(
            "Strong bullish structural consistency"
        )

    elif (
        strength >= 70
        and
        trend == "BEARISH"
    ):

        sell_score += 2

        sell_reasons.append(
            "Strong bearish structural consistency"
        )

    # --------------------------------------------------------
    # 5. PULLBACK QUALITY
    # --------------------------------------------------------

    result = pullback_quality(
        candles
    )

    layer_results[
        "pullback_quality"
    ] = result

    if result == "BUY":

        buy_score += 2

        buy_reasons.append(
            "Quality bullish pullback"
        )

    elif result == "SELL":

        sell_score += 2

        sell_reasons.append(
            "Quality bearish pullback"
        )

    # --------------------------------------------------------
    # 6. BREAKOUT + RETEST
    # --------------------------------------------------------

    result = breakout_retest(
        candles,
        resistance,
        support
    )

    layer_results[
        "breakout_retest"
    ] = result

    if result == "BUY":

        buy_score += 3

        buy_reasons.append(
            "Bullish breakout and retest"
        )

    elif result == "SELL":

        sell_score += 3

        sell_reasons.append(
            "Bearish breakout and retest"
        )

    # --------------------------------------------------------
    # 7. SWING REJECTION
    # --------------------------------------------------------

    result = swing_rejection(
        candles,
        resistance,
        support
    )

    layer_results[
        "swing_rejection"
    ] = result

    if result == "BUY":

        buy_score += 2

        buy_reasons.append(
            "Bullish swing rejection"
        )

    elif result == "SELL":

        sell_score += 2

        sell_reasons.append(
            "Bearish swing rejection"
        )

    # --------------------------------------------------------
    # 8. TREND EXHAUSTION
    # --------------------------------------------------------

    result = trend_exhaustion(
        candles
    )

    layer_results[
        "trend_exhaustion"
    ] = result

    if result == "BUY":

        buy_score += 1

        buy_reasons.append(
            "Bearish trend exhaustion"
        )

    elif result == "SELL":

        sell_score += 1

        sell_reasons.append(
            "Bullish trend exhaustion"
        )

    # --------------------------------------------------------
    # 9. CANDLE SEQUENCE QUALITY
    # --------------------------------------------------------

    result = candle_sequence_quality(
        candles
    )

    layer_results[
        "candle_sequence"
    ] = result

    if result == "BUY":

        buy_score += 2

        buy_reasons.append(
            "Strong bullish candle sequence"
        )

    elif result == "SELL":

        sell_score += 2

        sell_reasons.append(
            "Strong bearish candle sequence"
        )

    # --------------------------------------------------------
    # 10. IMPULSE VS CORRECTION
    # --------------------------------------------------------

    result = impulse_vs_correction(
        candles
    )

    layer_results[
        "impulse_correction"
    ] = result

    if result == "BUY":

        buy_score += 2

        buy_reasons.append(
            "Bullish impulse outweighs correction"
        )

    elif result == "SELL":

        sell_score += 2

        sell_reasons.append(
            "Bearish impulse outweighs correction"
        )

    # --------------------------------------------------------
    # 11. STRUCTURE BREAK
    # --------------------------------------------------------

    result = structure_break_confirmation(
        candles
    )

    layer_results[
        "structure_break"
    ] = result

    if result == "BUY":

        buy_score += 2

        buy_reasons.append(
            "Bullish structure break"
        )

    elif result == "SELL":

        sell_score += 2

        sell_reasons.append(
            "Bearish structure break"
        )

    # --------------------------------------------------------
    # 12. FAKE BREAKOUT
    # --------------------------------------------------------

    result = fake_breakout_detection(
        candles,
        resistance,
        support
    )

    layer_results[
        "fake_breakout"
    ] = result

    if result == "BUY":

        buy_score += 2

        buy_reasons.append(
            "Bullish fake-breakout rejection"
        )

    elif result == "SELL":

        sell_score += 2

        sell_reasons.append(
            "Bearish fake-breakout rejection"
        )

    # --------------------------------------------------------
    # 13. TREND TRANSITION
    # --------------------------------------------------------

    result = trend_transition_detection(
        candles
    )

    layer_results[
        "trend_transition"
    ] = result

    if result == "BUY":

        buy_score += 2

        buy_reasons.append(
            "Bullish trend transition"
        )

    elif result == "SELL":

        sell_score += 2

        sell_reasons.append(
            "Bearish trend transition"
        )

    # --------------------------------------------------------
    # 14. MULTI-CANDLE AGREEMENT
    # --------------------------------------------------------

    result = multi_candle_directional_agreement(
        candles
    )

    layer_results[
        "multi_candle_agreement"
    ] = result

    if result == "BUY":

        buy_score += 2

        buy_reasons.append(
            "Multi-candle bullish agreement"
        )

    elif result == "SELL":

        sell_score += 2

        sell_reasons.append(
            "Multi-candle bearish agreement"
        )

    # --------------------------------------------------------
    # 15. CHOPPY MARKET FILTER
    # --------------------------------------------------------

    choppy = sideways_choppy_filter(
        candles
    )

    layer_results[
        "choppy"
    ] = choppy

    if choppy:

        # The filter does not create a direction.
        # It penalizes both sides so the engine
        # is less likely to force a signal.

        buy_score = max(
            0,
            buy_score - 3
        )

        sell_score = max(
            0,
            sell_score - 3
        )

        buy_reasons.append(
            "Choppy-market penalty"
        )

        sell_reasons.append(
            "Choppy-market penalty"
        )

    return (
        buy_score,
        sell_score,
        buy_reasons,
        sell_reasons,
        layer_results
    )


# ============================================================
# MAIN STRATEGY
# ============================================================


def analyze_strategy(candles):

    if not candles:

        return {
            "decision": "NO SIGNAL",
            "confidence": 0,
            "buy_score": 0,
            "sell_score": 0,
            "agreement": 0,
            "reasons": []
        }

    resistance = find_resistance_zone(
        candles
    )

    support = find_support_zone(
        candles
    )

    resistance_result = resistance_behavior(
        candles,
        resistance
    )

    support_result = support_behavior(
        candles,
        support
    )

    resistance_strength = reaction_strength(
        resistance,
        candles,
        True
    )

    support_strength = reaction_strength(
        support,
        candles,
        False
    )

    momentum = analyze_body_momentum(
        candles
    )

    consecutive = consecutive_momentum(
        candles
    )

    reversal = body_reversal(
        candles
    )

    two_reversal = two_candle_reversal(
        candles
    )

    three_reversal = three_candle_reversal(
        candles
    )

    pullback = detect_pullback(
        candles
    )

    resistance_failure, support_failure = (
        breakout_failure(
            candles,
            resistance,
            support
        )
    )

    resistance_hold = resistance_breakout_hold(
        candles,
        resistance
    )

    support_hold = support_breakdown_hold(
        candles,
        support
    )

    green_red, red_green = color_confirmation(
        candles,
        resistance,
        support
    )

    continuation = continuation_structure(
        candles,
        resistance,
        support
    )

    buy_score = 0
    sell_score = 0

    buy_reasons = []
    sell_reasons = []

    # ========================================================
    # EXISTING STRATEGIES
    # ========================================================

    # SUPPORT

    if support:
        buy_score += 1

    if support_result["rejection"]:

        buy_score += 2

        buy_reasons.append(
            "Bullish rejection from support"
        )

    if support_failure:

        buy_score += 1

        buy_reasons.append(
            "Support breakdown failure"
        )

    # RESISTANCE

    if resistance:
        sell_score += 1

    if resistance_result["rejection"]:

        sell_score += 2

        sell_reasons.append(
            "Bearish rejection at resistance"
        )

    if resistance_failure:

        sell_score += 1

        sell_reasons.append(
            "Resistance breakout failure"
        )

    # BREAKOUT HOLD

    if resistance_hold:

        buy_score += 3

        buy_reasons.append(
            "Resistance broken and next body held above the zone"
        )

    if support_hold:

        sell_score += 3

        sell_reasons.append(
            "Support broken and next body held below the zone"
        )

    # MOMENTUM

    if consecutive == "BUY":

        buy_score += 1

        buy_reasons.append(
            "Three-candle bullish momentum"
        )

    if consecutive == "SELL":

        sell_score += 1

        sell_reasons.append(
            "Three-candle bearish momentum"
        )

    # CONTINUATION

    if continuation == "BUY":

        buy_score += 1

        buy_reasons.append(
            "Bullish continuation structure"
        )

    if continuation == "SELL":

        sell_score += 1

        sell_reasons.append(
            "Bearish continuation structure"
        )

    # REVERSALS

    if reversal == "BUY":

        buy_score += 2

        buy_reasons.append(
            "Bullish body reversal"
        )

    if reversal == "SELL":

        sell_score += 2

        sell_reasons.append(
            "Bearish body reversal"
        )

    if two_reversal == "BUY":

        buy_score += 1

        buy_reasons.append(
            "Two-candle bullish reversal"
        )

    if two_reversal == "SELL":

        sell_score += 1

        sell_reasons.append(
            "Two-candle bearish reversal"
        )

    if three_reversal == "BUY":

        buy_score += 1

        buy_reasons.append(
            "Three-candle bullish reversal"
        )

    if three_reversal == "SELL":

        sell_score += 1

        sell_reasons.append(
            "Three-candle bearish reversal"
        )

    # PULLBACK

    if pullback == "BUY":

        buy_score += 1

        buy_reasons.append(
            "Bullish pullback structure"
        )

    if pullback == "SELL":

        sell_score += 1

        sell_reasons.append(
            "Bearish pullback structure"
        )

    # COLOR CONFIRMATION

    if red_green:

        buy_score += 2

        buy_reasons.append(
            "RED → GREEN confirmation"
        )

    if green_red:

        sell_score += 2

        sell_reasons.append(
            "GREEN → RED confirmation"
        )

    # ========================================================
    # ADD ALL 15 NEW PRICE-ACTION LAYERS
    # ========================================================

    (
        buy_score,
        sell_score,
        buy_reasons,
        sell_reasons,
        layer_results
    ) = apply_new_price_action_layers(
        candles,
        resistance,
        support,
        buy_score,
        sell_score,
        buy_reasons,
        sell_reasons
    )

    # ========================================================
    # CHOPPY MARKET HARD PROTECTION
    # ========================================================

    choppy = layer_results.get(
        "choppy",
        False
    )

    if choppy:

        # If the chart is clearly alternating
        # without meaningful structural progress,
        # do not allow the engine to force a signal.

        if (
            abs(
                buy_score -
                sell_score
            ) <= 3
        ):

            buy_score = 0
            sell_score = 0

    # ========================================================
    # AGREEMENT
    # ========================================================

    total_score = (
        buy_score +
        sell_score
    )

    if total_score <= 0:

        agreement = 0

    else:

        agreement = round(
            (
                max(
                    buy_score,
                    sell_score
                )
                /
                total_score
            ) * 100
        )

    # ========================================================
    # CONFLICT PROTECTION
    # ========================================================

    difference = abs(
        buy_score -
        sell_score
    )

    conflict = (
        difference <= 1
        and
        total_score >= 4
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    decision = "NO SIGNAL"

    confidence = agreement

    if not conflict:

        if (
            not choppy
            and
            buy_score > sell_score
            and
            buy_score >= 3
            and
            agreement >= MIN_SIGNAL_AGREEMENT
        ):

            decision = "BUY SIGNAL"

        elif (
            not choppy
            and
            sell_score > buy_score
            and
            sell_score >= 3
            and
            agreement >= MIN_SIGNAL_AGREEMENT
        ):

            decision = "SELL SIGNAL"

    # ========================================================
    # REASONS
    # ========================================================

    if decision == "BUY SIGNAL":

        unique = []

        for reason in buy_reasons:

            if reason not in unique:

                unique.append(
                    reason
                )

        reasons = unique[:3]

    elif decision == "SELL SIGNAL":

        unique = []

        for reason in sell_reasons:

            if reason not in unique:

                unique.append(
                    reason
                )

        reasons = unique[:3]

    else:

        reasons = []

    # ========================================================
    # SAFETY
    # ========================================================

    if decision == "NO SIGNAL":

        if choppy:

            reasons = [
                "Market structure is choppy",
                "No clean directional structure",
                "Wait for clearer price action"
            ]

        elif conflict:

            reasons = [
                "BUY and SELL evidence are too close",
                "No clear directional advantage",
                "Wait for a stronger candle structure"
            ]

        elif total_score == 0:

            reasons = [
                "No qualifying setup detected",
                "Candle evidence is insufficient",
                "Do not trade"
            ]

        else:

            reasons = [
                "Directional evidence is below the required threshold",
                "No strong independent confirmation",
                "Do not trade"
            ]

    return {
        "decision": decision,
        "confidence": confidence,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "agreement": agreement,
        "reasons": reasons,
        "resistance": resistance,
        "support": support,
        "resistance_strength": resistance_strength,
        "support_strength": support_strength,
        "momentum": momentum,
        "resistance_hold": resistance_hold,
        "support_hold": support_hold,
        "layer_results": layer_results
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

        x = int(candle["x"])
        y = int(candle["y"])
        width = int(candle["w"])
        height = int(candle["h"])

        cv2.rectangle(
            output,
            (
                x,
                y
            ),
            (
                x + width,
                y + height
            ),
            (0, 255, 255),
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
# TELEGRAM CHANNEL SENDER
# ============================================================


async def send_telegram(response):

    try:

        application = (
            Application.builder()
            .token(TELEGRAM_TOKEN)
            .build()
        )

        await application.bot.send_message(
            chat_id=CHANNEL_ID,
            text=response,
            parse_mode=ParseMode.MARKDOWN
        )

        await application.shutdown()

    except Exception as e:

        print(
            "❌ Channel send error:",
            repr(e)
        )


# ============================================================
# START COMMAND
# ============================================================


async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🧠 OTC Candle Strategy Bot is ready.\n\n"
        "Send a Pocket Option OTC screenshot."
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
        f"strategy_chart_{update.message.message_id}.png"
    )

    detection_path = (
        f"strategy_detection_{update.message.message_id}.png"
    )

    try:

        # ====================================================
        # ANALYZING MESSAGE
        # ====================================================

        await update.message.reply_text(
            "🔍 Analyzing..."
        )

        # ====================================================
        # DOWNLOAD PHOTO
        # ====================================================

        photo = update.message.photo[-1]

        telegram_file = await context.bot.get_file(
            photo.file_id
        )

        await telegram_file.download_to_drive(
            original_path
        )

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        img = load_image(
            original_path
        )

        # ====================================================
        # DETECT CANDLES
        # ====================================================

        candles = detect_candles(
            img
        )

        total = len(
            candles
        )

        # ====================================================
        # NO CANDLES
        # ====================================================

        if total == 0:

            response = (
                "⚪ **NO SIGNAL — DON'T TRADE**"
            )

            await update.message.reply_text(
                response,
                parse_mode=ParseMode.MARKDOWN
            )

            elapsed = (
                time.time()
                -
                start_time
            )

            print(
                f"⚪ No signal in {elapsed:.2f} seconds"
            )

            return

        # ====================================================
        # STRATEGY
        # ====================================================

        strategy = analyze_strategy(
            candles
        )

        signal_time, entry_time = (
            signal_and_entry_times()
        )

        decision = strategy[
            "decision"
        ]

        confidence = strategy[
            "confidence"
        ]

        reasons = strategy[
            "reasons"
        ]

        # ====================================================
        # BUILD RESPONSE
        # ====================================================

        if decision == "BUY SIGNAL":

            response = (
                "🚨 **SIGNAL ALERT**\n\n"
                "🟢 **BUY**\n"
                f"🕐 **Signal Time:** "
                f"{signal_time} 🇳🇬\n"
                f"🎯 **Entry Time:** "
                f"{entry_time} 🇳🇬\n"
                f"💪 **Strength:** "
                f"{confidence}%\n\n"
            )

            for reason in reasons[:3]:

                response += (
                    f"• {reason}\n"
                )

        elif decision == "SELL SIGNAL":

            response = (
                "🚨 **SIGNAL ALERT**\n\n"
                "🔴 **SELL**\n"
                f"🕐 **Signal Time:** "
                f"{signal_time} 🇳🇬\n"
                f"🎯 **Entry Time:** "
                f"{entry_time} 🇳🇬\n"
                f"💪 **Strength:** "
                f"{confidence}%\n\n"
            )

            for reason in reasons[:3]:

                response += (
                    f"• {reason}\n"
                )

        else:

            response = (
                "⚪ **NO SIGNAL — DON'T TRADE**"
            )

        # ====================================================
        # USER RESPONSE
        # ====================================================

        await update.message.reply_text(
            response,
            parse_mode=ParseMode.MARKDOWN
        )

        # ====================================================
        # CHANNEL DELIVERY
        # ====================================================

        if decision in (
            "BUY SIGNAL",
            "SELL SIGNAL"
        ):

            await context.bot.forward_message(
                chat_id=CHANNEL_ID,
                from_chat_id=update.message.chat_id,
                message_id=update.message.message_id
            )

            await send_telegram(
                response
            )

            elapsed = (
                time.time()
                -
                start_time
            )

            print(
                f"✅ Signal sent in {elapsed:.2f} seconds"
            )

        else:

            elapsed = (
                time.time()
                -
                start_time
            )

            print(
                f"⚪ No signal. Analysis completed in "
                f"{elapsed:.2f} seconds"
            )

        # ====================================================
        # DETECTION MAP
        # ====================================================

        detection_map = create_detection_map(
            img,
            candles
        )

        cv2.imwrite(
            detection_path,
            detection_map
        )

    except Exception as e:

        print(
            "❌ ERROR:",
            repr(e)
        )

        await update.message.reply_text(
            f"❌ Error: {str(e)}"
        )

    finally:

        for path in (
            original_path,
            detection_path
        ):

            if os.path.exists(path):

                try:

                    os.remove(path)

                except Exception:

                    pass


# ============================================================
# ERROR HANDLER
# ============================================================


async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):

    print(
        "Telegram error:",
        repr(context.error)
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print(
        "========================================"
    )

    print(
        "🧠 OTC CANDLE STRATEGY BOT"
    )

    print(
        "========================================"
    )

    print(
        "RIGHT → LEFT scanning"
    )

    print(
        "Candle #1 = newest"
    )

    print(
        "Support/resistance body zones"
    )

    print(
        "Breakout + breakout-hold detection"
    )

    print(
        "Reversal detection"
    )

    print(
        "Pullback detection"
    )

    print(
        "Continuation detection"
    )

    print(
        "Conflict protection"
    )

    print(
        "NO SIGNAL protection"
    )

    print(
        "15 NEW PRICE-ACTION LAYERS ENABLED"
    )

    print(
        "Higher High / Higher Low"
    )

    print(
        "Lower High / Lower Low"
    )

    print(
        "Overall Trend Direction"
    )

    print(
        "Trend Strength"
    )

    print(
        "Pullback Quality"
    )

    print(
        "Breakout + Retest"
    )

    print(
        "Swing Rejection"
    )

    print(
        "Trend Exhaustion"
    )

    print(
        "Candle Sequence Quality"
    )

    print(
        "Impulse vs Correction"
    )

    print(
        "Structure-Break Confirmation"
    )

    print(
        "Fake Breakout Detection"
    )

    print(
        "Trend Transition Detection"
    )

    print(
        "Multi-Candle Directional Agreement"
    )

    print(
        "Sideways/Choppy-Market Filter"
    )

    print(
        "Nigeria Time: Africa/Lagos"
    )

    print(
        "Next-candle entry timing"
    )

    print(
        "No new indicators"
    )

    print(
        "No price mapping"
    )

    print(
        "No pair detection"
    )

    print(
        "No random candles"
    )

    print(
        "No automatic trading"
    )

    print(
        "========================================"
    )

    # ========================================================
    # FLASK
    # ========================================================

    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True
    )

    flask_thread.start()

    # ========================================================
    # TELEGRAM APPLICATION
    # ========================================================

    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start_command
        )
    )

    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_photo
        )
    )

    application.add_error_handler(
        error_handler
    )

    print(
        "🤖 Telegram polling started."
    )

    application.run_polling(
        drop_pending_updates=True
    )


# ============================================================
# RUN
# ============================================================


if __name__ == "__main__":

    main()

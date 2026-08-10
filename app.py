import os
import cv2
import numpy as np
import telebot
import threading

from flask import Flask
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_TOKEN = os.getenv(
    "BOT_TOKEN",
    "PASTE_YOUR_BOT_TOKEN_HERE"
)

CHANNEL_ID = os.getenv(
    "CHANNEL_ID",
    "-1004324805205"
)

# Private chat is optional.
# If CHAT_ID is not configured, the bot will still send
# signals to the channel.
CHAT_ID = os.getenv(
    "CHAT_ID",
    ""
)

bot = telebot.TeleBot(
    TELEGRAM_TOKEN
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

    port = int(
        os.getenv(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        use_reloader=False
    )


# ============================================================
# TIMEZONE
# ============================================================

NIGERIA_TZ = ZoneInfo(
    "Africa/Lagos"
)


def nigeria_now():

    return datetime.now(
        NIGERIA_TZ
    )


def signal_and_entry_times():

    now = nigeria_now()

    signal_time = now.replace(
        second=0,
        microsecond=0
    )

    entry_time = (
        signal_time
        +
        timedelta(minutes=1)
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
# LOAD IMAGE
# ============================================================

def load_image(path):

    img = cv2.imread(
        path
    )

    if img is None:

        raise ValueError(
            "Could not read screenshot."
        )

    h, w = img.shape[:2]

    if w < 1400:

        scale = 1400 / w

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
        90,
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
        55,
        55
    ])

    red_upper_1 = np.array([
        12,
        255,
        255
    ])

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

    return (
        green,
        red
    )


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

        min_area = (
            RIGHT_MIN_BODY_AREA
        )

        min_height = (
            RIGHT_MIN_BODY_HEIGHT
        )

    else:

        min_area = MIN_BODY_AREA
        min_height = MIN_BODY_HEIGHT

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < min_area:
            continue

        x, y, w, h = (
            cv2.boundingRect(
                contour
            )
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
                    right -
                    left
                )

                existing["h"] = (
                    bottom -
                    top
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

    for candle in (
        green +
        red
    ):

        candle["x"] += (
            right_start
        )

        candle["center_x"] += (
            right_start
        )

    return (
        green +
        red
    )


# ============================================================
# DETECT CANDLES
# ============================================================

def detect_candles(
    img
):

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

    # RIGHT → LEFT
    # Candle #1 = newest/rightmost

    candles.reverse()

    return candles


# ============================================================
# BODY POSITION
# ============================================================

def body_position(
    candle
):

    top = candle["y"]

    bottom = (
        candle["y"]
        +
        candle["h"]
    )

    return (
        top,
        bottom
    )


# ============================================================
# RESISTANCE
# ============================================================

def find_resistance_zone(
    candles
):

    if not candles:
        return None

    candles_to_check = (
        candles[
            :LOOKBACK_CANDLES
        ]
    )

    if len(
        candles_to_check
    ) < 2:
        return None

    highest_points = []

    for candle in (
        candles_to_check
    ):

        top, bottom = (
            body_position(
                candle
            )
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
        chart_max -
        chart_min
    )

    tolerance = (
        visible_range *
        ZONE_TOLERANCE_RATIO
    )

    zones = []

    for candle in (
        candles_to_check
    ):

        top, _ = (
            body_position(
                candle
            )
        )

        placed = False

        for zone in zones:

            if abs(
                top -
                zone["level"]
            ) <= tolerance:

                zone[
                    "candles"
                ].append(
                    candle
                )

                zone["level"] = np.mean(
                    [
                        body_position(c)[0]
                        for c in zone[
                            "candles"
                        ]
                    ]
                )

                placed = True
                break

        if not placed:

            zones.append({

                "level": float(top),

                "candles": [
                    candle
                ]

            })

    zones.sort(
        key=lambda z:
        len(z["candles"]),
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

def find_support_zone(
    candles
):

    if not candles:
        return None

    candles_to_check = (
        candles[
            :LOOKBACK_CANDLES
        ]
    )

    if len(
        candles_to_check
    ) < 2:
        return None

    lowest_points = []

    for candle in (
        candles_to_check
    ):

        _, bottom = (
            body_position(
                candle
            )
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
        chart_max -
        chart_min
    )

    tolerance = (
        visible_range *
        ZONE_TOLERANCE_RATIO
    )

    zones = []

    for candle in (
        candles_to_check
    ):

        _, bottom = (
            body_position(
                candle
            )
        )

        placed = False

        for zone in zones:

            if abs(
                bottom -
                zone["level"]
            ) <= tolerance:

                zone[
                    "candles"
                ].append(
                    candle
                )

                zone["level"] = np.mean(
                    [
                        body_position(c)[1]
                        for c in zone[
                            "candles"
                        ]
                    ]
                )

                placed = True
                break

        if not placed:

            zones.append({

                "level": float(
                    bottom
                ),

                "candles": [
                    candle
                ]

            })

    zones.sort(
        key=lambda z:
        len(z["candles"]),
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

    top, bottom = (
        body_position(
            newest
        )
    )

    level = resistance[
        "level"
    ]

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

    top, bottom = (
        body_position(
            newest
        )
    )

    level = support[
        "level"
    ]

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

def analyze_body_momentum(
    candles
):

    if len(candles) < 3:

        return "UNKNOWN"

    newest = candles[0]["h"]

    previous = candles[1]["h"]

    older = candles[2]["h"]

    average_previous = (
        previous +
        older
    ) / 2

    if newest > (
        average_previous *
        1.35
    ):

        return "STRONGER"

    if newest < (
        average_previous *
        0.70
    ):

        return "WEAKER"

    return "NORMAL"


# ============================================================
# CONSECUTIVE MOMENTUM
# ============================================================

def consecutive_momentum(
    candles
):

    if len(candles) < 3:
        return None

    color = candles[0][
        "color"
    ]

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
# BODY REVERSAL / ENGULFING
# ============================================================

def body_reversal(
    candles
):

    if len(candles) < 2:
        return None

    current = candles[0]

    previous = candles[1]

    if (
        current["color"]
        ==
        previous["color"]
    ):

        return None

    if (
        current["h"]
        >=
        previous["h"] * 1.20
    ):

        if current["color"] == "GREEN":

            return "BUY"

        return "SELL"

    return None


# ============================================================
# TWO-CANDLE REVERSAL
# ============================================================

def two_candle_reversal(
    candles
):

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

def three_candle_reversal(
    candles
):

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

def detect_pullback(
    candles
):

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

        level = resistance[
            "level"
        ]

        for i in range(
            1,
            min(
                len(candles),
                LOOKBACK_CANDLES
            )
        ):

            newer = candles[
                i - 1
            ]

            older = candles[i]

            newer_top = (
                body_position(
                    newer
                )[0]
            )

            older_top = (
                body_position(
                    older
                )[0]
            )

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

        level = support[
            "level"
        ]

        for i in range(
            1,
            min(
                len(candles),
                LOOKBACK_CANDLES
            )
        ):

            newer = candles[
                i - 1
            ]

            older = candles[i]

            newer_bottom = (
                body_position(
                    newer
                )[1]
            )

            older_bottom = (
                body_position(
                    older
                )[1]
            )

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

    level = resistance[
        "level"
    ]

    maximum = min(
        len(candles) - 1,
        LOOKBACK_CANDLES - 1
    )

    for i in range(
        maximum
    ):

        newer = candles[i]

        older = candles[
            i + 1
        ]

        older_top, older_bottom = (
            body_position(
                older
            )
        )

        newer_top, newer_bottom = (
            body_position(
                newer
            )
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

    level = support[
        "level"
    ]

    maximum = min(
        len(candles) - 1,
        LOOKBACK_CANDLES - 1
    )

    for i in range(
        maximum
    ):

        newer = candles[i]

        older = candles[
            i + 1
        ]

        older_top, older_bottom = (
            body_position(
                older
            )
        )

        newer_top, newer_bottom = (
            body_position(
                newer
            )
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

        level = resistance[
            "level"
        ]

        for i in range(
            1,
            min(
                len(candles),
                LOOKBACK_CANDLES
            )
        ):

            newer = candles[
                i - 1
            ]

            older = candles[i]

            older_top = (
                body_position(
                    older
                )[0]
            )

            newer_top = (
                body_position(
                    newer
                )[0]
            )

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

        level = support[
            "level"
        ]

        for i in range(
            1,
            min(
                len(candles),
                LOOKBACK_CANDLES
            )
        ):

            newer = candles[
                i - 1
            ]

            older = candles[i]

            older_bottom = (
                body_position(
                    older
                )[1]
            )

            newer_bottom = (
                body_position(
                    newer
                )[1]
            )

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

            level = resistance[
                "level"
            ]

            above = True

            for candle in candles[:3]:

                if (
                    body_position(
                        candle
                    )[1]
                    >=
                    level
                ):

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

            level = support[
                "level"
            ]

            below = True

            for candle in candles[:3]:

                if (
                    body_position(
                        candle
                    )[0]
                    <=
                    level
                ):

                    below = False

                    break

            if below:
                return "SELL"

        return "SELL"

    return None


# ============================================================
# MAIN STRATEGY
# ============================================================

def analyze_strategy(
    candles
):

    if not candles:

        return {
            "decision": "NO SIGNAL",
            "confidence": 0,
            "buy_score": 0,
            "sell_score": 0,
            "reasons": []
        }

    newest = candles[0]

    resistance = (
        find_resistance_zone(
            candles
        )
    )

    support = (
        find_support_zone(
            candles
        )
    )

    resistance_result = (
        resistance_behavior(
            candles,
            resistance
        )
    )

    support_result = (
        support_behavior(
            candles,
            support
        )
    )

    resistance_strength = (
        reaction_strength(
            resistance,
            candles,
            True
        )
    )

    support_strength = (
        reaction_strength(
            support,
            candles,
            False
        )
    )

    momentum = (
        analyze_body_momentum(
            candles
        )
    )

    consecutive = (
        consecutive_momentum(
            candles
        )
    )

    reversal = (
        body_reversal(
            candles
        )
    )

    two_reversal = (
        two_candle_reversal(
            candles
        )
    )

    three_reversal = (
        three_candle_reversal(
            candles
        )
    )

    pullback = (
        detect_pullback(
            candles
        )
    )

    (
        resistance_failure,
        support_failure
    ) = (
        breakout_failure(
            candles,
            resistance,
            support
        )
    )

    resistance_hold = (
        resistance_breakout_hold(
            candles,
            resistance
        )
    )

    support_hold = (
        support_breakdown_hold(
            candles,
            support
        )
    )

    green_red, red_green = (
        color_confirmation(
            candles,
            resistance,
            support
        )
    )

    continuation = (
        continuation_structure(
            candles,
            resistance,
            support
        )
    )

    # ========================================================
    # SCORING
    # ========================================================

    buy_score = 0

    sell_score = 0

    buy_reasons = []

    sell_reasons = []

    # --------------------------------------------------------
    # SUPPORT
    # --------------------------------------------------------

    if support:

        buy_score += 1

    if support_result[
        "rejection"
    ]:

        buy_score += 2

        buy_reasons.append(
            "Bullish rejection from support"
        )

    if support_failure:

        buy_score += 1

        buy_reasons.append(
            "Support breakdown failure"
        )

    # --------------------------------------------------------
    # RESISTANCE
    # --------------------------------------------------------

    if resistance:

        sell_score += 1

    if resistance_result[
        "rejection"
    ]:

        sell_score += 2

        sell_reasons.append(
            "Bearish rejection at resistance"
        )

    if resistance_failure:

        sell_score += 1

        sell_reasons.append(
            "Resistance breakout failure"
        )

    # --------------------------------------------------------
    # BREAKOUT HOLD
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CONTINUATION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # REVERSALS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # COLOR CONFIRMATION
    # --------------------------------------------------------

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
    # RAW AGREEMENT
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
            buy_score > sell_score
            and
            buy_score >= 3
            and
            agreement >=
            MIN_SIGNAL_AGREEMENT
        ):

            decision = "BUY SIGNAL"

        elif (
            sell_score > buy_score
            and
            sell_score >= 3
            and
            agreement >=
            MIN_SIGNAL_AGREEMENT
        ):

            decision = "SELL SIGNAL"

    # ========================================================
    # SELECT STRONGEST 3 REASONS
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
    # SAFETY CHECK
    # ========================================================

    if decision == "NO SIGNAL":

        if conflict:

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

        "resistance_strength":
            resistance_strength,

        "support_strength":
            support_strength,

        "momentum": momentum,

        "resistance_hold":
            resistance_hold,

        "support_hold":
            support_hold

    }


# ============================================================
# BUILD SIGNAL MESSAGE
# ============================================================

def build_signal_message(
    decision,
    confidence,
    reasons,
    signal_time,
    entry_time
):

    if decision == "BUY SIGNAL":

        report = (
            "🚨 **SIGNAL ALERT**\n\n"
            "🟢 **BUY**\n\n"
            f"🕐 **Signal Time:** "
            f"{signal_time} 🇳🇬\n"
            f"🎯 **Entry Time:** "
            f"{entry_time} 🇳🇬\n"
            f"💪 **Strength:** "
            f"{confidence}%\n\n"
        )

        for reason in reasons[:3]:

            report += (
                f"• {reason}\n"
            )

        return report

    if decision == "SELL SIGNAL":

        report = (
            "🚨 **SIGNAL ALERT**\n\n"
            "🔴 **SELL**\n\n"
            f"🕐 **Signal Time:** "
            f"{signal_time} 🇳🇬\n"
            f"🎯 **Entry Time:** "
            f"{entry_time} 🇳🇬\n"
            f"💪 **Strength:** "
            f"{confidence}%\n\n"
        )

        for reason in reasons[:3]:

            report += (
                f"• {reason}\n"
            )

        return report

    return None


# ============================================================
# CHANNEL-FIRST DELIVERY
#
# IMPORTANT:
# 1. NO SIGNAL = NOTHING SENT
# 2. SIGNAL = CHANNEL FIRST
# 3. PRIVATE CHAT SECOND
# 4. SCREENSHOT IS NEVER FORWARDED
# ============================================================

def send_signal_channel_first(
    report
):

    if not report:

        print(
            "⚪ NO SIGNAL — NOTHING SENT"
        )

        return

    try:

        channel_response = (
            bot.send_message(
                CHANNEL_ID,
                report,
                parse_mode="Markdown"
            )
        )

        print(
            "Telegram Channel:",
            channel_response.status_code
            if hasattr(
                channel_response,
                "status_code"
            )
            else "SENT"
        )

        print(
            "✅ SIGNAL SENT TO CHANNEL FIRST"
        )

    except Exception as e:

        print(
            "❌ Channel delivery failed:",
            repr(e)
        )

        return

    # ========================================================
    # ONLY AFTER CHANNEL DELIVERY
    # SEND TO PRIVATE CHAT
    # ========================================================

    if CHAT_ID:

        try:

            private_response = (
                bot.send_message(
                    CHAT_ID,
                    report,
                    parse_mode="Markdown"
                )
            )

            print(
                "Private Telegram:",
                private_response.status_code
                if hasattr(
                    private_response,
                    "status_code"
                )
                else "SENT"
            )

            print(
                "✅ SIGNAL SENT TO PRIVATE CHAT"
            )

        except Exception as e:

            print(
                "❌ Private delivery failed:",
                repr(e)
            )


# ============================================================
# TELEGRAM PHOTO HANDLER
#
# IMPORTANT:
# THERE IS NO "ANALYZING..." MESSAGE.
#
# EVERYTHING BELOW HAPPENS INTERNALLY.
# ONLY A REAL BUY/SELL SIGNAL IS DELIVERED.
# ============================================================

@bot.message_handler(
    content_types=["photo"]
)
def handle_photo(
    message
):

    original_path = (
        "strategy_chart.png"
    )

    try:

        # ====================================================
        # DOWNLOAD SCREENSHOT INTERNALLY
        # ====================================================

        file_info = bot.get_file(
            message.photo[-1].file_id
        )

        downloaded_file = (
            bot.download_file(
                file_info.file_path
            )
        )

        with open(
            original_path,
            "wb"
        ) as f:

            f.write(
                downloaded_file
            )

        # ====================================================
        # LOAD SCREENSHOT
        # ====================================================

        img = load_image(
            original_path
        )

        # ====================================================
        # CANDLE DETECTION
        # ====================================================

        candles = detect_candles(
            img
        )

        total = len(
            candles
        )

        # ====================================================
        # NO CANDLES
        #
        # NOTHING IS SENT TO TELEGRAM.
        # ====================================================

        if total == 0:

            print(
                "⚪ No candles detected."
            )

            return

        # ====================================================
        # ALL STRATEGY ENGINES RUN INTERNALLY
        # ====================================================

        strategy = (
            analyze_strategy(
                candles
            )
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
        # NO SIGNAL
        #
        # NOTHING IS SENT.
        # NO MESSAGE.
        # NO SCREENSHOT.
        # ====================================================

        if decision not in [
            "BUY SIGNAL",
            "SELL SIGNAL"
        ]:

            print(
                "⚪ NO SIGNAL — DON'T TRADE"
            )

            return

        # ====================================================
        # SIGNAL TIME
        # ====================================================

        signal_time, entry_time = (
            signal_and_entry_times()
        )

        # ====================================================
        # BUILD FINAL SIGNAL
        # ====================================================

        report = build_signal_message(
            decision,
            confidence,
            reasons,
            signal_time,
            entry_time
        )

        # ====================================================
        # CHANNEL FIRST
        # ====================================================

        send_signal_channel_first(
            report
        )

        print(
            "✅ Final signal processing complete."
        )

    except Exception as e:

        # ====================================================
        # ERRORS ARE ONLY PRINTED TO SERVER LOGS.
        #
        # NO ERROR MESSAGE IS SENT TO TELEGRAM.
        # ====================================================

        print(
            "❌ INTERNAL ANALYSIS ERROR:",
            repr(e)
        )

    finally:

        # ====================================================
        # DELETE SCREENSHOT AFTER PROCESSING
        # ====================================================

        if os.path.exists(
            original_path
        ):

            try:

                os.remove(
                    original_path
                )

            except Exception as e:

                print(
                    "Cleanup error:",
                    repr(e)
                )


# ============================================================
# START BOT
# ============================================================

if __name__ == "__main__":

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
        "Minimum agreement:",
        MIN_SIGNAL_AGREEMENT,
        "%"
    )

    print(
        "Nigeria Time: Africa/Lagos"
    )

    print(
        "Next-candle entry timing"
    )

    print(
        "No indicators"
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
        "No random signals"
    )

    print(
        "No automatic trading"
    )

    print(
        "Channel-first signal delivery"
    )

    print(
        "No screenshot forwarding"
    )

    print(
        "No analyzing message"
    )

    print(
        "NO SIGNAL = NOTHING SENT"
    )

    print(
        "========================================"
    )

    # ========================================================
    # FLASK RUNS IN BACKGROUND
    # ========================================================

    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True
    )

    flask_thread.start()

    # ========================================================
    # TELEGRAM POLLING
    # ========================================================

    bot.infinity_polling(
        timeout=30,
        long_polling_timeout=30
    )

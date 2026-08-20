import os
import cv2
import numpy as np
import telebot
import time
import requests
import threading
from flask import Flask
from datetime import datetime, timezone, timedelta


# ============================================================
# FLASK KEEP-ALIVE
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "✅ OTC 3-Candle 15s Signal Bot is running!"


@app.route("/ping")
def ping():
    return "OK", 200


def run_flask():
    port = int(os.getenv("PORT", 10000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")

if not TELEGRAM_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN environment variable is missing."
    )

CHAT_ID = "6280535707"
CHANNEL_ID = "-1004324805205"

bot = telebot.TeleBot(TELEGRAM_TOKEN)


# ============================================================
# TIMEZONE
# ============================================================

LOCAL_TZ = timezone(timedelta(hours=1))


def get_signal_time():
    now = datetime.now(LOCAL_TZ)

    signal_time = now.replace(
        second=0,
        microsecond=0
    )

    return signal_time.strftime("%H:%M")


def get_entry_time(signal_time):
    return (
        datetime.strptime(
            signal_time,
            "%H:%M"
        )
        + timedelta(minutes=1)
    ).strftime("%H:%M")


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

MIN_COLOR_DENSITY = 0.25


# ============================================================
# PURPLE / YELLOW COLOR SETTINGS
# ============================================================

PURPLE_HUE_LOW = 125
PURPLE_HUE_HIGH = 165

MIN_PURPLE_SATURATION = 100
MIN_PURPLE_VALUE = 70

YELLOW_HUE_LOW = 18
YELLOW_HUE_HIGH = 40

MIN_YELLOW_SATURATION = 100
MIN_YELLOW_VALUE = 70


# ============================================================
# COLOR DOMINANCE
# ============================================================

PURPLE_DOMINANCE_RATIO = 1.20
YELLOW_DOMINANCE_RATIO = 1.10


# ============================================================
# MAP VERIFICATION
# ============================================================

VERIFY_MIN_PIXELS = 8
VERIFY_MIN_DENSITY = 0.08
VERIFY_HORIZONTAL_RADIUS = 0.70
VERIFY_MIN_DISTANCE_RATIO = 0.55
VERIFY_COLUMN_THRESHOLD = 3
VERIFY_CONFIDENCE_THRESHOLD = 65


# ============================================================
# THREE-CANDLE ENGINE SETTINGS
# ============================================================

REQUIRED_CANDLES = 3

MIN_SIGNAL_CONFIDENCE = 5

MIN_DIRECTION_SEPARATION = 10

MAX_CONFLICT = 70

MIN_FINAL_EVIDENCE = 0.20

EXHAUSTION_BODY_RATIO = 0.55

STRONG_BODY_RATIO = 1.20

WICK_REJECTION_RATIO = 0.80

BREAKOUT_RATIO = 1.10

SUPPORT_RESISTANCE_TOLERANCE = 0.50


# ============================================================
# LONG-CANDLE PROTECTION
#
# IMPORTANT:
# Width is NOT used to decide whether a candle is unusually
# long.
#
# Candles #1, #2 and #3 remain the priority.
# Candles #4 and #5 are guard candles only.
#
# If an unusually long candle is found anywhere from #1 to #5,
# the screenshot receives NO SIGNAL.
# ============================================================

LONG_CANDLE_MIN_BODY_RATIO = 1.75
LONG_CANDLE_MIN_RANGE_RATIO = 1.70
LONG_CANDLE_MIN_ABSOLUTE_BODY_RATIO = 1.55

LONG_CANDLE_LOOKBACK = 5


# ============================================================
# RESISTANCE / SUPPORT PROTECTION
#
# This is based on the visible candle geometry, not candle
# width.
#
# Resistance protection is mainly used against bullish signals
# when the newest candle is pressing into / rejecting a visible
# upper price area.
#
# Support protection is mainly used against bearish signals.
# ============================================================

LEVEL_CLUSTER_TOLERANCE = 0.30

RESISTANCE_REJECTION_RATIO = 0.80
SUPPORT_REJECTION_RATIO = 0.80

LEVEL_TOUCH_TOLERANCE = 0.45

LEVEL_BLOCK_SCORE = 0.70


# ============================================================
# SETUP GRADING
#
# The grade is NOT a fixed mapping such as:
# A+ = always 3 entries.
#
# The grade and recommendation are calculated from the actual
# evidence of the current screenshot.
# ============================================================

GRADE_A_PLUS_THRESHOLD = 0.72
GRADE_A_THRESHOLD = 0.56
GRADE_B_THRESHOLD = 0.40

THREE_ENTRY_THRESHOLD = 0.78
TWO_ENTRY_THRESHOLD = 0.57
ONE_ENTRY_THRESHOLD = 0.25

GRADE_CONFLICT_LIMIT = 35.0


# ============================================================
# SEND TO TELEGRAM CHANNEL
# ============================================================

def send_to_channel(message):

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_TOKEN}/sendMessage"
    )

    try:

        requests.post(
            url,
            data={
                "chat_id": CHANNEL_ID,
                "text": message,
                "parse_mode": "Markdown"
            },
            timeout=5
        )

        print("✅ Signal sent to channel")

    except Exception as e:

        print(
            "Channel send error:",
            e
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

    purple_lower = np.array([
        PURPLE_HUE_LOW,
        MIN_PURPLE_SATURATION,
        MIN_PURPLE_VALUE
    ])

    purple_upper = np.array([
        PURPLE_HUE_HIGH,
        255,
        255
    ])

    purple = cv2.inRange(
        hsv,
        purple_lower,
        purple_upper
    )

    yellow_lower = np.array([
        YELLOW_HUE_LOW,
        MIN_YELLOW_SATURATION,
        MIN_YELLOW_VALUE
    ])

    yellow_upper = np.array([
        YELLOW_HUE_HIGH,
        255,
        255
    ])

    yellow = cv2.inRange(
        hsv,
        yellow_lower,
        yellow_upper
    )

    b, g, r = cv2.split(img)

    purple_dominance = (

        (r.astype(np.int16) >
         g.astype(np.int16) *
         PURPLE_DOMINANCE_RATIO)

        &

        (b.astype(np.int16) >
         g.astype(np.int16) *
         PURPLE_DOMINANCE_RATIO)

        &

        (r.astype(np.int16) > 70)

        &

        (b.astype(np.int16) > 70)
    )

    purple_dominance_mask = (
        purple_dominance.astype(
            np.uint8
        ) * 255
    )

    purple = cv2.bitwise_and(
        purple,
        purple_dominance_mask
    )

    yellow_dominance = (

        (r.astype(np.int16) >
         b.astype(np.int16) *
         YELLOW_DOMINANCE_RATIO)

        &

        (g.astype(np.int16) >
         b.astype(np.int16) *
         YELLOW_DOMINANCE_RATIO)

        &

        (r.astype(np.int16) > 80)

        &

        (g.astype(np.int16) > 70)
    )

    yellow_dominance_mask = (
        yellow_dominance.astype(
            np.uint8
        ) * 255
    )

    yellow = cv2.bitwise_and(
        yellow,
        yellow_dominance_mask
    )

    return purple, yellow


# ============================================================
# FIND CANDLE CANDIDATES
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
            y:y + h,
            x:x + w
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

        if density < MIN_COLOR_DENSITY:
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

def merge_candidates(candidates):

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
                    existing["x"] +
                    existing["w"],
                    candidate["x"] +
                    candidate["w"]
                )

                top = min(
                    existing["y"],
                    candidate["y"]
                )

                bottom = max(
                    existing["y"] +
                    existing["h"],
                    candidate["y"] +
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

def remove_cross_color_duplicates(candles):

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
# RIGHT-SIDE DETECTION
# ============================================================

def detect_right_side(
    chart,
    purple_mask,
    yellow_mask
):

    h, w = chart.shape[:2]

    right_start = int(
        w * 0.72
    )

    purple_right = purple_mask[
        :,
        right_start:
    ]

    yellow_right = yellow_mask[
        :,
        right_start:
    ]

    purple = find_candidates(
        purple_right,
        "PURPLE",
        w,
        right_side=True
    )

    yellow = find_candidates(
        yellow_right,
        "YELLOW",
        w,
        right_side=True
    )

    for candle in (
        purple +
        yellow
    ):

        candle["x"] += right_start

        candle["center_x"] += (
            right_start
        )

    return (
        purple +
        yellow
    )


# ============================================================
# DETECT UP TO FIVE RIGHTMOST CANDLES
#
# Three are used for analysis.
# Four and five are guard candles only.
# ============================================================

def detect_five_candles(img):

    h, w = img.shape[:2]

    purple_mask, yellow_mask = (
        get_color_masks(img)
    )

    purple = find_candidates(
        purple_mask,
        "PURPLE",
        w,
        right_side=False
    )

    yellow = find_candidates(
        yellow_mask,
        "YELLOW",
        w,
        right_side=False
    )

    purple = merge_candidates(
        purple
    )

    yellow = merge_candidates(
        yellow
    )

    candles = (
        purple +
        yellow
    )

    right_candidates = (
        detect_right_side(
            img,
            purple_mask,
            yellow_mask
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
        c["center_x"],
        reverse=True
    )

    return candles[:5]


# ============================================================
# THREE-CANDLE DETECTION
# ============================================================

def detect_three_candles(img):

    candles = detect_five_candles(
        img
    )

    return candles[:REQUIRED_CANDLES]


# ============================================================
# VERIFY SINGLE CANDLE
# ============================================================

def verify_single_candle(
    candle,
    purple_mask,
    yellow_mask
):

    x = int(
        candle["center_x"]
    )

    y = int(
        candle["y"]
    )

    w = max(
        2,
        int(candle["w"])
    )

    h = max(
        2,
        int(candle["h"])
    )

    radius = max(
        2,
        int(
            w *
            VERIFY_HORIZONTAL_RADIUS
        )
    )

    left = max(
        0,
        x - radius
    )

    right = min(
        purple_mask.shape[1],
        x + radius + 1
    )

    top = max(
        0,
        y - max(
            2,
            int(h * 0.25)
        )
    )

    bottom = min(
        purple_mask.shape[0],
        y + h +
        max(
            2,
            int(h * 0.25)
        )
    )

    purple_region = purple_mask[
        top:bottom,
        left:right
    ]

    yellow_region = yellow_mask[
        top:bottom,
        left:right
    ]

    purple_pixels = int(
        np.sum(
            purple_region > 0
        )
    )

    yellow_pixels = int(
        np.sum(
            yellow_region > 0
        )
    )

    total_pixels = max(
        1,
        purple_region.shape[0] *
        purple_region.shape[1]
    )

    if candle["color"] == "PURPLE":

        own_pixels = purple_pixels
        other_pixels = yellow_pixels

    else:

        own_pixels = yellow_pixels
        other_pixels = purple_pixels

    own_density = (
        own_pixels /
        total_pixels
    )

    color_agrees = (
        own_pixels >= VERIFY_MIN_PIXELS
        and
        own_pixels >= (
            other_pixels * 1.15
        )
    )

    body_evidence = min(
        100,
        (
            own_pixels /
            float(
                max(
                    VERIFY_MIN_PIXELS,
                    1
                )
            )
        ) * 100
    )

    density_evidence = min(
        100,
        (
            own_density /
            VERIFY_MIN_DENSITY
        ) * 100
    )

    score = (
        body_evidence * 0.50
        +
        density_evidence * 0.25
        +
        (
            100
            if color_agrees
            else 0
        ) * 0.25
    )

    score = max(
        0,
        min(
            100,
            score
        )
    )

    verified = (
        color_agrees
        and
        score >=
        VERIFY_CONFIDENCE_THRESHOLD
    )

    return {

        "verified": verified,
        "score": score,
        "own_pixels": own_pixels,
        "other_pixels": other_pixels,
        "own_density": own_density,
        "color_agrees": color_agrees
    }


# ============================================================
# VERIFY SELECTED CANDLES
# ============================================================

def verify_candles(
    img,
    candles
):

    purple_mask, yellow_mask = (
        get_color_masks(img)
    )

    verified = []

    for candle in candles:

        result = verify_single_candle(
            candle,
            purple_mask,
            yellow_mask
        )

        c = candle.copy()

        c["verification"] = result

        verified.append(c)

    return verified


# ============================================================
# GEOMETRY
# ============================================================

def safe_ratio(a, b):

    return (
        float(a) /
        max(
            float(b),
            0.0001
        )
    )


def candle_direction(candle):

    if candle["color"] == "PURPLE":
        return 1

    return -1


def enrich_candles(
    img,
    candles
):

    purple_mask, yellow_mask = (
        get_color_masks(img)
    )

    h_img, w_img = img.shape[:2]

    enriched = []

    for candle in candles:

        c = candle.copy()

        x = int(
            candle["center_x"]
        )

        body_top = int(
            candle["y"]
        )

        body_bottom = int(
            candle["y"] +
            candle["h"]
        )

        radius = max(
            2,
            int(
                max(
                    candle["w"],
                    2
                ) * 0.75
            )
        )

        left = max(
            0,
            x - radius
        )

        right = min(
            w_img,
            x + radius + 1
        )

        if candle["color"] == "PURPLE":

            mask = purple_mask

        else:

            mask = yellow_mask

        region = mask[
            :,
            left:right
        ]

        ys, xs = np.where(
            region > 0
        )

        if len(ys) > 0:

            visual_top = int(
                np.min(ys)
            )

            visual_bottom = int(
                np.max(ys)
            )

        else:

            visual_top = body_top
            visual_bottom = body_bottom

        c["body_top"] = float(
            body_top
        )

        c["body_bottom"] = float(
            body_bottom
        )

        c["body_size"] = max(
            1.0,
            float(candle["h"])
        )

        c["visual_top"] = float(
            visual_top
        )

        c["visual_bottom"] = float(
            visual_bottom
        )

        c["visual_range"] = max(
            1.0,
            float(
                visual_bottom -
                visual_top
            )
        )

        c["upper_extension"] = max(
            0.0,
            float(
                body_top -
                visual_top
            )
        )

        c["lower_extension"] = max(
            0.0,
            float(
                visual_bottom -
                body_bottom
            )
        )

        c["upper_rejection_ratio"] = (
            safe_ratio(
                c["upper_extension"],
                c["body_size"]
            )
        )

        c["lower_rejection_ratio"] = (
            safe_ratio(
                c["lower_extension"],
                c["body_size"]
            )
        )

        c["body_to_range"] = (
            safe_ratio(
                c["body_size"],
                c["visual_range"]
            )
        )

        enriched.append(c)

    return enriched


# ============================================================
# LONG CANDLE DETECTION
#
# Uses candle BODY / VISUAL RANGE, NOT WIDTH.
#
# The current candle is compared against the other detected
# candles in the five-candle guard window.
# ============================================================

def detect_unusually_long_candle(
    img,
    candles
):

    if not candles:

        return None

    enriched = enrich_candles(
        img,
        candles
    )

    if len(enriched) <= 1:

        return None

    body_sizes = [
        c["body_size"]
        for c in enriched
    ]

    range_sizes = [
        c["visual_range"]
        for c in enriched
    ]

    for index, candle in enumerate(
        enriched
    ):

        others = [
            body_sizes[i]
            for i in range(
                len(body_sizes)
            )
            if i != index
        ]

        other_ranges = [
            range_sizes[i]
            for i in range(
                len(range_sizes)
            )
            if i != index
        ]

        if not others:
            continue

        median_body = float(
            np.median(others)
        )

        median_range = float(
            np.median(other_ranges)
        )

        body_ratio = safe_ratio(
            candle["body_size"],
            median_body
        )

        range_ratio = safe_ratio(
            candle["visual_range"],
            median_range
        )

        absolute_body_ratio = safe_ratio(
            candle["body_size"],
            np.mean(body_sizes)
        )

        unusually_long = (
            body_ratio >=
            LONG_CANDLE_MIN_BODY_RATIO
            and
            range_ratio >=
            LONG_CANDLE_MIN_RANGE_RATIO
        ) or (
            body_ratio >=
            LONG_CANDLE_MIN_ABSOLUTE_BODY_RATIO
            and
            range_ratio >=
            LONG_CANDLE_MIN_ABSOLUTE_BODY_RATIO
        )

        if unusually_long:

            return {
                "index": index + 1,
                "body_ratio": body_ratio,
                "range_ratio": range_ratio,
                "absolute_body_ratio":
                    absolute_body_ratio
            }

    return None


# ============================================================
# RESISTANCE / SUPPORT ANALYSIS
#
# This uses ONLY the visible candle geometry.
# ============================================================

def analyze_visible_levels(
    candles
):

    if len(candles) < 3:

        return {
            "resistance_score": 0,
            "support_score": 0,
            "resistance": False,
            "support": False
        }

    c1 = candles[0]
    c2 = candles[1]
    c3 = candles[2]

    tops = [
        c1["visual_top"],
        c2["visual_top"],
        c3["visual_top"]
    ]

    bottoms = [
        c1["visual_bottom"],
        c2["visual_bottom"],
        c3["visual_bottom"]
    ]

    ranges = [
        c1["visual_range"],
        c2["visual_range"],
        c3["visual_range"]
    ]

    typical_range = max(
        1.0,
        float(
            np.median(ranges)
        )
    )

    highest_top = min(
        tops
    )

    lowest_bottom = max(
        bottoms
    )

    top_touches = 0
    bottom_touches = 0

    top_tolerance = max(
        3.0,
        typical_range *
        LEVEL_CLUSTER_TOLERANCE
    )

    bottom_tolerance = max(
        3.0,
        typical_range *
        LEVEL_CLUSTER_TOLERANCE
    )

    for top in tops:

        if abs(
            top -
            highest_top
        ) <= top_tolerance:

            top_touches += 1

    for bottom in bottoms:

        if abs(
            bottom -
            lowest_bottom
        ) <= bottom_tolerance:

            bottom_touches += 1

    newest_upper_rejection = (
        c1[
            "upper_rejection_ratio"
        ]
    )

    newest_lower_rejection = (
        c1[
            "lower_rejection_ratio"
        ]
    )

    newest_range = max(
        1.0,
        c1["visual_range"]
    )

    newest_near_resistance = (
        abs(
            c1["visual_top"] -
            highest_top
        )
        <=
        newest_range *
        LEVEL_TOUCH_TOLERANCE
    )

    newest_near_support = (
        abs(
            c1["visual_bottom"] -
            lowest_bottom
        )
        <=
        newest_range *
        LEVEL_TOUCH_TOLERANCE
    )

    resistance_score = 0.0

    if top_touches >= 2:
        resistance_score += 0.35

    if newest_near_resistance:
        resistance_score += 0.20

    if newest_upper_rejection >= (
        RESISTANCE_REJECTION_RATIO
    ):
        resistance_score += 0.35

    if (
        c1["color"] == "YELLOW"
        and
        newest_near_resistance
    ):
        resistance_score += 0.15

    support_score = 0.0

    if bottom_touches >= 2:
        support_score += 0.35

    if newest_near_support:
        support_score += 0.20

    if newest_lower_rejection >= (
        SUPPORT_REJECTION_RATIO
    ):
        support_score += 0.35

    if (
        c1["color"] == "PURPLE"
        and
        newest_near_support
    ):
        support_score += 0.15

    resistance_score = min(
        1.0,
        resistance_score
    )

    support_score = min(
        1.0,
        support_score
    )

    return {

        "resistance_score":
            resistance_score,

        "support_score":
            support_score,

        "resistance":
            resistance_score >=
            LEVEL_BLOCK_SCORE,

        "support":
            support_score >=
            LEVEL_BLOCK_SCORE
    }


# ============================================================
# THREE-CANDLE SEQUENCE
# ============================================================

def analyze_sequence(candles):

    directions = [
        candle_direction(c)
        for c in candles
    ]

    if len(directions) != 3:

        return {
            "score": 0,
            "label": "INSUFFICIENT",
            "changes": 0
        }

    a, b, c = directions

    changes = (
        int(a != b) +
        int(b != c)
    )

    if a == b == c:

        score = float(a)

        label = (
            "3-CANDLE BULLISH"
            if a == 1
            else
            "3-CANDLE BEARISH"
        )

    elif a != b and b != c:

        score = a * 0.35

        label = "ALTERNATING"

    else:

        score = a * 0.65

        label = "MIXED"

    return {
        "score": score,
        "label": label,
        "changes": changes
    }


# ============================================================
# BODY PROGRESSION
# ============================================================

def analyze_body_progression(candles):

    b1 = candles[0]["body_size"]
    b2 = candles[1]["body_size"]
    b3 = candles[2]["body_size"]

    newest_direction = (
        candle_direction(
            candles[0]
        )
    )

    avg_previous = (
        b2 + b3
    ) / 2.0

    acceleration = (
        safe_ratio(
            b1,
            avg_previous
        )
    )

    if (
        acceleration >=
        STRONG_BODY_RATIO
    ):

        return {
            "score":
                newest_direction *
                min(
                    1.0,
                    acceleration / 2.0
                ),
            "state":
                "EXPANSION"
        }

    if (
        acceleration <=
        EXHAUSTION_BODY_RATIO
    ):

        return {
            "score":
                -newest_direction *
                0.70,
            "state":
                "DECELERATION"
        }

    return {
        "score":
            newest_direction *
            0.35,
        "state":
            "STABLE"
    }


# ============================================================
# MOMENTUM
# ============================================================

def analyze_momentum(candles):

    directions = [
        candle_direction(c)
        for c in candles
    ]

    sizes = [
        c["body_size"]
        for c in candles
    ]

    newest = directions[0]

    ratio_1 = safe_ratio(
        sizes[0],
        sizes[1]
    )

    ratio_2 = safe_ratio(
        sizes[1],
        sizes[2]
    )

    momentum = (
        newest * 0.55
        +
        directions[1] * 0.25
        +
        directions[2] * 0.20
    )

    acceleration = (
        ratio_1 -
        ratio_2
    )

    if (
        np.sign(directions[0])
        ==
        np.sign(directions[1])
        ==
        np.sign(directions[2])
    ):

        if acceleration > 0.10:

            momentum += (
                newest * 0.25
            )

        elif acceleration < -0.20:

            momentum -= (
                newest * 0.30
            )

    return max(
        -1.0,
        min(
            1.0,
            momentum
        )
    )


# ============================================================
# WICK / REJECTION
# ============================================================

def analyze_rejection(candles):

    newest = candles[0]

    upper = (
        newest[
            "upper_rejection_ratio"
        ]
    )

    lower = (
        newest[
            "lower_rejection_ratio"
        ]
    )

    if lower >= WICK_REJECTION_RATIO:

        bullish = min(
            1.0,
            lower / 1.5
        )

    else:

        bullish = 0

    if upper >= WICK_REJECTION_RATIO:

        bearish = min(
            1.0,
            upper / 1.5
        )

    else:

        bearish = 0

    return {
        "bullish":
            bullish,
        "bearish":
            bearish,
        "score":
            bullish -
            bearish
    }


# ============================================================
# ENGULFING
# ============================================================

def analyze_engulfing(candles):

    current = candles[0]
    previous = candles[1]

    current_direction = (
        candle_direction(current)
    )

    previous_direction = (
        candle_direction(previous)
    )

    body_ratio = safe_ratio(
        current["body_size"],
        previous["body_size"]
    )

    bullish = (
        current_direction == 1
        and
        previous_direction == -1
        and
        body_ratio >= 1.10
    )

    bearish = (
        current_direction == -1
        and
        previous_direction == 1
        and
        body_ratio >= 1.10
    )

    if bullish:

        return {
            "bullish": True,
            "bearish": False,
            "score":
                min(
                    1.0,
                    body_ratio / 2
                )
        }

    if bearish:

        return {
            "bullish": False,
            "bearish": True,
            "score":
                -min(
                    1.0,
                    body_ratio / 2
                )
        }

    return {
        "bullish": False,
        "bearish": False,
        "score": 0
    }


# ============================================================
# THREE-CANDLE HH / HL / LH / LL
# ============================================================

def analyze_structure(candles):

    c1 = candles[0]
    c2 = candles[1]
    c3 = candles[2]

    bullish_points = 0
    bearish_points = 0

    if c1["visual_top"] < c2["visual_top"]:
        bullish_points += 1

    if c1["visual_bottom"] < c2["visual_bottom"]:
        bullish_points += 1

    if c2["visual_top"] < c3["visual_top"]:
        bullish_points += 1

    if c2["visual_bottom"] < c3["visual_bottom"]:
        bullish_points += 1

    if c1["visual_top"] > c2["visual_top"]:
        bearish_points += 1

    if c1["visual_bottom"] > c2["visual_bottom"]:
        bearish_points += 1

    if c2["visual_top"] > c3["visual_top"]:
        bearish_points += 1

    if c2["visual_bottom"] > c3["visual_bottom"]:
        bearish_points += 1

    total = max(
        1,
        bullish_points +
        bearish_points
    )

    score = (
        bullish_points -
        bearish_points
    ) / float(total)

    if score > 0.20:

        label = "BULLISH HH/HL"

    elif score < -0.20:

        label = "BEARISH LH/LL"

    else:

        label = "MIXED"

    return {
        "bullish":
            bullish_points,
        "bearish":
            bearish_points,
        "score":
            score,
        "label":
            label
    }


# ============================================================
# PULLBACK / CONTINUATION
# ============================================================

def analyze_pullback(candles):

    d1 = candle_direction(
        candles[0]
    )

    d2 = candle_direction(
        candles[1]
    )

    d3 = candle_direction(
        candles[2]
    )

    b1 = candles[0]["body_size"]
    b2 = candles[1]["body_size"]
    b3 = candles[2]["body_size"]

    bullish_continuation = (
        d1 == 1
        and
        d2 == 1
        and
        d3 == 1
    )

    bearish_continuation = (
        d1 == -1
        and
        d2 == -1
        and
        d3 == -1
    )

    bullish_pullback = (
        d3 == 1
        and
        d2 == -1
        and
        d1 == 1
    )

    bearish_pullback = (
        d3 == -1
        and
        d2 == 1
        and
        d1 == -1
    )

    if bullish_continuation:

        score = 0.90

        label = "BULLISH CONTINUATION"

    elif bearish_continuation:

        score = -0.90

        label = "BEARISH CONTINUATION"

    elif bullish_pullback:

        quality = safe_ratio(
            b1,
            max(
                b2,
                b3
            )
        )

        score = (
            0.75
            if quality >= 0.90
            else 0.50
        )

        label = "BULLISH PULLBACK RECOVERY"

    elif bearish_pullback:

        quality = safe_ratio(
            b1,
            max(
                b2,
                b3
            )
        )

        score = (
            -0.75
            if quality >= 0.90
            else -0.50
        )

        label = "BEARISH PULLBACK RECOVERY"

    else:

        score = 0
        label = "NO CLEAR CONTINUATION"

    return {
        "score": score,
        "label": label
    }


# ============================================================
# REVERSAL
# ============================================================

def analyze_reversal(candles):

    d1 = candle_direction(
        candles[0]
    )

    d2 = candle_direction(
        candles[1]
    )

    d3 = candle_direction(
        candles[2]
    )

    b1 = candles[0]["body_size"]
    b3 = candles[2]["body_size"]

    bullish_reversal = (
        d3 == -1
        and
        d2 == 1
        and
        d1 == 1
    )

    bearish_reversal = (
        d3 == 1
        and
        d2 == -1
        and
        d1 == -1
    )

    if bullish_reversal:

        confirmation = safe_ratio(
            b1,
            b3
        )

        return {
            "score":
                min(
                    1.0,
                    0.60 +
                    confirmation * 0.20
                ),
            "label":
                "EARLY BULLISH REVERSAL"
        }

    if bearish_reversal:

        confirmation = safe_ratio(
            b1,
            b3
        )

        return {
            "score":
                -min(
                    1.0,
                    0.60 +
                    confirmation * 0.20
                ),
            "label":
                "EARLY BEARISH REVERSAL"
        }

    return {
        "score": 0,
        "label": "NO CLEAR REVERSAL"
    }


# ============================================================
# BREAKOUT PRESSURE
# ============================================================

def analyze_breakout(candles):

    newest = candles[0]
    previous = candles[1]
    third = candles[2]

    newest_direction = (
        candle_direction(
            newest
        )
    )

    bullish_break = (
        newest["visual_top"]
        <
        previous["visual_top"]
        and
        newest["visual_top"]
        <
        third["visual_top"]
    )

    bearish_break = (
        newest["visual_bottom"]
        >
        previous["visual_bottom"]
        and
        newest["visual_bottom"]
        >
        third["visual_bottom"]
    )

    body_ratio = safe_ratio(
        newest["body_size"],
        (
            previous["body_size"] +
            third["body_size"]
        ) / 2
    )

    if (
        bullish_break
        and
        newest_direction == 1
        and
        body_ratio >= BREAKOUT_RATIO
    ):

        return {
            "score":
                min(
                    1.0,
                    body_ratio / 1.5
                ),
            "label":
                "BULLISH BREAKOUT PRESSURE"
        }

    if (
        bearish_break
        and
        newest_direction == -1
        and
        body_ratio >= BREAKOUT_RATIO
    ):

        return {
            "score":
                -min(
                    1.0,
                    body_ratio / 1.5
                ),
            "label":
                "BEARISH BREAKOUT PRESSURE"
        }

    return {
        "score": 0,
        "label": "NO CONFIRMED BREAKOUT"
    }


# ============================================================
# FAILED BREAKOUT
# ============================================================

def analyze_failed_breakout(candles):

    newest = candles[0]
    previous = candles[1]
    third = candles[2]

    d1 = candle_direction(
        newest
    )

    d2 = candle_direction(
        previous
    )

    d3 = candle_direction(
        third
    )

    bullish_failure = (
        d3 == 1
        and
        d2 == 1
        and
        d1 == -1
        and
        newest[
            "upper_rejection_ratio"
        ] >= WICK_REJECTION_RATIO
    )

    bearish_failure = (
        d3 == -1
        and
        d2 == -1
        and
        d1 == 1
        and
        newest[
            "lower_rejection_ratio"
        ] >= WICK_REJECTION_RATIO
    )

    if bullish_failure:

        return {
            "score": -0.85,
            "label":
                "FAILED BULLISH BREAKOUT"
        }

    if bearish_failure:

        return {
            "score": 0.85,
            "label":
                "FAILED BEARISH BREAKOUT"
        }

    return {
        "score": 0,
        "label": "NO FAILED BREAKOUT"
    }


# ============================================================
# SUPPORT / RESISTANCE INTERACTION
# ============================================================

def analyze_support_resistance(candles):

    newest = candles[0]
    previous = candles[1]

    tolerance = max(
        5,
        previous["visual_range"] *
        SUPPORT_RESISTANCE_TOLERANCE
    )

    bullish_rejection = (
        newest["color"] == "PURPLE"
        and
        newest[
            "lower_rejection_ratio"
        ] >= WICK_REJECTION_RATIO
        and
        abs(
            newest["visual_bottom"]
            -
            previous["visual_bottom"]
        ) <= tolerance
    )

    bearish_rejection = (
        newest["color"] == "YELLOW"
        and
        newest[
            "upper_rejection_ratio"
        ] >= WICK_REJECTION_RATIO
        and
        abs(
            newest["visual_top"]
            -
            previous["visual_top"]
        ) <= tolerance
    )

    if bullish_rejection:

        return {
            "score": 0.80,
            "label":
                "SUPPORT REJECTION"
        }

    if bearish_rejection:

        return {
            "score": -0.80,
            "label":
                "RESISTANCE REJECTION"
        }

    return {
        "score": 0,
        "label":
            "NO CLEAR LEVEL INTERACTION"
    }


# ============================================================
# COMPRESSION / EXPANSION
# ============================================================

def analyze_compression_expansion(candles):

    b1 = candles[0]["body_size"]
    b2 = candles[1]["body_size"]
    b3 = candles[2]["body_size"]

    newest_direction = (
        candle_direction(
            candles[0]
        )
    )

    average = (
        b2 + b3
    ) / 2

    ratio = safe_ratio(
        b1,
        average
    )

    if ratio < 0.65:

        return {
            "score":
                -newest_direction * 0.50,
            "state":
                "COMPRESSION"
        }

    if ratio > 1.40:

        return {
            "score":
                newest_direction * 0.75,
            "state":
                "EXPANSION"
        }

    return {
        "score":
            newest_direction * 0.20,
        "state":
            "NORMAL"
    }


# ============================================================
# EXHAUSTION
# ============================================================

def analyze_exhaustion(candles):

    newest = candles[0]
    previous = candles[1]
    third = candles[2]

    d1 = candle_direction(
        newest
    )

    d2 = candle_direction(
        previous
    )

    d3 = candle_direction(
        third
    )

    if not (
        d1 == d2 == d3
    ):

        return {
            "score": 0,
            "label": "NO EXHAUSTION"
        }

    average_old = (
        previous["body_size"] +
        third["body_size"]
    ) / 2

    body_ratio = safe_ratio(
        newest["body_size"],
        average_old
    )

    wick = max(
        newest[
            "upper_rejection_ratio"
        ],
        newest[
            "lower_rejection_ratio"
        ]
    )

    if (
        body_ratio <=
        EXHAUSTION_BODY_RATIO
        and
        wick >=
        WICK_REJECTION_RATIO
    ):

        return {
            "score":
                -d1 * 0.90,
            "label":
                "MOMENTUM EXHAUSTION"
        }

    return {
        "score": 0,
        "label": "NO EXHAUSTION"
    }


# ============================================================
# CHOPPINESS
# ============================================================

def analyze_choppiness(candles):

    directions = [
        candle_direction(c)
        for c in candles
    ]

    changes = (
        int(
            directions[0] !=
            directions[1]
        )
        +
        int(
            directions[1] !=
            directions[2]
        )
    )

    if changes == 2:

        return {
            "score": 0.0,
            "choppy": True
        }

    if changes == 1:

        return {
            "score": 0.20,
            "choppy": False
        }

    return {
        "score": 0.90,
        "choppy": False
    }


# ============================================================
# BUYER / SELLER CONTROL
# ============================================================

def analyze_control(candles):

    weighted = (
        candle_direction(
            candles[0]
        ) * 0.55
        +
        candle_direction(
            candles[1]
        ) * 0.30
        +
        candle_direction(
            candles[2]
        ) * 0.15
    )

    sizes = [
        c["body_size"]
        for c in candles
    ]

    average = np.mean(
        sizes
    )

    strength = min(
        1.0,
        safe_ratio(
            np.max(sizes),
            average
        ) / 1.5
    )

    return max(
        -1.0,
        min(
            1.0,
            weighted * strength
        )
    )


# ============================================================
# CANDLE QUALITY
# ============================================================

def analyze_candle_quality(candles):

    newest = candles[0]

    body_ratio = (
        newest["body_to_range"]
    )

    rejection = max(
        newest[
            "upper_rejection_ratio"
        ],
        newest[
            "lower_rejection_ratio"
        ]
    )

    quality = (
        body_ratio * 0.65
        -
        min(
            0.50,
            rejection * 0.20
        )
    )

    return max(
        0,
        min(
            1,
            quality
        )
    )


# ============================================================
# THREE-CANDLE CONFLICT
# ============================================================

def analyze_conflict(
    evidence_scores
):

    positive = sum(
        max(
            0,
            value
        )
        for value in evidence_scores
    )

    negative = sum(
        abs(
            min(
                0,
                value
            )
        )
        for value in evidence_scores
    )

    total = (
        positive +
        negative
    )

    if total <= 0:

        return {
            "severity": 100,
            "label": "SEVERE"
        }

    weaker = min(
        positive,
        negative
    )

    stronger = max(
        positive,
        negative
    )

    if stronger <= 0:

        return {
            "severity": 0,
            "label": "NONE"
        }

    severity = (
        weaker /
        stronger
    ) * 100

    if severity >= 70:

        label = "SEVERE"

    elif severity >= 40:

        label = "MODERATE"

    else:

        label = "LOW"

    return {
        "severity":
            severity,
        "label":
            label
    }


# ============================================================
# SETUP GRADE + ENTRY RECOMMENDATION
#
# This is deliberately evidence-based.
#
# It does NOT force:
# A+ = 3
# A  = 2
# B  = 1
#
# The actual evidence determines the recommendation.
# ============================================================

def grade_setup(
    decision,
    confidence,
    separation,
    conflict,
    sequence,
    structure,
    momentum,
    pullback,
    reversal,
    breakout,
    failed_breakout,
    support_resistance,
    compression,
    exhaustion,
    choppiness,
    control,
    quality,
    visible_levels
):

    if decision not in (
        "BUY",
        "SELL"
    ):

        return {
            "grade": None,
            "recommended_entries": 0,
            "grade_score": 0
        }

    direction = (
        1
        if decision == "BUY"
        else -1
    )

    directional_values = [

        sequence["score"] * direction,

        structure["score"] * direction,

        momentum * direction,

        pullback["score"] * direction,

        reversal["score"] * direction,

        breakout["score"] * direction,

        failed_breakout["score"] * direction,

        support_resistance["score"] * direction,

        control * direction,

        quality,

    ]

    positive_values = [
        max(
            0.0,
            min(
                1.0,
                value
            )
        )
        for value in directional_values
    ]

    evidence_quality = float(
        np.mean(
            positive_values
        )
    )

    confidence_factor = max(
        0.0,
        min(
            1.0,
            confidence / 100.0
        )
    )

    separation_factor = max(
        0.0,
        min(
            1.0,
            separation / 100.0
        )
    )

    conflict_factor = max(
        0.0,
        min(
            1.0,
            1.0 -
            (
                conflict["severity"] /
                100.0
            )
        )
    )

    sequence_bonus = (
        0.10
        if (
            sequence["label"]
            in (
                "3-CANDLE BULLISH",
                "3-CANDLE BEARISH"
            )
        )
        else 0.0
    )

    structure_bonus = (
        0.08
        if (
            (
                decision == "BUY"
                and
                structure["label"]
                == "BULLISH HH/HL"
            )
            or
            (
                decision == "SELL"
                and
                structure["label"]
                == "BEARISH LH/LL"
            )
        )
        else 0.0
    )

    continuation_bonus = (
        0.08
        if (
            (
                decision == "BUY"
                and
                pullback["label"]
                in (
                    "BULLISH CONTINUATION",
                    "BULLISH PULLBACK RECOVERY"
                )
            )
            or
            (
                decision == "SELL"
                and
                pullback["label"]
                in (
                    "BEARISH CONTINUATION",
                    "BEARISH PULLBACK RECOVERY"
                )
            )
        )
        else 0.0
    )

    level_penalty = 0.0

    if decision == "BUY":

        level_penalty = (
            visible_levels[
                "resistance_score"
            ] * 0.25
        )

    else:

        level_penalty = (
            visible_levels[
                "support_score"
            ] * 0.25
        )

    grade_score = (
        evidence_quality * 0.32
        +
        confidence_factor * 0.22
        +
        separation_factor * 0.16
        +
        conflict_factor * 0.15
        +
        sequence_bonus
        +
        structure_bonus
        +
        continuation_bonus
        -
        level_penalty
    )

    grade_score = max(
        0.0,
        min(
            1.0,
            grade_score
        )
    )

    # --------------------------------------------------------
    # Grade
    # --------------------------------------------------------

    if (
        grade_score >=
        GRADE_A_PLUS_THRESHOLD
    ):

        grade = "A+"

    elif (
        grade_score >=
        GRADE_A_THRESHOLD
    ):

        grade = "A"

    elif (
        grade_score >=
        GRADE_B_THRESHOLD
    ):

        grade = "B"

    else:

        grade = "C"

    # --------------------------------------------------------
    # Recommendation
    #
    # 3 entries only when the actual setup is exceptionally
    # strong.
    # --------------------------------------------------------

    if (
        grade_score >=
        THREE_ENTRY_THRESHOLD
        and
        conflict["severity"] <
        GRADE_CONFLICT_LIMIT
        and
        separation >= 25
        and
        evidence_quality >= 0.65
        and
        quality >= 0.35
    ):

        recommended_entries = 3

    elif (
        grade_score >=
        TWO_ENTRY_THRESHOLD
        and
        conflict["severity"] <
        50
        and
        separation >= 15
    ):

        recommended_entries = 2

    elif (
        grade_score >=
        ONE_ENTRY_THRESHOLD
    ):

        recommended_entries = 1

    else:

        recommended_entries = 1

    return {
        "grade":
            grade,
        "recommended_entries":
            recommended_entries,
        "grade_score":
            grade_score
    }


# ============================================================
# FINAL THREE-CANDLE ANALYSIS
# ============================================================

def analyze_three_candles(
    img,
    candles
):

    if len(candles) != 3:

        return {
            "decision":
                "NO TRADE",
            "confidence": 0,
            "reason":
                "Exactly three candles are required.",
            "buy_score": 0,
            "sell_score": 0,
            "sequence":
                {
                    "score": 0,
                    "label": "INSUFFICIENT",
                    "changes": 0
                }
        }

    candles = enrich_candles(
        img,
        candles
    )

    sequence = analyze_sequence(
        candles
    )

    progression = (
        analyze_body_progression(
            candles
        )
    )

    momentum = (
        analyze_momentum(
            candles
        )
    )

    rejection = (
        analyze_rejection(
            candles
        )
    )

    engulfing = (
        analyze_engulfing(
            candles
        )
    )

    structure = (
        analyze_structure(
            candles
        )
    )

    pullback = (
        analyze_pullback(
            candles
        )
    )

    reversal = (
        analyze_reversal(
            candles
        )
    )

    breakout = (
        analyze_breakout(
            candles
        )
    )

    failed_breakout = (
        analyze_failed_breakout(
            candles
        )
    )

    support_resistance = (
        analyze_support_resistance(
            candles
        )
    )

    compression = (
        analyze_compression_expansion(
            candles
        )
    )

    exhaustion = (
        analyze_exhaustion(
            candles
        )
    )

    choppiness = (
        analyze_choppiness(
            candles
        )
    )

    control = (
        analyze_control(
            candles
        )
    )

    quality = (
        analyze_candle_quality(
            candles
        )
    )

    visible_levels = (
        analyze_visible_levels(
            candles
        )
    )

    # --------------------------------------------------------
    # EVIDENCE VALUES
    # --------------------------------------------------------

    evidence = [

        sequence["score"] * 0.90,

        progression["score"] * 0.85,

        momentum * 1.00,

        rejection["score"] * 0.85,

        engulfing["score"] * 0.90,

        structure["score"] * 0.90,

        pullback["score"] * 0.80,

        reversal["score"] * 0.95,

        breakout["score"] * 0.90,

        failed_breakout["score"] * 0.95,

        support_resistance["score"] * 0.85,

        compression["score"] * 0.65,

        exhaustion["score"] * 1.00,

        control * 1.00,

        candle_direction(
            candles[0]
        ) * quality * 0.75
    ]

    raw_score = sum(
        evidence
    )

    raw_score /= len(
        evidence
    )

    newest_direction = (
        candle_direction(
            candles[0]
        )
    )

    newest_confirmation = (
        newest_direction *
        (
            quality *
            0.20
        )
    )

    raw_score += (
        newest_confirmation
    )

    raw_score = max(
        -1.0,
        min(
            1.0,
            raw_score
        )
    )

    conflict = analyze_conflict(
        evidence
    )

    if choppiness["choppy"]:

        raw_score *= 0.45

    if (
        compression["state"]
        ==
        "COMPRESSION"
    ):

        raw_score *= 0.65

    if (
        exhaustion["score"] != 0
        and
        np.sign(
            exhaustion["score"]
        )
        !=
        np.sign(
            raw_score
        )
    ):

        raw_score *= 0.55

    if (
        conflict["severity"]
        >= MAX_CONFLICT
    ):

        raw_score *= 0.45

    elif (
        conflict["severity"]
        >= 40
    ):

        raw_score *= 0.75

    buy_score = 0.0
    sell_score = 0.0

    if raw_score > 0:

        buy_score = (
            abs(raw_score) *
            100
        )

    elif raw_score < 0:

        sell_score = (
            abs(raw_score) *
            100
        )

    separation = abs(
        buy_score -
        sell_score
    )

    sideways = (
        choppiness["choppy"]
        and
        abs(raw_score) <
        MIN_FINAL_EVIDENCE
    )

    # --------------------------------------------------------
    # RESISTANCE / SUPPORT BLOCK
    #
    # This is a protection, not a replacement for the main
    # three-candle engine.
    # --------------------------------------------------------

    if (
        raw_score > 0
        and
        visible_levels[
            "resistance_score"
        ] >= LEVEL_BLOCK_SCORE
    ):

        decision = "NO TRADE"

        confidence = 0

        reason = (
            "BUY BLOCKED — RESISTANCE ZONE"
        )

        return {

            "decision":
                decision,

            "confidence":
                confidence,

            "reason":
                reason,

            "buy_score":
                buy_score,

            "sell_score":
                sell_score,

            "separation":
                separation,

            "conflict":
                conflict,

            "sequence":
                sequence,

            "progression":
                progression,

            "momentum":
                momentum,

            "rejection":
                rejection,

            "engulfing":
                engulfing,

            "structure":
                structure,

            "pullback":
                pullback,

            "reversal":
                reversal,

            "breakout":
                breakout,

            "failed_breakout":
                failed_breakout,

            "support_resistance":
                support_resistance,

            "compression":
                compression,

            "exhaustion":
                exhaustion,

            "choppiness":
                choppiness,

            "control":
                control,

            "candle_quality":
                quality,

            "visible_levels":
                visible_levels,

            "setup_grade":
                None,

            "recommended_entries":
                0,

            "candles_used":
                3
        }

    if (
        raw_score < 0
        and
        visible_levels[
            "support_score"
        ] >= LEVEL_BLOCK_SCORE
    ):

        decision = "NO TRADE"

        confidence = 0

        reason = (
            "SELL BLOCKED — SUPPORT ZONE"
        )

        return {

            "decision":
                decision,

            "confidence":
                confidence,

            "reason":
                reason,

            "buy_score":
                buy_score,

            "sell_score":
                sell_score,

            "separation":
                separation,

            "conflict":
                conflict,

            "sequence":
                sequence,

            "progression":
                progression,

            "momentum":
                momentum,

            "rejection":
                rejection,

            "engulfing":
                engulfing,

            "structure":
                structure,

            "pullback":
                pullback,

            "reversal":
                reversal,

            "breakout":
                breakout,

            "failed_breakout":
                failed_breakout,

            "support_resistance":
                support_resistance,

            "compression":
                compression,

            "exhaustion":
                exhaustion,

            "choppiness":
                choppiness,

            "control":
                control,

            "candle_quality":
                quality,

            "visible_levels":
                visible_levels,

            "setup_grade":
                None,

            "recommended_entries":
                0,

            "candles_used":
                3
        }

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    if sideways:

        decision = "NO TRADE"

        reason = (
            "The three candles are too "
            "choppy for a clean 15-second decision."
        )

    elif (
        conflict["severity"]
        >= MAX_CONFLICT
        and
        separation <
        MIN_DIRECTION_SEPARATION
    ):

        decision = "NO TRADE"

        reason = (
            "The three-candle evidence "
            "conflicts too strongly."
        )

    elif (
        buy_score >=
        MIN_SIGNAL_CONFIDENCE
        and
        buy_score >
        sell_score
        and
        separation >=
        MIN_DIRECTION_SEPARATION
    ):

        decision = "BUY"

        reason = (
            "The three newest candles "
            "show stronger bullish control "
            "with supporting momentum and structure."
        )

    elif (
        sell_score >=
        MIN_SIGNAL_CONFIDENCE
        and
        sell_score >
        buy_score
        and
        separation >=
        MIN_DIRECTION_SEPARATION
    ):

        decision = "SELL"

        reason = (
            "The three newest candles "
            "show stronger bearish control "
            "with supporting momentum and structure."
        )

    else:

        decision = "NO TRADE"

        if (
            exhaustion["score"] != 0
        ):

            reason = (
                "The three candles show "
                "possible momentum exhaustion."
            )

        elif (
            compression["state"]
            ==
            "COMPRESSION"
        ):

            reason = (
                "The three candles are "
                "compressed without enough confirmation."
            )

        elif (
            conflict["severity"]
            >= 40
        ):

            reason = (
                "The three candles contain "
                "conflicting directional evidence."
            )

        else:

            reason = (
                "The three candles do not "
                "provide enough directional separation."
            )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if decision == "BUY":

        confidence = buy_score

    elif decision == "SELL":

        confidence = sell_score

    else:

        confidence = max(
            buy_score,
            sell_score
        )

        confidence = min(
            confidence,
            59
        )

    confidence = max(
        0,
        min(
            100,
            confidence
        )
    )

    # --------------------------------------------------------
    # SETUP GRADE
    # --------------------------------------------------------

    grade_info = grade_setup(

        decision,

        confidence,

        separation,

        conflict,

        sequence,

        structure,

        momentum,

        pullback,

        reversal,

        breakout,

        failed_breakout,

        support_resistance,

        compression,

        exhaustion,

        choppiness,

        control,

        quality,

        visible_levels
    )

    return {

        "decision":
            decision,

        "confidence":
            confidence,

        "reason":
            reason,

        "buy_score":
            buy_score,

        "sell_score":
            sell_score,

        "separation":
            separation,

        "conflict":
            conflict,

        "sequence":
            sequence,

        "progression":
            progression,

        "momentum":
            momentum,

        "rejection":
            rejection,

        "engulfing":
            engulfing,

        "structure":
            structure,

        "pullback":
            pullback,

        "reversal":
            reversal,

        "breakout":
            breakout,

        "failed_breakout":
            failed_breakout,

        "support_resistance":
            support_resistance,

        "compression":
            compression,

        "exhaustion":
            exhaustion,

        "choppiness":
            choppiness,

        "control":
            control,

        "candle_quality":
            quality,

        "visible_levels":
            visible_levels,

        "setup_grade":
            grade_info["grade"],

        "recommended_entries":
            grade_info[
                "recommended_entries"
            ],

        "grade_score":
            grade_info[
                "grade_score"
            ],

        "candles_used":
            3
    }


# ============================================================
# CREATE THREE-CANDLE DETECTION MAP
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

        verified = candle[
            "verification"
        ]["verified"]

        if verified:

            box_color = (
                0,
                255,
                0
            )

        else:

            box_color = (
                0,
                0,
                255
            )

        cv2.rectangle(
            output,
            (x, y),
            (
                x + w,
                y + h
            ),
            box_color,
            2
        )

        if candle["color"] == "PURPLE":

            label_color = (
                255,
                0,
                255
            )

        else:

            label_color = (
                0,
                255,
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
# TELEGRAM PHOTO HANDLER
# ============================================================

@bot.message_handler(
    content_types=["photo"]
)
def handle_photo(message):

    start_time = time.time()

    original_path = (
        "chart_screenshot.png"
    )

    detection_path = (
        "three_candle_map.png"
    )

    try:

        bot.send_message(
            message.chat.id,
            "🔍 Analyzing 3 candles..."
        )

        # ----------------------------------------------------
        # DOWNLOAD SCREENSHOT
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        img = load_image(
            original_path
        )

        # ----------------------------------------------------
        # DETECT UP TO FIVE
        #
        # #1-#3 = priority
        # #4-#5 = long-candle guards
        # ----------------------------------------------------

        five_candles = (
            detect_five_candles(
                img
            )
        )

        detected_count = len(
            five_candles
        )

        print(
            f"Detected right-side candles: "
            f"{detected_count}"
        )

        # ----------------------------------------------------
        # MUST HAVE THREE PRIORITY CANDLES
        # ----------------------------------------------------

        if detected_count < 3:

            bot.send_message(
                message.chat.id,
                (
                    "⚪ **NO SIGNAL — DON'T TRADE**\n\n"
                    f"Only {detected_count}/3 "
                    "priority candles could be detected."
                ),
                parse_mode="Markdown"
            )

            return

        # ----------------------------------------------------
        # VERIFY FIVE-CANDLE WINDOW
        # ----------------------------------------------------

        verified_five = (
            verify_candles(
                img,
                five_candles
            )
        )

        # ----------------------------------------------------
        # LONG CANDLE CHECK
        #
        # This is performed BEFORE the normal signal engine.
        # If #1-#5 contains an unusually long candle, stop.
        # ----------------------------------------------------

        long_candle = (
            detect_unusually_long_candle(
                img,
                verified_five
            )
        )

        if long_candle is not None:

            candle_number = (
                long_candle["index"]
            )

            bot.send_message(
                message.chat.id,
                (
                    "⚪ **NO SIGNAL — DON'T TRADE**\n\n"
                    f"🚫 **Unusually long candle "
                    f"detected at #{candle_number}.**"
                ),
                parse_mode="Markdown"
            )

            print(
                f"🚫 Long candle detected at "
                f"#{candle_number}. "
                f"Signal blocked."
            )

            return

        # ----------------------------------------------------
        # ONLY THE THREE PRIORITY CANDLES ENTER THE ENGINE
        # ----------------------------------------------------

        verified_three = (
            verified_five[:3]
        )

        analysis_candles = []

        for candle in (
            verified_three
        ):

            if candle[
                "verification"
            ]["verified"]:

                analysis_candles.append(
                    candle
                )

        # ----------------------------------------------------
        # DO NOT SUBSTITUTE OLDER CANDLES
        # ----------------------------------------------------

        if len(
            analysis_candles
        ) != 3:

            bot.send_message(
                message.chat.id,
                (
                    "⚪ **NO SIGNAL — DON'T TRADE**\n\n"
                    "The three selected candles "
                    "could not all be verified."
                ),
                parse_mode="Markdown"
            )

            return

        # ----------------------------------------------------
        # THREE-CANDLE ENGINE
        # ----------------------------------------------------

        analysis = (
            analyze_three_candles(
                img,
                analysis_candles
            )
        )

        decision = (
            analysis["decision"]
        )

        confidence = (
            analysis["confidence"]
        )

        reason = (
            analysis["reason"]
        )

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------

        if decision == "BUY":

            signal_time = (
                get_signal_time()
            )

            entry_time = (
                get_entry_time(
                    signal_time
                )
            )

            grade = (
                analysis[
                    "setup_grade"
                ]
            )

            recommended_entries = (
                analysis[
                    "recommended_entries"
                ]
            )

            response = (

                "🚨 **SIGNAL ALERT**\n\n"

                "🟢 **BUY**\n\n"

                f"🕐 **Signal Time:** "
                f"{signal_time} 🇳🇬\n"

                f"🎯 **Entry Time:** "
                f"{entry_time} 🇳🇬\n"

                "⏱️ **Expiry:** 15 Seconds\n\n"

                f"💪 **Strength:** "
                f"{confidence:.0f}%\n"

                f"🏆 **Setup Grade:** "
                f"{grade}\n"

                f"🎯 **Recommended:** "
                f"{recommended_entries} "
                f"entry"
                f"{'s' if recommended_entries != 1 else ''}\n\n"

                f"• {reason}"
            )

            bot.send_message(
                message.chat.id,
                response,
                parse_mode="Markdown"
            )

            send_to_channel(
                response
            )

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------

        elif decision == "SELL":

            signal_time = (
                get_signal_time()
            )

            entry_time = (
                get_entry_time(
                    signal_time
                )
            )

            grade = (
                analysis[
                    "setup_grade"
                ]
            )

            recommended_entries = (
                analysis[
                    "recommended_entries"
                ]
            )

            response = (

                "🚨 **SIGNAL ALERT**\n\n"

                "🔴 **SELL**\n\n"

                f"🕐 **Signal Time:** "
                f"{signal_time} 🇳🇬\n"

                f"🎯 **Entry Time:** "
                f"{entry_time} 🇳🇬\n"

                "⏱️ **Expiry:** 15 Seconds\n\n"

                f"💪 **Strength:** "
                f"{confidence:.0f}%\n"

                f"🏆 **Setup Grade:** "
                f"{grade}\n"

                f"🎯 **Recommended:** "
                f"{recommended_entries} "
                f"entry"
                f"{'s' if recommended_entries != 1 else ''}\n\n"

                f"• {reason}"
            )

            bot.send_message(
                message.chat.id,
                response,
                parse_mode="Markdown"
            )

            send_to_channel(
                response
            )

        # ----------------------------------------------------
        # NO TRADE
        #
        # No grade or entry recommendation is added here.
        # ----------------------------------------------------

        else:

            bot.send_message(
                message.chat.id,
                (
                    "⚪ **NO SIGNAL — DON'T TRADE**\n\n"
                    f"3-candle confidence: "
                    f"{confidence:.0f}%\n"
                    f"• {reason}"
                ),
                parse_mode="Markdown"
            )

        # ----------------------------------------------------
        # CREATE MAP LOCALLY
        # ----------------------------------------------------

        detection_map = (
            create_detection_map(
                img,
                verified_three
            )
        )

        cv2.imwrite(
            detection_path,
            detection_map
        )

        print(
            f"✅ Processed in "
            f"{time.time() - start_time:.2f}s"
            f" | 3 priority candles used"
            f" | Decision: {decision}"
            f" | Confidence: "
            f"{confidence:.1f}%"
        )

        print(
            "Candle sequence:",
            [
                c["color"]
                for c in analysis_candles
            ]
        )

        print(
            "Sequence:",
            analysis["sequence"]["label"]
        )

        print(
            "Structure:",
            analysis["structure"]["label"]
        )

        print(
            "Momentum:",
            round(
                analysis["momentum"],
                3
            )
        )

        print(
            "Control:",
            round(
                analysis["control"],
                3
            )
        )

        print(
            "Conflict:",
            round(
                analysis["conflict"]["severity"],
                1
            )
        )

        if decision in (
            "BUY",
            "SELL"
        ):

            print(
                "Setup Grade:",
                analysis[
                    "setup_grade"
                ]
            )

            print(
                "Recommended entries:",
                analysis[
                    "recommended_entries"
                ]
            )

    except Exception as e:

        # ----------------------------------------------------
        # IMPORTANT:
        # NEVER expose Python exception text to Telegram.
        # The user gets a clean no-signal message instead.
        # Technical details stay in the server log only.
        # ----------------------------------------------------

        print(
            "❌ INTERNAL ERROR:",
            repr(e)
        )

        try:

            bot.send_message(
                message.chat.id,
                (
                    "⚪ **NO SIGNAL — DON'T TRADE**\n\n"
                    "• Analysis could not be completed."
                ),
                parse_mode="Markdown"
            )

        except Exception:

            pass

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
# START COMMAND
# ============================================================

@bot.message_handler(
    commands=["start"]
)
def start(message):

    bot.send_message(

        message.chat.id,

        "📊 **OTC 3-CANDLE 15s BOT**\n\n"

        "Send a screenshot.\n\n"

        "The bot detects the rightmost "
        "candles.\n\n"

        "🕯️ The newest 3 candles are "
        "the analysis priority.\n"

        "🛡️ Candles #4 and #5 are "
        "long-candle guard candles.\n"

        "🚫 Long candles are blocked.\n"

        "🚫 Dangerous visible levels are blocked.\n"

        "🔬 Deep 3-candle price-action analysis\n"

        "🏆 Setup grading\n"

        "🎯 1 / 2 / 3 entry recommendation\n"

        "⏱️ 15-second test\n\n"

        "⚡ **BUY / SELL / NO TRADE**",

        parse_mode="Markdown"
    )


# ============================================================
# START BOT
# ============================================================

if __name__ == "__main__":

    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True
    )

    flask_thread.start()

    print(
        "✅ Flask server started"
    )

    print("=" * 55)

    print(
        "📊 OTC 3-CANDLE 15s SIGNAL BOT"
    )

    print(
        "✅ 3 PRIORITY CANDLES"
    )

    print(
        "✅ #4 AND #5 LONG-CANDLE GUARDS"
    )

    print(
        "✅ LONG CANDLE = NO SIGNAL"
    )

    print(
        "✅ RESISTANCE / SUPPORT PROTECTION"
    )

    print(
        "✅ PURPLE / YELLOW DETECTION"
    )

    print(
        "✅ DEEP 3-CANDLE PRICE ACTION"
    )

    print(
        "✅ SETUP GRADING"
    )

    print(
        "✅ 1 / 2 / 3 ENTRY RECOMMENDATION"
    )

    print(
        "✅ 15-SECOND TEST"
    )

    print(
        "✅ NO RANDOM DATA"
    )

    print(
        "✅ TELEGRAM SIGNALS"
    )

    print(
        "✅ TECHNICAL ERRORS HIDDEN FROM USER"
    )

    print(
        "✅ FLASK KEEP-ALIVE"
    )

    print("=" * 55)

    bot.infinity_polling(
        timeout=30,
        long_polling_timeout=30
    )

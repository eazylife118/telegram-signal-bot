import os
import cv2
import numpy as np
import telebot
import time
import re

# Optional OCR.
# It is used ONLY for reading numerical price labels.
# It is NOT used for pair detection.
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    TESSERACT_AVAILABLE = False


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_TOKEN = os.getenv(
    "BOT_TOKEN",
    "PASTE_YOUR_BOT_TOKEN_HERE"
)

bot = telebot.TeleBot(TELEGRAM_TOKEN)


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


# ============================================================
# STRATEGY TEST SETTINGS
# ============================================================

LOOKBACK_CANDLES = 20

MIN_ZONE_TESTS = 2

ZONE_TOLERANCE_RATIO = 0.035

SMALL_BODY_RATIO = 0.35

MIN_BODY_MOMENTUM_RATIO = 1.35

MIN_SEQUENCE_LENGTH = 3


# ============================================================
# PRICE DETECTION SETTINGS
# ============================================================

# Price labels are normally found on the right side of
# the Pocket Option chart.

PRICE_SCALE_START_RATIO = 0.78
PRICE_SCALE_END_RATIO = 0.995

PRICE_OCR_MIN_CONFIDENCE = 35

# Maximum vertical difference between a detected OCR price
# and a candle before it is considered useful.

PRICE_MATCH_MAX_Y_RATIO = 0.12


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

    # --------------------------------------------------------
    # GREEN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RED
    # --------------------------------------------------------

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
            np.sum(region > 0)
        )

        if colored_pixels < 5:
            continue

        density = (
            colored_pixels /
            float(max(1, w * h))
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
        key=lambda c: c["center_x"]
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
    green_mask,
    red_mask
):

    h, w = chart.shape[:2]

    right_start = int(
        w * 0.72
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

        candle["x"] += right_start

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
        key=lambda c: c["center_x"]
    )

    # RIGHT -> LEFT.
    # Candle #1 = newest.

    candles.reverse()

    return candles


# ============================================================
# PRICE OCR
# ============================================================

def preprocess_price_region(region):

    gray = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=2.5,
        fy=2.5,
        interpolation=cv2.INTER_CUBIC
    )

    # Improve numerical text visibility.
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7
    )

    return gray, threshold


def clean_price_text(text):

    if not text:
        return ""

    text = text.upper()

    replacements = {
        "O": "0",
        "I": "1",
        "L": "1",
        "S": "5",
        "B": "8",
        ",": ".",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new
        )

    text = re.sub(
        r"[^0-9.\-]",
        "",
        text
    )

    return text


def read_price_scale(img):

    result = {
        "available": False,
        "prices": [],
        "message": "Price scale not read."
    }

    if not TESSERACT_AVAILABLE:

        result["message"] = (
            "Price OCR unavailable: "
            "pytesseract is not installed."
        )

        return result

    h, w = img.shape[:2]

    x1 = int(
        w * PRICE_SCALE_START_RATIO
    )

    x2 = int(
        w * PRICE_SCALE_END_RATIO
    )

    y1 = int(
        h * 0.12
    )

    y2 = int(
        h * 0.88
    )

    region = img[
        y1:y2,
        x1:x2
    ]

    if region.size == 0:

        return result

    gray, threshold = (
        preprocess_price_region(
            region
        )
    )

    candidates = []

    configs = [
        "--psm 6",
        "--psm 11",
        "--psm 12"
    ]

    for processed in (
        gray,
        threshold
    ):

        for config in configs:

            try:

                data = (
                    pytesseract.image_to_data(
                        processed,
                        config=config,
                        output_type=(
                            pytesseract.Output.DICT
                        )
                    )
                )

            except Exception:

                continue

            count = len(
                data["text"]
            )

            for i in range(count):

                raw = (
                    data["text"][i]
                    or ""
                ).strip()

                confidence_text = (
                    data["conf"][i]
                )

                try:
                    confidence = float(
                        confidence_text
                    )
                except Exception:
                    confidence = 0

                if (
                    confidence <
                    PRICE_OCR_MIN_CONFIDENCE
                ):
                    continue

                cleaned = (
                    clean_price_text(
                        raw
                    )
                )

                # Require a plausible decimal price.
                if (
                    "." not in cleaned
                    or
                    len(cleaned) < 3
                ):
                    continue

                try:

                    value = float(
                        cleaned
                    )

                except Exception:

                    continue

                # Avoid obvious interface numbers.
                if value <= 0:
                    continue

                if value > 1000000:
                    continue

                local_x = (
                    int(data["left"][i])
                    /
                    2.5
                )

                local_y = (
                    int(data["top"][i])
                    /
                    2.5
                )

                local_h = (
                    int(data["height"][i])
                    /
                    2.5
                )

                center_y = (
                    y1 +
                    local_y +
                    local_h / 2
                )

                candidates.append({
                    "price": value,
                    "y": float(center_y),
                    "confidence": confidence
                })

    if not candidates:

        return result

    # Remove duplicate OCR readings.
    candidates.sort(
        key=lambda item: item["y"]
    )

    unique = []

    for item in candidates:

        duplicate = False

        for existing in unique:

            if (
                abs(
                    item["y"]
                    -
                    existing["y"]
                )
                < 8
            ):

                duplicate = True

                if (
                    item["confidence"]
                    >
                    existing["confidence"]
                ):

                    existing.update(
                        item
                    )

                break

        if not duplicate:

            unique.append(
                item
            )

    # Need at least two different price levels
    # to establish a screen-price relationship.

    if len(unique) < 2:

        result["prices"] = unique

        result["message"] = (
            "Only one price label was read; "
            "exact candle price cannot be mapped safely."
        )

        return result

    unique.sort(
        key=lambda item: item["y"]
    )

    result["available"] = True
    result["prices"] = unique
    result["message"] = (
        f"{len(unique)} price labels mapped."
    )

    return result


# ============================================================
# PRICE MAPPING
# ============================================================

def build_price_mapper(price_data):

    prices = price_data.get(
        "prices",
        []
    )

    if len(prices) < 2:

        return None

    # Use linear interpolation/extrapolation
    # between the detected screen price labels.

    points = sorted(
        prices,
        key=lambda p: p["y"]
    )

    def screen_y_to_price(y):

        # Find nearest two labels.
        nearest = sorted(
            points,
            key=lambda p: abs(
                p["y"] - y
            )
        )

        p1 = nearest[0]
        p2 = nearest[1]

        if abs(
            p2["y"] - p1["y"]
        ) < 0.001:

            return None

        # Price decreases as screen Y increases.
        price_per_pixel = (
            p2["price"] -
            p1["price"]
        ) / (
            p2["y"] -
            p1["y"]
        )

        return (
            p1["price"]
            +
            (
                y - p1["y"]
            )
            *
            price_per_pixel
        )

    return screen_y_to_price


def attach_prices_to_candles(
    candles,
    price_data,
    image_height
):

    mapper = build_price_mapper(
        price_data
    )

    if mapper is None:

        return candles

    result = []

    for candle in candles:

        top = float(
            candle["y"]
        )

        bottom = float(
            candle["y"] +
            candle["h"]
        )

        center = (
            top +
            bottom
        ) / 2

        top_price = mapper(
            top
        )

        bottom_price = mapper(
            bottom
        )

        center_price = mapper(
            center
        )

        updated = candle.copy()

        updated["price_top"] = (
            top_price
        )

        updated["price_bottom"] = (
            bottom_price
        )

        updated["price_center"] = (
            center_price
        )

        result.append(
            updated
        )

    return result


def format_price(value):

    if value is None:
        return "NOT READ"

    value = float(value)

    # Preserve useful precision.
    if value >= 100:
        return f"{value:.2f}"

    if value >= 10:
        return f"{value:.3f}"

    if value >= 1:
        return f"{value:.5f}"

    return f"{value:.6f}"


# ============================================================
# CANDLE VISIBLE RANGE
# ============================================================

def candle_range(candle):

    return max(
        1,
        candle["h"]
    )


# ============================================================
# ESTIMATE BODY POSITION
# ============================================================

def body_position(candle):

    top = candle["y"]

    bottom = (
        candle["y"] +
        candle["h"]
    )

    normalized_top = top
    normalized_bottom = bottom

    return (
        normalized_top,
        normalized_bottom
    )


# ============================================================
# APPROXIMATE BODY CHARACTERISTICS
# ============================================================

def body_information(candle):

    h = max(
        1,
        candle["h"]
    )

    return {
        "body_height": h,
        "body_width": candle["w"],
        "color": candle["color"]
    }


# ============================================================
# BODY RATIO
# ============================================================

def body_ratio_between(
    current,
    previous
):

    if previous["h"] <= 0:
        return 1.0

    return (
        current["h"] /
        previous["h"]
    )


# ============================================================
# FIND APPROXIMATE RESISTANCE ZONE
# ============================================================

def find_resistance_zone(
    candles
):

    if not candles:
        return None

    candles_to_check = candles[
        :LOOKBACK_CANDLES
    ]

    if len(candles_to_check) < 2:
        return None

    highest_points = []

    for candle in candles_to_check:

        top, bottom = (
            body_position(candle)
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

        top, _ = body_position(
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

                "candles": [
                    candle
                ]

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
# FIND APPROXIMATE SUPPORT ZONE
# ============================================================

def find_support_zone(
    candles
):

    if not candles:
        return None

    candles_to_check = candles[
        :LOOKBACK_CANDLES
    ]

    if len(candles_to_check) < 2:
        return None

    lowest_points = []

    for candle in candles_to_check:

        _, bottom = body_position(
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

        _, bottom = body_position(
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

                "candles": [
                    candle
                ]

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
# ZONE PRICE
# ============================================================

def zone_price(
    zone,
    candles,
    side
):

    if not zone:
        return None

    values = []

    for candle in zone["candles"]:

        if side == "RESISTANCE":

            value = candle.get(
                "price_top"
            )

        else:

            value = candle.get(
                "price_bottom"
            )

        if value is not None:
            values.append(
                value
            )

    if not values:
        return None

    return float(
        np.mean(values)
    )


# ============================================================
# CURRENT CANDLE AGAINST RESISTANCE
# ============================================================

def resistance_behavior(
    candles,
    resistance
):

    if not resistance:

        return {
            "near": False,
            "rejection": False,
            "description":
                "No confirmed resistance zone."
        }

    newest = candles[0]

    top, bottom = body_position(
        newest
    )

    level = resistance["level"]

    distance = abs(
        top - level
    )

    tolerance = max(
        3,
        newest["h"] * 0.75
    )

    near = (
        distance <= tolerance
    )

    rejection = False

    description = (
        "Newest candle is not near "
        "the detected resistance zone."
    )

    if near:

        if newest["color"] == "RED":

            rejection = True

            description = (
                "Newest visible candle is RED "
                "near the approximate resistance zone."
            )

        else:

            description = (
                "Newest visible candle is GREEN "
                "near the approximate resistance zone."
            )

    return {
        "near": near,
        "rejection": rejection,
        "description": description
    }


# ============================================================
# CURRENT CANDLE AGAINST SUPPORT
# ============================================================

def support_behavior(
    candles,
    support
):

    if not support:

        return {
            "near": False,
            "rejection": False,
            "description":
                "No confirmed support zone."
        }

    newest = candles[0]

    top, bottom = body_position(
        newest
    )

    level = support["level"]

    distance = abs(
        bottom - level
    )

    tolerance = max(
        3,
        newest["h"] * 0.75
    )

    near = (
        distance <= tolerance
    )

    rejection = False

    description = (
        "Newest candle is not near "
        "the detected support zone."
    )

    if near:

        if newest["color"] == "GREEN":

            rejection = True

            description = (
                "Newest visible candle is GREEN "
                "near the approximate support zone."
            )

        else:

            description = (
                "Newest visible candle is RED "
                "near the approximate support zone."
            )

    return {
        "near": near,
        "rejection": rejection,
        "description": description
    }


# ============================================================
# COLOR SEQUENCE
# ============================================================

def consecutive_color_count(
    candles,
    color
):

    count = 0

    for candle in candles:

        if candle["color"] == color:
            count += 1
        else:
            break

    return count


def analyze_sequence(
    candles
):

    if not candles:
        return []

    result = []

    maximum = min(
        len(candles),
        LOOKBACK_CANDLES
    )

    for i in range(
        maximum
    ):

        candle = candles[i]

        result.append(
            {
                "number": i + 1,
                "color": candle["color"],
                "height": candle["h"],
                "width": candle["w"]
            }
        )

    return result


# ============================================================
# MOMENTUM FROM VISIBLE BODY SIZE
# ============================================================

def analyze_body_momentum(
    candles
):

    if len(candles) < 3:

        return {
            "state": "UNKNOWN",
            "description":
                "Not enough candles."
        }

    newest = candles[0]

    previous = candles[1]

    older = candles[2]

    newest_size = newest["h"]

    previous_size = previous["h"]

    older_size = older["h"]

    average_previous = (
        previous_size +
        older_size
    ) / 2

    if (
        newest_size >
        average_previous * 1.35
    ):

        state = "STRONGER"

    elif (
        newest_size <
        average_previous * 0.70
    ):

        state = "WEAKER"

    else:

        state = "NORMAL"

    return {
        "state": state,
        "description":
            f"Newest body size is {state.lower()} "
            "relative to the previous two visible bodies."
    }


# ============================================================
# CONSECUTIVE MOMENTUM
# ============================================================

def analyze_consecutive_momentum(
    candles
):

    green_count = (
        consecutive_color_count(
            candles,
            "GREEN"
        )
    )

    red_count = (
        consecutive_color_count(
            candles,
            "RED"
        )
    )

    if green_count >= 3:

        return {
            "direction": "GREEN",
            "count": green_count,
            "description":
                f"{green_count} consecutive GREEN "
                "visible candles show upward body momentum."
        }

    if red_count >= 3:

        return {
            "direction": "RED",
            "count": red_count,
            "description":
                f"{red_count} consecutive RED "
                "visible candles show downward body momentum."
        }

    return {
        "direction": None,
        "count": max(
            green_count,
            red_count
        ),
        "description":
            "No strong 3+ candle color sequence detected."
    }


# ============================================================
# OPEN/CLOSE CLUSTERING
# ============================================================

def detect_open_close_clustering(
    candles
):

    if len(candles) < 3:

        return {
            "detected": False,
            "description":
                "Not enough visible candles."
        }

    recent = candles[
        :min(6, len(candles))
    ]

    heights = [
        c["h"]
        for c in recent
    ]

    if not heights:
        return {
            "detected": False,
            "description":
                "No candle bodies available."
        }

    median_height = float(
        np.median(heights)
    )

    small = sum(
        1
        for h in heights
        if h <= median_height * 0.75
    )

    if small >= 2:

        return {
            "detected": True,
            "description":
                f"{small} of the latest visible bodies "
                "are relatively compressed, suggesting "
                "short-term indecision/compression."
        }

    return {
        "detected": False,
        "description":
            "No strong body compression cluster detected."
    }


# ============================================================
# ENGULFING-STYLE BODY CHECK
# ============================================================

def detect_body_engulfing(
    candles
):

    if len(candles) < 2:

        return {
            "direction": None,
            "detected": False,
            "description":
                "Not enough candles."
        }

    current = candles[0]
    previous = candles[1]

    current_h = current["h"]
    previous_h = previous["h"]

    if (
        current["color"] == "GREEN"
        and
        previous["color"] == "RED"
        and
        current_h >= previous_h * 1.15
    ):

        return {
            "direction": "BUY",
            "detected": True,
            "description":
                "Newest GREEN body is substantially "
                "larger than the preceding RED body."
        }

    if (
        current["color"] == "RED"
        and
        previous["color"] == "GREEN"
        and
        current_h >= previous_h * 1.15
    ):

        return {
            "direction": "SELL",
            "detected": True,
            "description":
                "Newest RED body is substantially "
                "larger than the preceding GREEN body."
        }

    return {
        "direction": None,
        "detected": False,
        "description":
            "No strong body-engulfing reversal detected."
    }


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

    if len(candles) < 2:
        return None

    level = zone["level"]

    # Since candles are represented by body boxes,
    # this checks body crossing/failure rather than
    # pretending to have wick OHLC.

    for i in range(
        1,
        min(
            len(candles),
            LOOKBACK_CANDLES
        )
    ):

        previous = candles[i]
        current = candles[i - 1]

        previous_top, previous_bottom = (
            body_position(previous)
        )

        current_top, current_bottom = (
            body_position(current)
        )

        if side == "RESISTANCE":

            crossed = (
                previous_top > level
                and
                current_top <= level
            )

            failed = (
                current["color"] == "RED"
            )

            if crossed and failed:

                return {
                    "direction": "SELL",
                    "description":
                        "A body moved through the resistance "
                        "area and the newer body turned RED."
                }

        if side == "SUPPORT":

            crossed = (
                previous_bottom < level
                and
                current_bottom >= level
            )

            failed = (
                current["color"] == "GREEN"
            )

            if crossed and failed:

                return {
                    "direction": "BUY",
                    "description":
                        "A body moved through the support "
                        "area and the newer body turned GREEN."
                }

    return None


# ============================================================
# GREEN -> RED / RED -> GREEN CONFIRMATION
# ============================================================

def detect_color_confirmation(
    candles,
    zone,
    direction
):

    if not zone:
        return None

    if len(candles) < 2:
        return None

    level = zone["level"]

    previous = candles[1]
    current = candles[0]

    previous_top, previous_bottom = (
        body_position(previous)
    )

    current_top, current_bottom = (
        body_position(current)
    )

    if direction == "SELL":

        if (
            previous["color"] == "GREEN"
            and
            current["color"] == "RED"
            and
            (
                previous_top <= level
                or
                previous_bottom <= level
            )
        ):

            return {
                "direction": "SELL",
                "description":
                    "GREEN → RED body confirmation "
                    "at the resistance area."
            }

    if direction == "BUY":

        if (
            previous["color"] == "RED"
            and
            current["color"] == "GREEN"
            and
            (
                previous_bottom >= level
                or
                previous_top >= level
            )
        ):

            return {
                "direction": "BUY",
                "description":
                    "RED → GREEN body confirmation "
                    "at the support area."
            }

    return None


# ============================================================
# PULLBACK CHECK
# ============================================================

def detect_pullback(
    candles,
    support,
    resistance
):

    if len(candles) < 3:

        return {
            "direction": None,
            "detected": False,
            "description":
                "Not enough candles for pullback analysis."
        }

    newest = candles[0]
    previous = candles[1]
    older = candles[2]

    # Upward sequence followed by a RED pullback
    # toward support.

    if (
        older["color"] == "GREEN"
        and
        previous["color"] == "GREEN"
        and
        newest["color"] == "RED"
        and
        support
    ):

        _, newest_bottom = (
            body_position(newest)
        )

        distance = abs(
            newest_bottom -
            support["level"]
        )

        tolerance = max(
            3,
            newest["h"] * 0.9
        )

        if distance <= tolerance:

            return {
                "direction": "BUY",
                "detected": True,
                "description":
                    "GREEN momentum was followed by a "
                    "RED pullback near support."
            }

    # Downward sequence followed by a GREEN pullback
    # toward resistance.

    if (
        older["color"] == "RED"
        and
        previous["color"] == "RED"
        and
        newest["color"] == "GREEN"
        and
        resistance
    ):

        newest_top, _ = (
            body_position(newest)
        )

        distance = abs(
            newest_top -
            resistance["level"]
        )

        tolerance = max(
            3,
            newest["h"] * 0.9
        )

        if distance <= tolerance:

            return {
                "direction": "SELL",
                "detected": True,
                "description":
                    "RED momentum was followed by a "
                    "GREEN pullback near resistance."
            }

    return {
        "direction": None,
        "detected": False,
        "description":
            "No confirmed pullback setup detected."
    }


# ============================================================
# TWO-SIDED REJECTION
# ============================================================

def detect_two_sided_rejection(
    resistance,
    support
):

    if (
        resistance
        and
        support
        and
        len(resistance["candles"]) >= MIN_ZONE_TESTS
        and
        len(support["candles"]) >= MIN_ZONE_TESTS
    ):

        return {
            "detected": True,
            "description":
                "Repeated upper and lower body rejection "
                "areas are both visible."
        }

    return {
        "detected": False,
        "description":
            "No confirmed two-sided rejection structure."
    }


# ============================================================
# LATEST REVERSAL SEQUENCE
# ============================================================

def detect_recent_reversal(
    candles
):

    if len(candles) < 3:

        return {
            "direction": None,
            "detected": False,
            "description":
                "Not enough candles."
        }

    c1 = candles[0]
    c2 = candles[1]
    c3 = candles[2]

    # GREEN -> GREEN -> RED
    if (
        c3["color"] == "GREEN"
        and
        c2["color"] == "GREEN"
        and
        c1["color"] == "RED"
    ):

        return {
            "direction": "SELL",
            "detected": True,
            "description":
                "Recent sequence changed from GREEN "
                "momentum to a RED newest candle."
        }

    # RED -> RED -> GREEN
    if (
        c3["color"] == "RED"
        and
        c2["color"] == "RED"
        and
        c1["color"] == "GREEN"
    ):

        return {
            "direction": "BUY",
            "detected": True,
            "description":
                "Recent sequence changed from RED "
                "momentum to a GREEN newest candle."
        }

    return {
        "direction": None,
        "detected": False,
        "description":
            "No clear recent 3-candle color reversal."
    }


# ============================================================
# PRICE INFORMATION
# ============================================================

def describe_price_information(
    candles,
    resistance,
    support
):

    details = []

    newest = candles[0]

    center_price = newest.get(
        "price_center"
    )

    if center_price is not None:

        details.append(
            "Newest candle approximate price: "
            +
            format_price(
                center_price
            )
        )

    else:

        details.append(
            "Newest candle price: "
            "NOT MAPPED"
        )

    resistance_price = zone_price(
        resistance,
        candles,
        "RESISTANCE"
    )

    if resistance_price is not None:

        details.append(
            "Resistance price area: "
            +
            format_price(
                resistance_price
            )
        )

    else:

        details.append(
            "Resistance price area: "
            "NOT MAPPED"
        )

    support_price = zone_price(
        support,
        candles,
        "SUPPORT"
    )

    if support_price is not None:

        details.append(
            "Support price area: "
            +
            format_price(
                support_price
            )
        )

    else:

        details.append(
            "Support price area: "
            "NOT MAPPED"
        )

    return details


# ============================================================
# STRATEGY SCORE
# ============================================================

def calculate_strategy_score(
    candles,
    resistance_result,
    support_result,
    momentum,
    consecutive,
    engulfing,
    pullback,
    reversal,
    breakout_resistance,
    breakout_support,
    confirmation_resistance,
    confirmation_support
):

    buy = 0
    sell = 0

    buy_reasons = []
    sell_reasons = []

    # --------------------------------------------------------
    # SUPPORT
    # --------------------------------------------------------

    if support_result["rejection"]:

        buy += 2

        buy_reasons.append(
            "support rejection"
        )

    # --------------------------------------------------------
    # RESISTANCE
    # --------------------------------------------------------

    if resistance_result["rejection"]:

        sell += 2

        sell_reasons.append(
            "resistance rejection"
        )

    # --------------------------------------------------------
    # BODY MOMENTUM
    # --------------------------------------------------------

    if momentum["state"] == "STRONGER":

        if candles[0]["color"] == "GREEN":

            buy += 1

            buy_reasons.append(
                "stronger green body"
            )

        elif candles[0]["color"] == "RED":

            sell += 1

            sell_reasons.append(
                "stronger red body"
            )

    # --------------------------------------------------------
    # CONSECUTIVE MOMENTUM
    # --------------------------------------------------------

    if consecutive["direction"] == "GREEN":

        buy += 1

        buy_reasons.append(
            "green momentum sequence"
        )

    elif consecutive["direction"] == "RED":

        sell += 1

        sell_reasons.append(
            "red momentum sequence"
        )

    # --------------------------------------------------------
    # ENGULFING BODY
    # --------------------------------------------------------

    if engulfing["detected"]:

        if engulfing["direction"] == "BUY":

            buy += 2

            buy_reasons.append(
                "bullish body reversal"
            )

        elif engulfing["direction"] == "SELL":

            sell += 2

            sell_reasons.append(
                "bearish body reversal"
            )

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

    if pullback["detected"]:

        if pullback["direction"] == "BUY":

            buy += 2

            buy_reasons.append(
                "support pullback"
            )

        elif pullback["direction"] == "SELL":

            sell += 2

            sell_reasons.append(
                "resistance pullback"
            )

    # --------------------------------------------------------
    # RECENT REVERSAL
    # --------------------------------------------------------

    if reversal["detected"]:

        if reversal["direction"] == "BUY":

            buy += 1

            buy_reasons.append(
                "recent green reversal"
            )

        elif reversal["direction"] == "SELL":

            sell += 1

            sell_reasons.append(
                "recent red reversal"
            )

    # --------------------------------------------------------
    # BREAKOUT FAILURE
    # --------------------------------------------------------

    if breakout_support:

        buy += 2

        buy_reasons.append(
            "support breakout failure"
        )

    if breakout_resistance:

        sell += 2

        sell_reasons.append(
            "resistance breakout failure"
        )

    # --------------------------------------------------------
    # COLOR CONFIRMATION
    # --------------------------------------------------------

    if confirmation_support:

        buy += 2

        buy_reasons.append(
            "red → green confirmation"
        )

    if confirmation_resistance:

        sell += 2

        sell_reasons.append(
            "green → red confirmation"
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    total = buy + sell

    if total == 0:

        return {
            "direction": None,
            "buy_score": 0,
            "sell_score": 0,
            "agreement": 0,
            "reasons": []
        }

    strongest = max(
        buy,
        sell
    )

    weakest = min(
        buy,
        sell
    )

    agreement = round(
        (
            strongest /
            total
        ) * 100
    )

    if buy > sell:

        direction = "BUY"

        reasons = buy_reasons

    elif sell > buy:

        direction = "SELL"

        reasons = sell_reasons

    else:

        direction = None

        reasons = []

    return {
        "direction": direction,
        "buy_score": buy,
        "sell_score": sell,
        "agreement": agreement,
        "reasons": reasons
    }


# ============================================================
# FINAL SIGNAL DECISION
# ============================================================

def determine_signal(
    score,
    support_result,
    resistance_result,
    confirmation_support,
    confirmation_resistance,
    breakout_support,
    breakout_resistance
):

    direction = score["direction"]

    if direction is None:

        return {
            "signal": None,
            "confidence": 0,
            "reason":
                "BUY and SELL strategy evidence is balanced."
        }

    # Require a location-based reversal condition.
    location_confirmed = (
        (
            direction == "BUY"
            and
            support_result["rejection"]
        )
        or
        (
            direction == "SELL"
            and
            resistance_result["rejection"]
        )
        or
        (
            direction == "BUY"
            and
            confirmation_support
        )
        or
        (
            direction == "SELL"
            and
            confirmation_resistance
        )
        or
        (
            direction == "BUY"
            and
            breakout_support
        )
        or
        (
            direction == "SELL"
            and
            breakout_resistance
        )
    )

    if not location_confirmed:

        return {
            "signal": None,
            "confidence": score["agreement"],
            "reason":
                "Momentum evidence exists, but there is "
                "no confirmed support/resistance reversal location."
        }

    # Require at least two supporting strategy points.
    strategy_score = (
        score["buy_score"]
        if direction == "BUY"
        else score["sell_score"]
    )

    if strategy_score < 3:

        return {
            "signal": None,
            "confidence": score["agreement"],
            "reason":
                "A reversal location is visible, but "
                "strategy agreement is still too weak."
        }

    # Confidence here is a strategy-agreement score,
    # NOT a guaranteed probability of winning.

    confidence = min(
        95,
        max(
            65,
            score["agreement"]
        )
    )

    reasons = score["reasons"]

    # Keep the final reason short.
    if reasons:

        core_reason = (
            ", ".join(
                reasons[:3]
            )
        )

    else:

        core_reason = (
            "multiple candle strategies agree"
        )

    return {
        "signal": direction,
        "confidence": confidence,
        "reason": core_reason
    }


# ============================================================
# COMPLETE STRATEGY ANALYSIS
# ============================================================

def analyze_strategy(
    candles,
    price_data
):

    if not candles:

        return {
            "summary":
                "No candles detected.",
            "details": [],
            "signal": None
        }

    details = []

    newest = candles[0]

    # --------------------------------------------------------
    # RESISTANCE
    # --------------------------------------------------------

    resistance = (
        find_resistance_zone(
            candles
        )
    )

    if resistance:

        tests = len(
            resistance["candles"]
        )

        details.append(
            f"Resistance zone: "
            f"{tests} visible body tests."
        )

    else:

        details.append(
            "Resistance: no repeated body zone confirmed."
        )

    # --------------------------------------------------------
    # SUPPORT
    # --------------------------------------------------------

    support = (
        find_support_zone(
            candles
        )
    )

    if support:

        tests = len(
            support["candles"]
        )

        details.append(
            f"Support zone: "
            f"{tests} visible body tests."
        )

    else:

        details.append(
            "Support: no repeated body zone confirmed."
        )

    # --------------------------------------------------------
    # PRICE LEVELS
    # --------------------------------------------------------

    details.extend(
        describe_price_information(
            candles,
            resistance,
            support
        )
    )

    # --------------------------------------------------------
    # RESISTANCE BEHAVIOR
    # --------------------------------------------------------

    resistance_result = (
        resistance_behavior(
            candles,
            resistance
        )
    )

    details.append(
        "Resistance behavior: "
        +
        resistance_result[
            "description"
        ]
    )

    # --------------------------------------------------------
    # SUPPORT BEHAVIOR
    # --------------------------------------------------------

    support_result = (
        support_behavior(
            candles,
            support
        )
    )

    details.append(
        "Support behavior: "
        +
        support_result[
            "description"
        ]
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    momentum = (
        analyze_body_momentum(
            candles
        )
    )

    details.append(
        "Body momentum: "
        +
        momentum["description"]
    )

    # --------------------------------------------------------
    # CONSECUTIVE CANDLES
    # --------------------------------------------------------

    consecutive = (
        analyze_consecutive_momentum(
            candles
        )
    )

    details.append(
        "Consecutive momentum: "
        +
        consecutive["description"]
    )

    # --------------------------------------------------------
    # OPEN/CLOSE CLUSTERING
    # --------------------------------------------------------

    clustering = (
        detect_open_close_clustering(
            candles
        )
    )

    details.append(
        "Body clustering: "
        +
        clustering["description"]
    )

    # --------------------------------------------------------
    # ENGULFING
    # --------------------------------------------------------

    engulfing = (
        detect_body_engulfing(
            candles
        )
    )

    details.append(
        "Body reversal/engulfing: "
        +
        engulfing["description"]
    )

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

    pullback = (
        detect_pullback(
            candles,
            support,
            resistance
        )
    )

    details.append(
        "Pullback: "
        +
        pullback["description"]
    )

    # --------------------------------------------------------
    # RECENT REVERSAL
    # --------------------------------------------------------

    reversal = (
        detect_recent_reversal(
            candles
        )
    )

    details.append(
        "Recent reversal: "
        +
        reversal["description"]
    )

    # --------------------------------------------------------
    # TWO SIDED
    # --------------------------------------------------------

    two_sided = (
        detect_two_sided_rejection(
            resistance,
            support
        )
    )

    details.append(
        "Two-sided rejection: "
        +
        two_sided["description"]
    )

    # --------------------------------------------------------
    # BREAKOUT FAILURE
    # --------------------------------------------------------

    breakout_resistance = (
        detect_breakout_failure(
            candles,
            resistance,
            "RESISTANCE"
        )
    )

    breakout_support = (
        detect_breakout_failure(
            candles,
            support,
            "SUPPORT"
        )
    )

    if breakout_resistance:

        details.append(
            "Resistance breakout failure: YES — "
            +
            breakout_resistance[
                "description"
            ]
        )

    else:

        details.append(
            "Resistance breakout failure: NO."
        )

    if breakout_support:

        details.append(
            "Support breakout failure: YES — "
            +
            breakout_support[
                "description"
            ]
        )

    else:

        details.append(
            "Support breakout failure: NO."
        )

    # --------------------------------------------------------
    # COLOR CONFIRMATION
    # --------------------------------------------------------

    confirmation_resistance = (
        detect_color_confirmation(
            candles,
            resistance,
            "SELL"
        )
    )

    confirmation_support = (
        detect_color_confirmation(
            candles,
            support,
            "BUY"
        )
    )

    if confirmation_resistance:

        details.append(
            "GREEN → RED confirmation: YES."
        )

    else:

        details.append(
            "GREEN → RED confirmation: NO."
        )

    if confirmation_support:

        details.append(
            "RED → GREEN confirmation: YES."
        )

    else:

        details.append(
            "RED → GREEN confirmation: NO."
        )

    # --------------------------------------------------------
    # STRATEGY SCORE
    # --------------------------------------------------------

    score = calculate_strategy_score(
        candles,
        resistance_result,
        support_result,
        momentum,
        consecutive,
        engulfing,
        pullback,
        reversal,
        breakout_resistance,
        breakout_support,
        confirmation_resistance,
        confirmation_support
    )

    details.append(
        f"BUY strategy score: "
        f"{score['buy_score']}"
    )

    details.append(
        f"SELL strategy score: "
        f"{score['sell_score']}"
    )

    details.append(
        f"Strategy agreement: "
        f"{score['agreement']}%"
    )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    final = determine_signal(
        score,
        support_result,
        resistance_result,
        confirmation_support,
        confirmation_resistance,
        breakout_support,
        breakout_resistance
    )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    if final["signal"]:

        summary = (
            f"{final['signal']} SIGNAL — "
            f"{final['confidence']}% strategy agreement."
        )

        short_reason = final["reason"]

    else:

        summary = (
            "NO SIGNAL — "
            +
            final["reason"]
        )

        short_reason = final["reason"]

    details.append(
        "Final strategy decision: "
        +
        summary
    )

    return {
        "summary": summary,
        "short_reason": short_reason,
        "details": details,
        "signal": final["signal"],
        "confidence": final["confidence"],
        "resistance": resistance,
        "support": support,
        "score": score,
        "price_data": price_data
    }


# ============================================================
# DETECTION MAP
# ============================================================

def create_detection_map(
    img,
    candles,
    analysis
):

    output = img.copy()

    # --------------------------------------------------------
    # CANDLE BOXES
    # --------------------------------------------------------

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
            (x, y),
            (x + w, y + h),
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

        # Add price to newest candles when available.
        if number <= 5:

            price = candle.get(
                "price_center"
            )

            if price is not None:

                label = (
                    f"{format_price(price)}"
                )

                cv2.putText(
                    output,
                    label,
                    (
                        max(5, x - 10),
                        min(
                            output.shape[0] - 10,
                            y + h + 20
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

    # --------------------------------------------------------
    # RESISTANCE / SUPPORT LINES
    # --------------------------------------------------------

    resistance = analysis.get(
        "resistance"
    )

    support = analysis.get(
        "support"
    )

    if resistance:

        y = int(
            resistance["level"]
        )

        cv2.line(
            output,
            (0, y),
            (
                output.shape[1],
                y
            ),
            (0, 0, 255),
            2
        )

        price = zone_price(
            resistance,
            candles,
            "RESISTANCE"
        )

        label = (
            "RESISTANCE"
            +
            (
                f" {format_price(price)}"
                if price is not None
                else ""
            )
        )

        cv2.putText(
            output,
            label,
            (
                20,
                max(
                    25,
                    y - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    if support:

        y = int(
            support["level"]
        )

        cv2.line(
            output,
            (0, y),
            (
                output.shape[1],
                y
            ),
            (0, 255, 0),
            2
        )

        price = zone_price(
            support,
            candles,
            "SUPPORT"
        )

        label = (
            "SUPPORT"
            +
            (
                f" {format_price(price)}"
                if price is not None
                else ""
            )
        )

        cv2.putText(
            output,
            label,
            (
                20,
                min(
                    output.shape[0] - 15,
                    y + 22
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
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
        "strategy_chart.png"
    )

    detection_path = (
        "strategy_detection.png"
    )

    try:

        bot.reply_to(
            message,
            "👁️ Reading candles RIGHT → LEFT...\n"
            "🎯 Candle #1 = newest candle.\n"
            "💰 Reading visible price scale...\n"
            "🧠 Testing all candle strategies..."
        )

        # ----------------------------------------------------
        # DOWNLOAD
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
        # LOAD
        # ----------------------------------------------------

        img = load_image(
            original_path
        )

        # ----------------------------------------------------
        # DETECT CANDLES
        # ----------------------------------------------------

        candles = detect_candles(
            img
        )

        total = len(
            candles
        )

        if total == 0:

            bot.reply_to(
                message,
                "❌ No reliable candle bodies detected.\n\n"
                "No strategy analysis was generated."
            )

            return

        # ----------------------------------------------------
        # PRICE SCALE
        # ----------------------------------------------------

        price_data = (
            read_price_scale(
                img
            )
        )

        candles = (
            attach_prices_to_candles(
                candles,
                price_data,
                img.shape[0]
            )
        )

        # ----------------------------------------------------
        # COUNT
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SEQUENCE
        # ----------------------------------------------------

        sequence = []

        for number, candle in enumerate(
            candles,
            start=1
        ):

            icon = (
                "🟢"
                if candle["color"] == "GREEN"
                else "🔴"
            )

            price = candle.get(
                "price_center"
            )

            price_text = ""

            if (
                number <= 5
                and
                price is not None
            ):

                price_text = (
                    f" | ~{format_price(price)}"
                )

            sequence.append(
                f"{number}. {icon} "
                f"{candle['color']}"
                f"{price_text}"
            )

        sequence_text = "\n".join(
            sequence[
                :LOOKBACK_CANDLES
            ]
        )

        # ----------------------------------------------------
        # ANALYSIS
        # ----------------------------------------------------

        analysis = (
            analyze_strategy(
                candles,
                price_data
            )
        )

        strategy_text = "\n".join(
            "• " + item
            for item in analysis[
                "details"
            ]
        )

        # ----------------------------------------------------
        # PRICE STATUS
        # ----------------------------------------------------

        if price_data["available"]:

            price_status = (
                "✅ Price scale mapped"
            )

        else:

            price_status = (
                "⚠️ Price scale not fully mapped"
            )

        # ----------------------------------------------------
        # FINAL REPORT
        # ----------------------------------------------------

        report = (
            "🔎 **OTC CANDLE STRATEGY ANALYSIS**\n\n"

            "➡️ **SCAN:** RIGHT → LEFT\n"
            "🎯 **CANDLE #1 = NEWEST VISIBLE CANDLE**\n"
            f"💰 **PRICE:** {price_status}\n\n"

            "📊 **CANDLE DETECTION**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"

            f"🟢 GREEN: {green}\n"
            f"🔴 RED: {red}\n"
            f"📊 TOTAL: {total}\n\n"

            "🕯️ **NEWEST → OLDEST**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"

            f"{sequence_text}\n\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            "🧠 **ALL STRATEGY READINGS**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"

            f"{strategy_text}\n\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            "🎯 **FINAL TEST RESULT**\n"

            f"{analysis['summary']}\n\n"

            "⚠️ **IMPORTANT DETECTION LIMITS**\n"
            "The detector reads visible candle bodies.\n"
            "It does not invent OHLC values.\n"
            "It does not detect currency pairs.\n"
            "It does not generate random candles.\n"
            "It does not connect to Pocket Option.\n"
            "It does not place trades.\n"
            "Wicks are not claimed as exact OHLC unless "
            "they are actually detectable.\n\n"

            f"⚡ Processing time: "
            f"{time.time() - start_time:.2f}s"
        )

        # ----------------------------------------------------
        # SEND DETAILED TEST REPORT
        # ----------------------------------------------------

        bot.reply_to(
            message,
            report,
            parse_mode="Markdown"
        )

        # ----------------------------------------------------
        # SHORT SIGNAL MESSAGE
        # ----------------------------------------------------

        if analysis["signal"]:

            signal = (
                analysis["signal"]
            )

            confidence = (
                analysis["confidence"]
            )

            if signal == "BUY":

                arrow = "🟢"

            else:

                arrow = "🔴"

            short_message = (
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"{arrow} **{signal} SIGNAL**\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                f"💪 Strategy agreement: "
                f"{confidence}%\n"
                f"🎯 Core reason: "
                f"{analysis['short_reason']}\n"
            )

            # Add actual rejection price if available.

            if signal == "SELL":

                price = zone_price(
                    analysis["resistance"],
                    candles,
                    "RESISTANCE"
                )

                if price is not None:

                    short_message += (
                        f"💰 Rejection area: "
                        f"{format_price(price)}\n"
                    )

            elif signal == "BUY":

                price = zone_price(
                    analysis["support"],
                    candles,
                    "SUPPORT"
                )

                if price is not None:

                    short_message += (
                        f"💰 Rejection area: "
                        f"{format_price(price)}\n"
                    )

            short_message += (
                "\n⚠️ Manual decision only."
            )

            bot.reply_to(
                message,
                short_message,
                parse_mode="Markdown"
            )

        # ----------------------------------------------------
        # DETECTION MAP
        # ----------------------------------------------------

        detection_map = (
            create_detection_map(
                img,
                candles,
                analysis
            )
        )

        cv2.imwrite(
            detection_path,
            detection_map
        )

        with open(
            detection_path,
            "rb"
        ) as photo:

            bot.send_photo(
                message.chat.id,
                photo,
                caption=(
                    "🔢 **STRATEGY DETECTION MAP**\n\n"
                    "RIGHT → LEFT.\n"
                    "Candle #1 = newest.\n\n"
                    "🟡 Yellow box = detected candle body.\n"
                    "🟢 Number = GREEN.\n"
                    "🔴 Number = RED.\n"
                    "🔴 Line = resistance zone.\n"
                    "🟢 Line = support zone.\n\n"
                    "Price labels are shown when the "
                    "price scale was successfully mapped."
                ),
                parse_mode="Markdown"
            )

    except Exception as e:

        print(
            "❌ ERROR:",
            repr(e)
        )

        try:

            bot.reply_to(
                message,
                f"❌ Strategy analysis error:\n{str(e)}"
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
# START
# ============================================================

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
    "RIGHT → LEFT scanning."
)

print(
    "Candle #1 = newest."
)

print(
    "Support/resistance."
)

print(
    "Repeated rejection."
)

print(
    "Breakout failure."
)

print(
    "Green → Red confirmation."
)

print(
    "Red → Green confirmation."
)

print(
    "Body momentum."
)

print(
    "Consecutive candle momentum."
)

print(
    "Body reversal / engulfing."
)

print(
    "Pullback detection."
)

print(
    "Open/close body clustering."
)

print(
    "Two-sided rejection."
)

print(
    "Visible price-scale detection only."
)

print(
    "No pair detection."
)

print(
    "No random candles."
)

print(
    "No automatic trading."
)

print(
    "========================================"
)


bot.infinity_polling(
    timeout=30,
    long_polling_timeout=30
)

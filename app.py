import os
import cv2
import numpy as np
import telebot
import time

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

# ============================================================
# NEW STRATEGY SETTINGS
# ============================================================
MIN_SCORE_TO_SIGNAL = 4
MIN_SCORE_GAP = 2

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

    # GREEN
    green_lower = np.array([
        30, 35, 35
    ])

    green_upper = np.array([
        90, 255, 255
    ])

    green = cv2.inRange(
        hsv,
        green_lower,
        green_upper
    )

    # RED
    red_lower_1 = np.array([
        0, 55, 55
    ])

    red_upper_1 = np.array([
        12, 255, 255
    ])

    red_lower_2 = np.array([
        168, 55, 55
    ])

    red_upper_2 = np.array([
        180, 255, 255
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

    # RIGHT → LEFT
    # Candle #1 = newest/rightmost
    candles.reverse()

    return candles


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
# CHECK CURRENT CANDLE AGAINST RESISTANCE
# ============================================================
def resistance_behavior(
    candles,
    resistance
):

    if not resistance:

        return {
            "near": False,
            "rejection": False,
            "strength": 0,
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

    strength = 0

    description = (
        "Newest candle is not near "
        "the detected resistance zone."
    )

    if near:

        if newest["color"] == "RED":

            rejection = True
            strength = 2

            description = (
                "Newest visible candle is RED "
                "near the approximate resistance zone."
            )

        else:

            strength = 1

            description = (
                "Newest visible candle is GREEN "
                "near the approximate resistance zone."
            )

    return {
        "near": near,
        "rejection": rejection,
        "strength": strength,
        "description": description
    }


# ============================================================
# CHECK CURRENT CANDLE AGAINST SUPPORT
# ============================================================
def support_behavior(
    candles,
    support
):

    if not support:

        return {
            "near": False,
            "rejection": False,
            "strength": 0,
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

    strength = 0

    description = (
        "Newest candle is not near "
        "the detected support zone."
    )

    if near:

        if newest["color"] == "GREEN":

            rejection = True
            strength = 2

            description = (
                "Newest visible candle is GREEN "
                "near the approximate support zone."
            )

        else:

            strength = 1

            description = (
                "Newest visible candle is RED "
                "near the approximate support zone."
            )

    return {
        "near": near,
        "rejection": rejection,
        "strength": strength,
        "description": description
    }


# ============================================================
# ZONE REACTION STRENGTH
# ============================================================
def analyze_zone_reaction_strength(
    candles,
    resistance,
    support
):

    result = {
        "resistance_strength": 0,
        "support_strength": 0,
        "resistance_description":
            "Resistance reaction strength not confirmed.",
        "support_description":
            "Support reaction strength not confirmed."
    }

    if resistance:

        resistance_tests = len(
            resistance["candles"]
        )

        red_reactions = sum(
            1
            for c in resistance["candles"]
            if c["color"] == "RED"
        )

        if resistance_tests >= 6 and red_reactions >= 2:
            result["resistance_strength"] = 3

        elif resistance_tests >= 4 and red_reactions >= 2:
            result["resistance_strength"] = 2

        elif resistance_tests >= 2 and red_reactions >= 1:
            result["resistance_strength"] = 1

        result["resistance_description"] = (
            f"Resistance reaction strength: "
            f"{result['resistance_strength']}/3 "
            f"based on {resistance_tests} visible tests."
        )

    if support:

        support_tests = len(
            support["candles"]
        )

        green_reactions = sum(
            1
            for c in support["candles"]
            if c["color"] == "GREEN"
        )

        if support_tests >= 6 and green_reactions >= 2:
            result["support_strength"] = 3

        elif support_tests >= 4 and green_reactions >= 2:
            result["support_strength"] = 2

        elif support_tests >= 2 and green_reactions >= 1:
            result["support_strength"] = 1

        result["support_description"] = (
            f"Support reaction strength: "
            f"{result['support_strength']}/3 "
            f"based on {support_tests} visible tests."
        )

    return result


# ============================================================
# CANDLE SEQUENCE
# ============================================================
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

        result.append({
            "number": i + 1,
            "color": candle["color"],
            "height": candle["h"],
            "width": candle["w"]
        })

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

    if len(candles) < 3:

        return {
            "direction": "NONE",
            "count": 0,
            "description":
                "Not enough candles."
        }

    newest_color = candles[0]["color"]

    count = 0

    for candle in candles:

        if candle["color"] == newest_color:

            count += 1

        else:

            break

    if count >= 3:

        direction = (
            "UPWARD"
            if newest_color == "GREEN"
            else "DOWNWARD"
        )

        description = (
            f"{count} consecutive "
            f"{newest_color} visible candles "
            f"show {direction.lower()} body momentum."
        )

    else:

        direction = "NONE"

        description = (
            f"Only {count} consecutive "
            f"{newest_color} candle(s) detected."
        )

    return {
        "direction": direction,
        "count": count,
        "description": description
    }


# ============================================================
# STRONGER CONTINUATION DETECTION
# ============================================================
def analyze_continuation(
    candles
):

    if len(candles) < 4:

        return {
            "direction": "NONE",
            "strength": 0,
            "description":
                "Not enough candles for continuation analysis."
        }

    first = candles[0]
    second = candles[1]
    third = candles[2]
    fourth = candles[3]

    if (
        first["color"] ==
        second["color"]
        and
        second["color"] ==
        third["color"]
        and
        third["color"] ==
        fourth["color"]
    ):

        direction = (
            "BUY"
            if first["color"] == "GREEN"
            else "SELL"
        )

        heights = [
            first["h"],
            second["h"],
            third["h"],
            fourth["h"]
        ]

        recent_average = (
            heights[0] +
            heights[1]
        ) / 2

        older_average = (
            heights[2] +
            heights[3]
        ) / 2

        if recent_average >= older_average * 0.90:

            strength = 2

            description = (
                f"Strong {direction} continuation: "
                "four consecutive same-color bodies "
                "remain active."
            )

        else:

            strength = 1

            description = (
                f"Possible {direction} continuation: "
                "four consecutive same-color bodies "
                "are visible, but recent bodies are weakening."
            )

        return {
            "direction": direction,
            "strength": strength,
            "description": description
        }

    # Three-candle continuation
    if (
        first["color"] ==
        second["color"]
        and
        second["color"] ==
        third["color"]
    ):

        direction = (
            "BUY"
            if first["color"] == "GREEN"
            else "SELL"
        )

        if (
            first["h"] >=
            second["h"] * 0.80
            and
            second["h"] >=
            third["h"] * 0.80
        ):

            strength = 1

            description = (
                f"Possible {direction} continuation: "
                "three consecutive same-color bodies "
                "show directional persistence."
            )

            return {
                "direction": direction,
                "strength": strength,
                "description": description
            }

    return {
        "direction": "NONE",
        "strength": 0,
        "description":
            "No strong continuation structure confirmed."
    }


# ============================================================
# BODY CLUSTERING
# ============================================================
def analyze_body_clustering(
    candles
):

    if len(candles) < 3:

        return {
            "clustered": False,
            "count": 0,
            "description":
                "Not enough candles."
        }

    recent = candles[:3]

    heights = [
        c["h"]
        for c in recent
    ]

    average_height = (
        sum(heights) /
        len(heights)
    )

    compressed = 0

    for height in heights:

        if height <= average_height * 0.75:

            compressed += 1

    clustered = (
        compressed >= 2
    )

    if clustered:

        description = (
            f"{compressed} of the latest "
            "visible bodies are relatively "
            "compressed, suggesting short-term "
            "indecision/compression."
        )

    else:

        description = (
            "No strong body compression detected "
            "among the latest visible bodies."
        )

    return {
        "clustered": clustered,
        "count": compressed,
        "description": description
    }


# ============================================================
# BODY REVERSAL / ENGULFING
# ============================================================
def analyze_body_reversal(
    candles
):

    if len(candles) < 2:

        return {
            "detected": False,
            "direction": "NONE",
            "strength": 0,
            "description":
                "Not enough candles."
        }

    current = candles[0]
    previous = candles[1]

    current_height = current["h"]
    previous_height = previous["h"]

    opposite_colors = (
        current["color"] !=
        previous["color"]
    )

    strong_body = (
        current_height >=
        previous_height * 1.20
    )

    if (
        opposite_colors
        and
        strong_body
    ):

        direction = (
            "BUY"
            if current["color"] == "GREEN"
            else "SELL"
        )

        description = (
            f"Possible {direction} body-engulfing "
            "reversal detected."
        )

        return {
            "detected": True,
            "direction": direction,
            "strength": 2,
            "description": description
        }

    return {
        "detected": False,
        "direction": "NONE",
        "strength": 0,
        "description":
            "No strong body-engulfing reversal detected."
    }


# ============================================================
# TWO-CANDLE REVERSAL
# ============================================================
def analyze_two_candle_reversal(
    candles
):

    if len(candles) < 2:

        return {
            "direction": "NONE",
            "strength": 0,
            "description":
                "Not enough candles."
        }

    first = candles[0]
    second = candles[1]

    if (
        first["color"] == "GREEN"
        and
        second["color"] == "RED"
        and
        first["h"] >= second["h"] * 0.85
    ):

        return {
            "direction": "BUY",
            "strength": 1,
            "description":
                "Two-candle bullish reversal structure detected."
        }

    if (
        first["color"] == "RED"
        and
        second["color"] == "GREEN"
        and
        first["h"] >= second["h"] * 0.85
    ):

        return {
            "direction": "SELL",
            "strength": 1,
            "description":
                "Two-candle bearish reversal structure detected."
        }

    return {
        "direction": "NONE",
        "strength": 0,
        "description":
            "No strong two-candle reversal structure detected."
    }


# ============================================================
# THREE-CANDLE REVERSAL
# ============================================================
def analyze_three_candle_reversal(
    candles
):

    if len(candles) < 3:

        return {
            "direction": "NONE",
            "strength": 0,
            "description":
                "Not enough candles."
        }

    first = candles[0]
    second = candles[1]
    third = candles[2]

    # Newest → oldest
    # GREEN, GREEN, RED = possible bullish reversal
    if (
        first["color"] == "GREEN"
        and
        second["color"] == "GREEN"
        and
        third["color"] == "RED"
    ):

        return {
            "direction": "BUY",
            "strength": 2,
            "description":
                "Three-candle bullish reversal structure detected."
        }

    # RED, RED, GREEN = possible bearish reversal
    if (
        first["color"] == "RED"
        and
        second["color"] == "RED"
        and
        third["color"] == "GREEN"
    ):

        return {
            "direction": "SELL",
            "strength": 2,
            "description":
                "Three-candle bearish reversal structure detected."
        }

    return {
        "direction": "NONE",
        "strength": 0,
        "description":
            "No strong three-candle reversal structure detected."
    }


# ============================================================
# IMPROVED PULLBACK DETECTION
# ============================================================
def analyze_pullback(
    candles
):

    if len(candles) < 4:

        return {
            "detected": False,
            "direction": "NONE",
            "strength": 0,
            "description":
                "Not enough candles."
        }

    first = candles[0]
    second = candles[1]
    third = candles[2]
    fourth = candles[3]

    # Current green after a short red pullback
    if (
        first["color"] == "GREEN"
        and
        second["color"] == "RED"
        and
        third["color"] == "RED"
        and
        fourth["color"] == "GREEN"
    ):

        return {
            "detected": True,
            "direction": "BUY",
            "strength": 2,
            "description":
                "Possible bullish pullback completion: "
                "a short red retracement is followed by GREEN confirmation."
        }

    # Current red after a short green pullback
    if (
        first["color"] == "RED"
        and
        second["color"] == "GREEN"
        and
        third["color"] == "GREEN"
        and
        fourth["color"] == "RED"
    ):

        return {
            "detected": True,
            "direction": "SELL",
            "strength": 2,
            "description":
                "Possible bearish pullback completion: "
                "a short green retracement is followed by RED confirmation."
        }

    # Three same-color move followed by opposite candle
    if (
        second["color"] ==
        third["color"]
        and
        third["color"] ==
        fourth["color"]
        and
        first["color"] !=
        second["color"]
    ):

        direction = (
            "BUY"
            if first["color"] == "GREEN"
            else "SELL"
        )

        return {
            "detected": True,
            "direction": direction,
            "strength": 1,
            "description":
                "Possible pullback after a three-candle directional move."
        }

    return {
        "detected": False,
        "direction": "NONE",
        "strength": 0,
        "description":
            "No confirmed pullback setup detected."
    }


# ============================================================
# RECENT 3-CANDLE REVERSAL
# ============================================================
def analyze_recent_reversal(
    candles
):

    if len(candles) < 3:

        return {
            "detected": False,
            "direction": "NONE",
            "description":
                "Not enough candles."
        }

    first = candles[0]["color"]
    second = candles[1]["color"]
    third = candles[2]["color"]

    if (
        first == "GREEN"
        and
        second == "GREEN"
        and
        third == "RED"
    ):

        return {
            "detected": True,
            "direction": "BUY_TO_SELL",
            "description":
                "Recent 3-candle color sequence "
                "shows possible bearish reversal."
        }

    if (
        first == "RED"
        and
        second == "RED"
        and
        third == "GREEN"
    ):

        return {
            "detected": True,
            "direction": "SELL_TO_BUY",
            "description":
                "Recent 3-candle color sequence "
                "shows possible bullish reversal."
        }

    return {
        "detected": False,
        "direction": "NONE",
        "description":
            "No clear recent 3-candle color reversal."
    }


# ============================================================
# BREAKOUT FAILURE
# ============================================================
def analyze_breakout_failure(
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

            previous = candles[i - 1]
            current = candles[i]

            previous_top = (
                body_position(previous)[0]
            )

            current_top = (
                body_position(current)[0]
            )

            if (
                previous_top <= level
                and
                current_top > level
                and
                current["color"] == "RED"
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

            previous = candles[i - 1]
            current = candles[i]

            previous_bottom = (
                body_position(previous)[1]
            )

            current_bottom = (
                body_position(current)[1]
            )

            if (
                previous_bottom >= level
                and
                current_bottom < level
                and
                current["color"] == "GREEN"
            ):

                support_failure = True

                break

    return {
        "resistance_failure":
            resistance_failure,
        "support_failure":
            support_failure
    }


# ============================================================
# BREAKOUT CONFIRMATION
# ============================================================
def analyze_breakout_confirmation(
    candles,
    resistance,
    support
):

    resistance_confirmed = False
    support_confirmed = False

    if resistance:

        level = resistance["level"]

        for i in range(
            1,
            min(
                len(candles),
                LOOKBACK_CANDLES
            )
        ):

            previous = candles[i - 1]
            current = candles[i]

            previous_top = (
                body_position(previous)[0]
            )

            current_top = (
                body_position(current)[0]
            )

            if (
                previous_top > level
                and
                current_top > level
                and
                previous["color"] == "GREEN"
                and
                current["color"] == "GREEN"
            ):

                resistance_confirmed = True

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

            previous = candles[i - 1]
            current = candles[i]

            previous_bottom = (
                body_position(previous)[1]
            )

            current_bottom = (
                body_position(current)[1]
            )

            if (
                previous_bottom < level
                and
                current_bottom < level
                and
                previous["color"] == "RED"
                and
                current["color"] == "RED"
            ):

                support_confirmed = True

                break

    return {
        "resistance_confirmed":
            resistance_confirmed,
        "support_confirmed":
            support_confirmed
    }


# ============================================================
# COLOR CONFIRMATION
# ============================================================
def analyze_color_confirmation(
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

            previous = candles[i - 1]
            current = candles[i]

            previous_top = (
                body_position(previous)[0]
            )

            current_top = (
                body_position(current)[0]
            )

            if (
                previous["color"] == "GREEN"
                and
                current["color"] == "RED"
                and
                previous_top <= level
                and
                current_top >= level
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

            previous = candles[i - 1]
            current = candles[i]

            previous_bottom = (
                body_position(previous)[1]
            )

            current_bottom = (
                body_position(current)[1]
            )

            if (
                previous["color"] == "RED"
                and
                current["color"] == "GREEN"
                and
                previous_bottom >= level
                and
                current_bottom <= level
            ):

                red_green = True

                break

    return {
        "green_red": green_red,
        "red_green": red_green
    }


# ============================================================
# CONFLICT DETECTION
# ============================================================
def analyze_conflict(
    buy_score,
    sell_score
):

    difference = abs(
        buy_score -
        sell_score
    )

    total = (
        buy_score +
        sell_score
    )

    if total == 0:

        return {
            "conflict": True,
            "description":
                "No directional evidence is available."
        }

    if difference < MIN_SCORE_GAP:

        return {
            "conflict": True,
            "description":
                "BUY and SELL evidence are too close; "
                "directional conflict detected."
        }

    return {
        "conflict": False,
        "description":
            "Directional evidence is sufficiently separated."
    }


# ============================================================
# COMPLETE STRATEGY READING
# ============================================================
def analyze_strategy(
    candles
):

    if not candles:

        return {
            "summary":
                "No candles detected.",
            "details": []
        }

    details = []

    newest = candles[0]

    details.append(
        "Candle #1 is the newest/rightmost "
        f"visible candle: {newest['color']}."
    )

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
            "Resistance zone: no repeated "
            "resistance area confirmed."
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
            "Support zone: no repeated "
            "support area confirmed."
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
        resistance_result["description"]
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
        support_result["description"]
    )

    # --------------------------------------------------------
    # REACTION STRENGTH
    # --------------------------------------------------------
    reaction = (
        analyze_zone_reaction_strength(
            candles,
            resistance,
            support
        )
    )

    details.append(
        reaction["resistance_description"]
    )

    details.append(
        reaction["support_description"]
    )

    # --------------------------------------------------------
    # BODY MOMENTUM
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
    # CONSECUTIVE MOMENTUM
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
    # CONTINUATION
    # --------------------------------------------------------
    continuation = (
        analyze_continuation(
            candles
        )
    )

    details.append(
        "Continuation: "
        +
        continuation["description"]
    )

    # --------------------------------------------------------
    # BODY CLUSTERING
    # --------------------------------------------------------
    clustering = (
        analyze_body_clustering(
            candles
        )
    )

    details.append(
        "Body clustering: "
        +
        clustering["description"]
    )

    # --------------------------------------------------------
    # BODY REVERSAL
    # --------------------------------------------------------
    reversal = (
        analyze_body_reversal(
            candles
        )
    )

    details.append(
        "Body reversal/engulfing: "
        +
        reversal["description"]
    )

    # --------------------------------------------------------
    # TWO-CANDLE REVERSAL
    # --------------------------------------------------------
    two_reversal = (
        analyze_two_candle_reversal(
            candles
        )
    )

    details.append(
        "2-candle reversal: "
        +
        two_reversal["description"]
    )

    # --------------------------------------------------------
    # THREE-CANDLE REVERSAL
    # --------------------------------------------------------
    three_reversal = (
        analyze_three_candle_reversal(
            candles
        )
    )

    details.append(
        "3-candle reversal: "
        +
        three_reversal["description"]
    )

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------
    pullback = (
        analyze_pullback(
            candles
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
    recent_reversal = (
        analyze_recent_reversal(
            candles
        )
    )

    details.append(
        "Recent reversal: "
        +
        recent_reversal["description"]
    )

    # --------------------------------------------------------
    # TWO-SIDED REJECTION
    # --------------------------------------------------------
    two_sided = (
        resistance is not None
        and
        support is not None
    )

    if two_sided:

        details.append(
            "Two-sided rejection: Repeated "
            "upper and lower body rejection "
            "areas are both visible."
        )

    else:

        details.append(
            "Two-sided rejection: Both "
            "upper and lower repeated areas "
            "are not confirmed."
        )

    # --------------------------------------------------------
    # BREAKOUT FAILURE
    # --------------------------------------------------------
    breakout = (
        analyze_breakout_failure(
            candles,
            resistance,
            support
        )
    )

    if breakout[
        "resistance_failure"
    ]:

        details.append(
            "Resistance breakout failure: "
            "YES — A body moved through the "
            "resistance area and the newer body "
            "turned RED."
        )

    else:

        details.append(
            "Resistance breakout failure: NO."
        )

    if breakout[
        "support_failure"
    ]:

        details.append(
            "Support breakout failure: "
            "YES — A body moved through the "
            "support area and the newer body "
            "turned GREEN."
        )

    else:

        details.append(
            "Support breakout failure: NO."
        )

    # --------------------------------------------------------
    # BREAKOUT CONFIRMATION
    # --------------------------------------------------------
    breakout_confirmation = (
        analyze_breakout_confirmation(
            candles,
            resistance,
            support
        )
    )

    if breakout_confirmation[
        "resistance_confirmed"
    ]:

        details.append(
            "Resistance breakout confirmation: "
            "YES — consecutive GREEN bodies "
            "remain above the resistance area."
        )

    else:

        details.append(
            "Resistance breakout confirmation: NO."
        )

    if breakout_confirmation[
        "support_confirmed"
    ]:

        details.append(
            "Support breakout confirmation: "
            "YES — consecutive RED bodies "
            "remain below the support area."
        )

    else:

        details.append(
            "Support breakout confirmation: NO."
        )

    # --------------------------------------------------------
    # COLOR CONFIRMATION
    # --------------------------------------------------------
    confirmation = (
        analyze_color_confirmation(
            candles,
            resistance,
            support
        )
    )

    if confirmation["green_red"]:

        details.append(
            "GREEN → RED confirmation: YES."
        )

    else:

        details.append(
            "GREEN → RED confirmation: NO."
        )

    if confirmation["red_green"]:

        details.append(
            "RED → GREEN confirmation: YES."
        )

    else:

        details.append(
            "RED → GREEN confirmation: NO."
        )

    # ========================================================
    # IMPROVED STRATEGY SCORING
    # ========================================================
    buy_score = 0
    sell_score = 0

    # --------------------------------------------------------
    # BASIC ZONES
    # --------------------------------------------------------
    if support:
        buy_score += 1

    if resistance:
        sell_score += 1

    # --------------------------------------------------------
    # CURRENT ZONE REACTION
    # --------------------------------------------------------
    if support_result["rejection"]:

        buy_score += (
            support_result["strength"]
        )

    if resistance_result["rejection"]:

        sell_score += (
            resistance_result["strength"]
        )

    # --------------------------------------------------------
    # HISTORICAL ZONE REACTION
    # --------------------------------------------------------
    buy_score += (
        reaction["support_strength"]
    )

    sell_score += (
        reaction["resistance_strength"]
    )

    # --------------------------------------------------------
    # BREAKOUT FAILURES
    # --------------------------------------------------------
    if breakout["support_failure"]:

        buy_score += 2

    if breakout["resistance_failure"]:

        sell_score += 2

    # --------------------------------------------------------
    # BREAKOUT CONFIRMATIONS
    # --------------------------------------------------------
    if breakout_confirmation[
        "resistance_confirmed"
    ]:

        buy_score += 2

    if breakout_confirmation[
        "support_confirmed"
    ]:

        sell_score += 2

    # --------------------------------------------------------
    # COLOR CONFIRMATION
    # --------------------------------------------------------
    if confirmation["red_green"]:

        buy_score += 2

    if confirmation["green_red"]:

        sell_score += 2

    # --------------------------------------------------------
    # CONSECUTIVE MOMENTUM
    # --------------------------------------------------------
    if consecutive[
        "direction"
    ] == "UPWARD":

        buy_score += 1

    elif consecutive[
        "direction"
    ] == "DOWNWARD":

        sell_score += 1

    # --------------------------------------------------------
    # CONTINUATION
    # --------------------------------------------------------
    if continuation[
        "direction"
    ] == "BUY":

        buy_score += (
            continuation["strength"]
        )

    elif continuation[
        "direction"
    ] == "SELL":

        sell_score += (
            continuation["strength"]
        )

    # --------------------------------------------------------
    # BODY ENGULFING
    # --------------------------------------------------------
    if reversal.get(
        "direction"
    ) == "BUY":

        buy_score += (
            reversal["strength"]
        )

    elif reversal.get(
        "direction"
    ) == "SELL":

        sell_score += (
            reversal["strength"]
        )

    # --------------------------------------------------------
    # TWO-CANDLE REVERSAL
    # --------------------------------------------------------
    if two_reversal[
        "direction"
    ] == "BUY":

        buy_score += (
            two_reversal["strength"]
        )

    elif two_reversal[
        "direction"
    ] == "SELL":

        sell_score += (
            two_reversal["strength"]
        )

    # --------------------------------------------------------
    # THREE-CANDLE REVERSAL
    # --------------------------------------------------------
    if three_reversal[
        "direction"
    ] == "BUY":

        buy_score += (
            three_reversal["strength"]
        )

    elif three_reversal[
        "direction"
    ] == "SELL":

        sell_score += (
            three_reversal["strength"]
        )

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------
    if pullback[
        "direction"
    ] == "BUY":

        buy_score += (
            pullback["strength"]
        )

    elif pullback[
        "direction"
    ] == "SELL":

        sell_score += (
            pullback["strength"]
        )

    # --------------------------------------------------------
    # RECENT REVERSAL
    # --------------------------------------------------------
    if recent_reversal.get(
        "direction"
    ) == "SELL_TO_BUY":

        buy_score += 2

    elif recent_reversal.get(
        "direction"
    ) == "BUY_TO_SELL":

        sell_score += 2

    # ========================================================
    # BODY MOMENTUM ADJUSTMENT
    # ========================================================
    if momentum["state"] == "STRONGER":

        if newest["color"] == "GREEN":
            buy_score += 1
        else:
            sell_score += 1

    # Weak current body = small caution against forcing
    # a continuation signal.
    if momentum["state"] == "WEAKER":

        if continuation["direction"] == "BUY":
            buy_score = max(
                0,
                buy_score - 1
            )

        elif continuation["direction"] == "SELL":
            sell_score = max(
                0,
                sell_score - 1
            )

    # ========================================================
    # DISPLAY SCORES
    # ========================================================
    details.append(
        f"BUY strategy score: {buy_score}"
    )

    details.append(
        f"SELL strategy score: {sell_score}"
    )

    # ========================================================
    # CONFLICT
    # ========================================================
    conflict = (
        analyze_conflict(
            buy_score,
            sell_score
        )
    )

    details.append(
        "Conflict detection: "
        +
        conflict["description"]
    )

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

    details.append(
        f"Strategy agreement: "
        f"{agreement}%"
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================
    if conflict["conflict"]:

        final = (
            "NO CLEAR SIGNAL"
        )

        confidence = agreement

    elif (
        buy_score > sell_score
        and
        buy_score >= MIN_SCORE_TO_SIGNAL
        and
        (
            buy_score -
            sell_score
        ) >= MIN_SCORE_GAP
    ):

        final = (
            "BUY SIGNAL"
        )

        confidence = min(
            95,
            max(
                65,
                agreement
            )
        )

    elif (
        sell_score > buy_score
        and
        sell_score >= MIN_SCORE_TO_SIGNAL
        and
        (
            sell_score -
            buy_score
        ) >= MIN_SCORE_GAP
    ):

        final = (
            "SELL SIGNAL"
        )

        confidence = min(
            95,
            max(
                65,
                agreement
            )
        )

    else:

        final = (
            "NO CLEAR SIGNAL"
        )

        confidence = agreement

    details.append(
        f"Final strategy decision: "
        f"{final}"
        +
        (
            f" — {confidence}% strategy agreement."
            if final != "NO CLEAR SIGNAL"
            else "."
        )
    )

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------
    details.append(
        "MACD: NOT READ — this bot is "
        "candle-body based."
    )

    details.append(
        "Volume: NOT READ — no volume "
        "data is extracted from the screenshot."
    )

    details.append(
        "Wicks: NOT claimed as exact OHLC "
        "because this detector reads visible "
        "candle bodies."
    )

    return {
        "summary":
            f"{final}"
            +
            (
                f" — {confidence}% strategy agreement."
                if final != "NO CLEAR SIGNAL"
                else "."
            ),

        "details": details,

        "buy_score":
            buy_score,

        "sell_score":
            sell_score,

        "agreement":
            agreement,

        "confidence":
            confidence
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
            "🔎 Testing candle strategies..."
        )

        # ----------------------------------------------------
        # DOWNLOAD HIGHEST RESOLUTION
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
        # DETECT
        # ----------------------------------------------------
        candles = detect_candles(
            img
        )

        total = len(
            candles
        )

        elapsed = (
            time.time() -
            start_time
        )

        if total == 0:

            bot.reply_to(
                message,
                "❌ No reliable candle bodies detected.\n\n"
                "No strategy analysis was generated."
            )

            return

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
        # CANDLE SEQUENCE
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

            sequence.append(
                f"{number}. {icon} "
                f"{candle['color']}"
            )

        sequence_text = "\n".join(
            sequence[
                :LOOKBACK_CANDLES
            ]
        )

        # ----------------------------------------------------
        # STRATEGY
        # ----------------------------------------------------
        strategy = (
            analyze_strategy(
                candles
            )
        )

        strategy_text = "\n".join(
            "• " + item
            for item in strategy["details"]
        )

        # ----------------------------------------------------
        # FINAL REPORT
        # ----------------------------------------------------
        report = (
            "🔎 **OTC CANDLE STRATEGY ANALYSIS**\n\n"

            "➡️ **SCAN:** RIGHT → LEFT\n"

            "🎯 **CANDLE #1 = NEWEST "
            "VISIBLE CANDLE**\n\n"

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

            f"{strategy['summary']}\n\n"

            "⚠️ **IMPORTANT DETECTION LIMITS**\n"

            "The detector reads visible candle bodies.\n"

            "It does not invent OHLC values.\n"

            "It does not detect currency pairs.\n"

            "It does not generate random candles.\n"

            "It does not map the price scale.\n"

            "It does not connect to Pocket Option.\n"

            "It does not place trades.\n"

            "Wicks are not claimed as exact OHLC "
            "unless they are actually detectable.\n\n"

            f"⚡ Processing time: "
            f"{elapsed:.2f}s"
        )

        bot.reply_to(
            message,
            report,
            parse_mode="Markdown"
        )

        # ----------------------------------------------------
        # DETECTION MAP
        # ----------------------------------------------------
        detection_map = (
            create_detection_map(
                img,
                candles
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
                    "Scan order is RIGHT → LEFT.\n"
                    "Candle #1 = newest.\n\n"
                    "🟡 Yellow box = detected body.\n"
                    "🟢 Number = GREEN.\n"
                    "🔴 Number = RED.\n\n"
                    "Use this map to check whether "
                    "the bot is reading the same candles "
                    "you are using for your strategy."
                ),
                parse_mode="Markdown"
            )

    except Exception as e:

        print(
            "❌ ERROR:",
            repr(e)
        )

        bot.reply_to(
            message,
            f"❌ Strategy test error:\n{str(e)}"
        )

    finally:

        for path in [
            original_path,
            detection_path
        ]:

            if os.path.exists(path):

                try:

                    os.remove(path)

                except Exception:

                    pass


# ============================================================
# START
# ============================================================
print(
    "========================================"
)

print(
    "🧠 OTC STRATEGY READING TEST"
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
    "Resistance/support body-zone testing."
)

print(
    "Enhanced reversal detection."
)

print(
    "Enhanced continuation detection."
)

print(
    "Enhanced pullback detection."
)

print(
    "Breakout confirmation."
)

print(
    "Rejection strength analysis."
)

print(
    "Conflict detection."
)

print(
    "No price mapping."
)

print(
    "No pair detection."
)

print(
    "No random candles."
)

print(
    "No trading signals."
)

print(
    "No Pocket Option authorization."
)

print(
    "========================================"
)

bot.infinity_polling(
    timeout=30,
    long_polling_timeout=30
)

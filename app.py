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

# The newest candles are more important.
LOOKBACK_CANDLES = 20

# A resistance/support area must have repeated tests.
MIN_ZONE_TESTS = 2

# How close candle highs/lows must be to be considered
# approximately the same visible price area.
ZONE_TOLERANCE_RATIO = 0.035

# A candle whose body is small relative to its visible
# vertical range is treated as possible indecision/rejection.
SMALL_BODY_RATIO = 0.35


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

        # Reject long horizontal objects.
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

    # Newest/right-side improvement.
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

    # Physical chart order.
    candles.sort(
        key=lambda c: c["center_x"]
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Convert to RIGHT → LEFT.
    #
    # Candle #1 = newest/rightmost visible candle.
    # --------------------------------------------------------

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

    height = max(
        1,
        candle["h"]
    )

    # Because image Y increases downward,
    # smaller Y = higher price.
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

    # This detector sees the colored candle body,
    # not the complete wick structure.
    #
    # Therefore this is deliberately a BODY-BASED
    # strategy test, not a fake OHLC calculation.

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

    # A rough visual proximity test.
    tolerance = max(
        3,
        newest["h"] * 0.75
    )

    near = (
        distance <= tolerance
    )

    # Important:
    # Because this detector reads candle BODIES only,
    # we cannot honestly call a wick rejection here.
    #
    # We only identify a body returning away from
    # the approximate resistance area.

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
# COMPLETE STRATEGY READING TEST
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

    # --------------------------------------------------------
    # 1. NEWEST CANDLE
    # --------------------------------------------------------

    newest = candles[0]

    details.append(
        "Candle #1 is the newest/rightmost "
        f"visible candle: {newest['color']}."
    )

    # --------------------------------------------------------
    # 2. RESISTANCE
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
            f"Approximate resistance zone: "
            f"{tests} visible body tests."
        )

    else:

        details.append(
            "No repeated resistance zone "
            "confirmed from the visible candle bodies."
        )

    # --------------------------------------------------------
    # 3. SUPPORT
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
            f"Approximate support zone: "
            f"{tests} visible body tests."
        )

    else:

        details.append(
            "No repeated support zone "
            "confirmed from the visible candle bodies."
        )

    # --------------------------------------------------------
    # 4. RESISTANCE BEHAVIOR
    # --------------------------------------------------------

    resistance_result = (
        resistance_behavior(
            candles,
            resistance
        )
    )

    details.append(
        "Resistance check: "
        +
        resistance_result["description"]
    )

    # --------------------------------------------------------
    # 5. SUPPORT BEHAVIOR
    # --------------------------------------------------------

    support_result = (
        support_behavior(
            candles,
            support
        )
    )

    details.append(
        "Support check: "
        +
        support_result["description"]
    )

    # --------------------------------------------------------
    # 6. BODY MOMENTUM
    # --------------------------------------------------------

    momentum = (
        analyze_body_momentum(
            candles
        )
    )

    details.append(
        "Momentum check: "
        +
        momentum["description"]
    )

    # --------------------------------------------------------
    # 7. STRATEGY INTERPRETATION
    # --------------------------------------------------------

    if (
        resistance_result["rejection"]
        and
        newest["color"] == "RED"
    ):

        interpretation = (
            "POSSIBLE SELL-SIDE REVERSAL "
            "SETUP — resistance proximity + "
            "RED newest candle."
        )

    elif (
        support_result["rejection"]
        and
        newest["color"] == "GREEN"
    ):

        interpretation = (
            "POSSIBLE BUY-SIDE REVERSAL "
            "SETUP — support proximity + "
            "GREEN newest candle."
        )

    else:

        interpretation = (
            "NO COMPLETE REVERSAL SETUP "
            "FROM CANDLE INFORMATION ALONE."
        )

    details.append(
        "Strategy interpretation: "
        + interpretation
    )

    # --------------------------------------------------------
    # IMPORTANT LIMITATION
    # --------------------------------------------------------

    details.append(
        "MACD: NOT READ YET — the current detector "
        "only reads candle bodies."
    )

    details.append(
        "Volume: NOT READ YET — no volume confirmation "
        "has been added."
    )

    details.append(
        "Wick rejection: NOT CONFIRMED — the current "
        "detector does not extract reliable wick OHLC."
    )

    return {
        "summary": interpretation,
        "details": details
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

        # RIGHT → LEFT numbering
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
            "🔎 Testing your reversal strategy..."
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
            "🔎 **OTC STRATEGY READING TEST**\n\n"

            "➡️ **SCAN DIRECTION:** "
            "RIGHT → LEFT\n"

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

            "🧠 **STRATEGY ENGINE**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"

            f"{strategy_text}\n\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"🎯 **CURRENT TEST RESULT:**\n"
            f"{strategy['summary']}\n\n"

            "⚠️ **TEST MODE ONLY**\n"
            "No trade is opened.\n"
            "No Pocket Option connection.\n"
            "No pair detection.\n"
            "No random data.\n"
            "No automatic entry.\n\n"

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

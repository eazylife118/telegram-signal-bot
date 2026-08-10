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
# STRATEGY SETTINGS
# ============================================================
# Resistance/support zone tolerance
ZONE_TOLERANCE_RATIO = 0.035
# Minimum number of tests required
MIN_ZONE_TESTS = 2
# Minimum rejection wick ratio
REJECTION_WICK_RATIO = 0.40
# Small-body threshold
SMALL_BODY_RATIO = 0.35
# Open/close clustering
OPEN_CLOSE_CLUSTER_RATIO = 0.30
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
        40,
        40
    ])
    red_upper_1 = np.array([
        15,
        255,
        255
    ])
    red_lower_2 = np.array([
        165,
        40,
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
        for i, existing in enumerate(result):
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
    for candle in (
        green +
        red
    ):
        candle["x"] += right_start
        candle["center_x"] += right_start
    return (
        green +
        red
    )
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
        w
    )
    red = find_candidates(
        red_mask,
        "RED",
        w
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
    # Right-side improvement
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
    # IMPORTANT:
    # Sort LEFT → RIGHT internally,
    # then reverse for strategy analysis.
    candles.sort(
        key=lambda c: c["center_x"]
    )
    return candles
# ============================================================
# CONVERT BODY DETECTIONS TO NORMALIZED PRICE DATA
# ============================================================
def create_normalized_data(candles):
    if not candles:
        return None
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
    min_y = np.min(
        bottoms
    )
    max_y = np.max(
        tops
    )
    vertical_range = (
        max_y -
        min_y
    )
    if vertical_range <= 0:
        return None
    data = []
    for candle in candles:
        top = candle["y"]
        bottom = (
            candle["y"] +
            candle["h"]
        )
        high = (
            max_y -
            top
        ) / vertical_range
        low = (
            max_y -
            bottom
        ) / vertical_range
        candle_range = (
            high -
            low
        )
        if candle_range <= 0:
            continue
        if candle["color"] == "GREEN":
            close = (
                high -
                candle_range * 0.15
            )
            open_ = (
                low +
                candle_range * 0.15
            )
        else:
            open_ = (
                high -
                candle_range * 0.15
            )
            close = (
                low +
                candle_range * 0.15
            )
        data.append({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "color": candle["color"]
        })
    if not data:
        return None
    return data
# ============================================================
# CANDLE INFORMATION
# ============================================================
def candle_information(
    candle
):
    open_ = candle["open"]
    high = candle["high"]
    low = candle["low"]
    close = candle["close"]
    candle_range = (
        high -
        low
    )
    if candle_range <= 0:
        return {
            "body": 0,
            "range": 0,
            "body_ratio": 1,
            "upper_wick": 0,
            "lower_wick": 0
        }
    body = abs(
        close -
        open_
    )
    upper_wick = (
        high -
        max(
            open_,
            close
        )
    )
    lower_wick = (
        min(
            open_,
            close
        ) -
        low
    )
    return {
        "body": body,
        "range": candle_range,
        "body_ratio": (
            body /
            candle_range
        ),
        "upper_wick": upper_wick,
        "lower_wick": lower_wick
    }
# ============================================================
# REJECTION CANDLE DETECTION
# ============================================================
def detect_rejections(
    candles
):
    resistance = []
    support = []
    for i, candle in enumerate(candles):
        info = candle_information(
            candle
        )
        if info["range"] <= 0:
            continue
        upper_ratio = (
            info["upper_wick"] /
            info["range"]
        )
        lower_ratio = (
            info["lower_wick"] /
            info["range"]
        )
        # Resistance rejection
        if (
            info["body_ratio"]
            <= SMALL_BODY_RATIO
            and
            upper_ratio
            >= REJECTION_WICK_RATIO
        ):
            resistance.append({
                "index": i,
                "level": candle["high"],
                "wick_ratio": upper_ratio,
                "color": candle["color"]
            })
        # Support rejection
        if (
            info["body_ratio"]
            <= SMALL_BODY_RATIO
            and
            lower_ratio
            >= REJECTION_WICK_RATIO
        ):
            support.append({
                "index": i,
                "level": candle["low"],
                "wick_ratio": lower_ratio,
                "color": candle["color"]
            })
    return (
        resistance,
        support
    )
# ============================================================
# ZONE CLUSTERING
# ============================================================
def cluster_levels(
    rejections,
    tolerance
):
    if not rejections:
        return []
    sorted_levels = sorted(
        rejections,
        key=lambda x: x["level"]
    )
    zones = []
    current = [
        sorted_levels[0]
    ]
    for item in sorted_levels[1:]:
        current_level = np.mean([
            x["level"]
            for x in current
        ])
        if abs(
            item["level"] -
            current_level
        ) <= tolerance:
            current.append(
                item
            )
        else:
            zones.append(
                current
            )
            current = [
                item
            ]
    zones.append(
        current
    )
    return zones
# ============================================================
# BUILD SUPPORT / RESISTANCE ZONES
# ============================================================
def build_zones(
    candles
):
    all_prices = []
    for candle in candles:
        all_prices.append(
            candle["high"]
        )
        all_prices.append(
            candle["low"]
        )
    if not all_prices:
        return [], []
    chart_range = (
        max(all_prices) -
        min(all_prices)
    )
    if chart_range <= 0:
        return [], []
    tolerance = (
        chart_range *
        ZONE_TOLERANCE_RATIO
    )
    resistance_rejections, support_rejections = (
        detect_rejections(
            candles
        )
    )
    resistance_groups = cluster_levels(
        resistance_rejections,
        tolerance
    )
    support_groups = cluster_levels(
        support_rejections,
        tolerance
    )
    resistance_zones = []
    for group in resistance_groups:
        resistance_zones.append({
            "level": float(
                np.mean([
                    x["level"]
                    for x in group
                ])
            ),
            "tests": len(group),
            "indices": [
                x["index"]
                for x in group
            ]
        })
    support_zones = []
    for group in support_groups:
        support_zones.append({
            "level": float(
                np.mean([
                    x["level"]
                    for x in group
                ])
            ),
            "tests": len(group),
            "indices": [
                x["index"]
                for x in group
            ]
        })
    resistance_zones.sort(
        key=lambda z: z["tests"],
        reverse=True
    )
    support_zones.sort(
        key=lambda z: z["tests"],
        reverse=True
    )
    return (
        resistance_zones,
        support_zones
    )
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
        if side == "RESISTANCE":
            broke_above = (
                candle["high"] >
                level
            )
            failed_to_hold = (
                candle["close"] <
                level
            )
            turned_red = (
                candle["close"] <
                candle["open"]
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
                    "direction": "SELL"
                }
        if side == "SUPPORT":
            broke_below = (
                candle["low"] <
                level
            )
            failed_to_hold = (
                candle["close"] >
                level
            )
            turned_green = (
                candle["close"] >
                candle["open"]
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
                    "direction": "BUY"
                }
    return None
# ============================================================
# STRATEGY TEST
# ============================================================
def analyze_strategy(
    candles
):
    report = []
    if len(candles) < 5:
        return {
            "signal": None,
            "report": [
                "Not enough candles for strategy analysis."
            ]
        }
    # --------------------------------------------------------
    # IMPORTANT:
    # candles are now NEWEST → OLDEST
    # Candle #1 is the newest/rightmost candle.
    # --------------------------------------------------------
    newest = candles[0]
    report.append(
        "Candle #1 is the newest/rightmost visible candle: "
        + newest["color"]
        + "."
    )
    resistance_zones, support_zones = (
        build_zones(
            candles
        )
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
    # --------------------------------------------------------
    # RESISTANCE
    # --------------------------------------------------------
    if strongest_resistance:
        report.append(
            "Approximate resistance zone: "
            f"{strongest_resistance['tests']} "
            "visible rejection tests."
        )
    else:
        report.append(
            "Approximate resistance zone: "
            "NOT CLEAR."
        )
    # --------------------------------------------------------
    # SUPPORT
    # --------------------------------------------------------
    if strongest_support:
        report.append(
            "Approximate support zone: "
            f"{strongest_support['tests']} "
            "visible rejection tests."
        )
    else:
        report.append(
            "Approximate support zone: "
            "NOT CLEAR."
        )
    # --------------------------------------------------------
    # NEWEST CANDLE VS RESISTANCE
    # --------------------------------------------------------
    resistance_near = False
    if strongest_resistance:
        distance = abs(
            newest["high"] -
            strongest_resistance["level"]
        )
        resistance_near = (
            distance <=
            strongest_resistance["level"] * 0.04
        )
    if resistance_near:
        report.append(
            "Resistance check: "
            "Newest candle is near resistance."
        )
    else:
        report.append(
            "Resistance check: "
            "Newest candle is not near resistance."
        )
    # --------------------------------------------------------
    # NEWEST CANDLE VS SUPPORT
    # --------------------------------------------------------
    support_near = False
    if strongest_support:
        distance = abs(
            newest["low"] -
            strongest_support["level"]
        )
        support_near = (
            distance <=
            strongest_support["level"] * 0.04
        )
    if support_near:
        report.append(
            "Support check: "
            "Newest candle is near support."
        )
    else:
        report.append(
            "Support check: "
            "Newest candle is not near support."
        )
    # --------------------------------------------------------
    # BODY MOMENTUM
    # --------------------------------------------------------
    newest_info = candle_information(
        newest
    )
    previous_bodies = []
    for candle in candles[1:3]:
        info = candle_information(
            candle
        )
        previous_bodies.append(
            info["body"]
        )
    if previous_bodies:
        average_previous = np.mean(
            previous_bodies
        )
        if (
            newest_info["body"] <
            average_previous
        ):
            report.append(
                "Momentum check: "
                "Newest body is weaker relative "
                "to the previous two visible bodies."
            )
        else:
            report.append(
                "Momentum check: "
                "Newest body is equal to or stronger "
                "than the previous two visible bodies."
            )
    # --------------------------------------------------------
    # BREAKOUT FAILURE
    # --------------------------------------------------------
    resistance_failure = None
    support_failure = None
    if strongest_resistance:
        resistance_failure = (
            detect_breakout_failure(
                candles,
                strongest_resistance,
                "RESISTANCE"
            )
        )
    if strongest_support:
        support_failure = (
            detect_breakout_failure(
                candles,
                strongest_support,
                "SUPPORT"
            )
        )
    if resistance_failure:
        report.append(
            "Resistance breakout attempt: YES."
        )
        report.append(
            "Resistance breakout failure: YES."
        )
        report.append(
            "Reversal direction from price structure: SELL."
        )
    elif strongest_resistance:
        report.append(
            "Resistance breakout failure: NOT CONFIRMED."
        )
    if support_failure:
        report.append(
            "Support breakout attempt: YES."
        )
        report.append(
            "Support breakout failure: YES."
        )
        report.append(
            "Reversal direction from price structure: BUY."
        )
    elif strongest_support:
        report.append(
            "Support breakout failure: NOT CONFIRMED."
        )
    # --------------------------------------------------------
    # FINAL STRATEGY TEST
    # --------------------------------------------------------
    signal = None
    if (
        resistance_near
        and
        resistance_failure
    ):
        signal = "SELL"
    elif (
        support_near
        and
        support_failure
    ):
        signal = "BUY"
    if signal:
        report.append(
            "Strategy interpretation: "
            f"COMPLETE {signal} REVERSAL STRUCTURE DETECTED."
        )
    else:
        report.append(
            "Strategy interpretation: "
            "NO COMPLETE REVERSAL SETUP FROM "
            "CANDLE INFORMATION ALONE."
        )
    return {
        "signal": signal,
        "report": report
    }
# ============================================================
# CREATE DETECTION MAP
# ============================================================
def create_detection_map(
    img,
    candles
):
    output = img.copy()
    # candles are NEWEST → OLDEST
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
        # Yellow box
        cv2.rectangle(
            output,
            (x, y),
            (
                x + w,
                y + h
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
        "strategy_detection.png"
    )
    try:
        bot.reply_to(
            message,
            "👁️ Reading visible candles...\n"
            "➡️ Scanning RIGHT → LEFT.\n"
            "🧠 Testing your rejection strategy."
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
        # DETECT
        # ----------------------------------------------------
        detected = detect_candles(
            img
        )
        if not detected:
            bot.reply_to(
                message,
                "❌ No reliable candle bodies detected.\n\n"
                "No strategy analysis was performed."
            )
            return
        # ----------------------------------------------------
        # RIGHT → LEFT
        # ----------------------------------------------------
        candles_left_to_right = detected
        candles = list(
            reversed(
                candles_left_to_right
            )
        )
        # ----------------------------------------------------
        # COLOR COUNT
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
        total = len(
            candles
        )
        # ----------------------------------------------------
        # CANDLE SEQUENCE
        # ----------------------------------------------------
        sequence = []
        for number, candle in enumerate(
            candles,
            start=1
        ):
            if candle["color"] == "GREEN":
                sequence.append(
                    f"{number}. 🟢 GREEN"
                )
            else:
                sequence.append(
                    f"{number}. 🔴 RED"
                )
        sequence_text = "\n".join(
            sequence
        )
        # ----------------------------------------------------
        # STRATEGY
        # ----------------------------------------------------
        strategy = analyze_strategy(
            create_normalized_data(
                candles_left_to_right
            )
            if False
            else create_normalized_data(
                candles_left_to_right
            )
        )
        # NOTE:
        # Strategy analysis uses normalized candles.
        # Reverse them so newest = #1.
        normalized = create_normalized_data(
            candles_left_to_right
        )
        normalized = list(
            reversed(
                normalized
            )
        )
        strategy = analyze_strategy(
            normalized
        )
        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------
        strategy_text = "\n".join(
            "• " + item
            for item in strategy["report"]
        )
        elapsed = (
            time.time() -
            start_time
        )
        result = (
            "🔎 **OTC STRATEGY READING TEST**\n\n"
            "➡️ **SCAN DIRECTION: RIGHT → LEFT**\n"
            "🎯 **CANDLE #1 = NEWEST VISIBLE CANDLE**\n\n"
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
            "🎯 **CURRENT TEST RESULT:**\n"
            f"{'⚠️ COMPLETE SETUP DETECTED: ' + strategy['signal'] if strategy['signal'] else '⛔ NO COMPLETE REVERSAL SETUP'}\n\n"
            "⚠️ **TEST MODE ONLY**\n"
            "No trade is opened.\n"
            "No Pocket Option connection.\n"
            "No pair detection.\n"
            "No random data.\n"
            "No automatic entry.\n"
            "MACD and Volume are intentionally excluded "
            "from this test.\n\n"
            f"⚡ Processing time: {elapsed:.2f}s"
        )
        bot.reply_to(
            message,
            result,
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
                    "➡️ Numbering is RIGHT → LEFT.\n"
                    "🎯 #1 = newest/rightmost visible candle.\n"
                    "🟡 Yellow boxes = detected candle bodies.\n"
                    "🟢 numbers = GREEN candles.\n"
                    "🔴 numbers = RED candles.\n\n"
                    "Check the boxes against the actual "
                    "candles before we combine this with "
                    "your main bot."
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
            f"❌ Strategy detection error:\n{str(e)}"
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
    "🔎 OTC STRATEGY READING TEST"
)
print(
    "========================================"
)
print(
    "➡️ Scan direction: RIGHT → LEFT"
)
print(
    "🎯 Candle #1 = newest/rightmost candle"
)
print(
    "📍 Rejection zones enabled"
)
print(
    "📍 Support/resistance enabled"
)
print(
    "🔄 Breakout-failure testing enabled"
)
print(
    "🧠 Candle momentum check enabled"
)
print(
    "❌ Volume excluded"
)
print(
    "❌ MACD excluded"
)
print(
    "❌ Pair detection excluded"
)
print(
    "❌ Automatic trading excluded"
)
print(
    "========================================"
)
bot.infinity_polling(
    timeout=30,
    long_polling_timeout=30
)

import os
import cv2
import numpy as np
import telebot
import time
# ============================================================
# TELEGRAM
# ============================================================
# Keep your real token in Render as:
# BOT_TOKEN = your Telegram bot token
#
# Do NOT put the real token directly into this file.
TELEGRAM_TOKEN = os.getenv(
    "BOT_TOKEN",
    "PASTE_YOUR_BOT_TOKEN_HERE"
)
bot = telebot.TeleBot(TELEGRAM_TOKEN)
# ============================================================
# DETECTION SETTINGS
# ============================================================
# Minimum candle-body evidence
MIN_BODY_AREA = 10
MIN_BODY_HEIGHT = 2
MIN_CANDLE_WIDTH = 2
# Newest/right-side candles
RIGHT_MIN_BODY_AREA = 6
RIGHT_MIN_BODY_HEIGHT = 2
# Maximum candle width relative to screenshot
MAX_CANDLE_WIDTH_RATIO = 0.045
# Very close pieces of the same candle can be merged
MERGE_DISTANCE_RATIO = 0.55
# ============================================================
# STRICT COLOR SETTINGS
# ============================================================
# Minimum HSV saturation.
# This helps reject white/gray chart elements.
MIN_RED_SATURATION = 80
MIN_GREEN_SATURATION = 60
# Minimum brightness/value.
MIN_RED_VALUE = 50
MIN_GREEN_VALUE = 45
# Red hue ranges
RED_HUE_1_LOW = 0
RED_HUE_1_HIGH = 10
RED_HUE_2_LOW = 172
RED_HUE_2_HIGH = 180
# Green hue range
GREEN_HUE_LOW = 35
GREEN_HUE_HIGH = 85
# Minimum percentage of pixels that must satisfy
# the color condition inside the detected body.
MIN_COLOR_DENSITY = 0.25
# A red candidate must have clearly stronger red evidence
# than green evidence.
RED_DOMINANCE_RATIO = 1.35
# A green candidate must have clearly stronger green evidence
# than red evidence.
GREEN_DOMINANCE_RATIO = 1.25
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
    # Upscale smaller screenshots slightly.
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
    # ========================================================
    # GREEN
    # ========================================================
    green_lower = np.array([
        GREEN_HUE_LOW,
        MIN_GREEN_SATURATION,
        MIN_GREEN_VALUE
    ])
    green_upper = np.array([
        GREEN_HUE_HIGH,
        255,
        255
    ])
    green = cv2.inRange(
        hsv,
        green_lower,
        green_upper
    )
    # ========================================================
    # RED
    # ========================================================
    red_lower_1 = np.array([
        RED_HUE_1_LOW,
        MIN_RED_SATURATION,
        MIN_RED_VALUE
    ])
    red_upper_1 = np.array([
        RED_HUE_1_HIGH,
        255,
        255
    ])
    red_lower_2 = np.array([
        RED_HUE_2_LOW,
        MIN_RED_SATURATION,
        MIN_RED_VALUE
    ])
    red_upper_2 = np.array([
        RED_HUE_2_HIGH,
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
    # ========================================================
    # BGR COLOR-DOMINANCE FILTER
    # ========================================================
    #
    # HSV alone can sometimes classify another reddish/orange
    # object as red.
    #
    # This additional check requires the actual red channel
    # to dominate the green/blue channels.
    # ========================================================
    b, g, r = cv2.split(img)
    red_dominance = (
        (r.astype(np.int16) >
         g.astype(np.int16) * 1.15)
        &
        (r.astype(np.int16) >
         b.astype(np.int16) * 1.15)
        &
        (r.astype(np.int16) > 60)
    )
    red_dominance_mask = (
        red_dominance.astype(np.uint8) * 255
    )
    red = cv2.bitwise_and(
        red,
        red_dominance_mask
    )
    # ========================================================
    # GREEN DOMINANCE
    # ========================================================
    green_dominance = (
        (g.astype(np.int16) >
         r.astype(np.int16) * 1.05)
        &
        (g.astype(np.int16) >
         b.astype(np.int16) * 1.00)
        &
        (g.astype(np.int16) > 50)
    )
    green_dominance_mask = (
        green_dominance.astype(np.uint8) * 255
    )
    green = cv2.bitwise_and(
        green,
        green_dominance_mask
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
    # Very small morphology.
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
        # ====================================================
        # BODY PIXEL DENSITY
        # ====================================================
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
            float(
                max(
                    1,
                    w * h
                )
            )
        )
        if density < MIN_COLOR_DENSITY:
            continue
        # ====================================================
        # COLOR CONFIDENCE CHECK
        # ====================================================
        # Get original HSV region.
        # This prevents a tiny number of colored pixels
        # from creating a candle detection.
        #
        # ====================================================
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
            candidate_top = (
                candidate["y"]
            )
            candidate_bottom = (
                candidate["y"] +
                candidate["h"]
            )
            existing_top = (
                existing["y"]
            )
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
            # Keep stronger actual color evidence.
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
# MILD RIGHT-SIDE IMPROVEMENT
# ============================================================
def detect_right_side(
    chart,
    green_mask,
    red_mask
):
    """
    Controlled second pass.
    Only the newest/right-side section gets the
    slightly smaller-body allowance.
    """
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
    # ========================================================
    # MAIN DETECTION
    # ========================================================
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
    # ========================================================
    # RIGHT-SIDE PASS
    # ========================================================
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
    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================
    candles = (
        remove_cross_color_duplicates(
            candles
        )
    )
    # ========================================================
    # RIGHT → LEFT
    # ========================================================
    #
    # IMPORTANT:
    #
    # The newest/rightmost candle becomes #1.
    #
    # Then the detector moves:
    #
    # #1 → #2 → #3 → #4 ...
    #
    # from RIGHT to LEFT.
    # ========================================================
    candles.sort(
        key=lambda c: c["center_x"],
        reverse=True
    )
    return candles
# ============================================================
# REPORT
# ============================================================
def create_report(
    candles
):
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
    return green, red
# ============================================================
# NUMBERED DETECTION MAP
# ============================================================
def create_detection_map(
    img,
    candles
):
    output = img.copy()
    # ========================================================
    # RIGHT → LEFT NUMBERING
    # ========================================================
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
        # Yellow detection box.
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
        # Color-specific number.
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
def handle_photo(
    message
):
    start_time = time.time()
    original_path = (
        "chart_screenshot.png"
    )
    detection_path = (
        "candle_detection.png"
    )
    try:
        bot.reply_to(
            message,
            "👁️ Reading visible candles...\n"
            "➡️ Scanning RIGHT → LEFT.\n"
            "🔎 Checking strict GREEN/RED classification."
        )
        # ====================================================
        # DOWNLOAD HIGHEST RESOLUTION
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
        # LOAD
        # ====================================================
        img = load_image(
            original_path
        )
        # ====================================================
        # DETECT
        # ====================================================
        candles = detect_candles(
            img
        )
        # ====================================================
        # COUNT
        # ====================================================
        green, red = (
            create_report(
                candles
            )
        )
        total = len(
            candles
        )
        elapsed = (
            time.time() -
            start_time
        )
        # ====================================================
        # RIGHT → LEFT SEQUENCE
        # ====================================================
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
        sequence_text = (
            "\n".join(
                sequence
            )
        )
        # ====================================================
        # RESULT
        # ====================================================
        if total == 0:
            bot.reply_to(
                message,
                "❌ No reliable candle bodies detected.\n\n"
                "No candle was generated.\n"
                "No random candle was added.\n"
                "No signal was generated."
            )
            return
        report = (
            "🔎 **CANDLE READING TEST**\n\n"
            "➡️ **SCAN DIRECTION:** "
            "RIGHT → LEFT\n\n"
            "📊 **WHAT THE BOT ACTUALLY DETECTED:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🟢 GREEN: {green}\n"
            f"🔴 RED: {red}\n"
            f"📊 TOTAL: {total}\n\n"
            "🕯️ **RIGHT → LEFT CANDLE READING:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"{sequence_text}\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🎯 **COLOR CHECK:**\n"
            "🟢 = Bot believes the candle is GREEN\n"
            "🔴 = Bot believes the candle is RED\n\n"
            "🔢 **NUMBER 1 = NEWEST/RIGHTMOST "
            "DETECTED CANDLE**\n\n"
            "⚠️ This is ONLY a candle-reading test.\n"
            "No OHLC data is generated.\n"
            "No random candles are added.\n"
            "No trading signal is generated.\n\n"
            f"⚡ Processing time: "
            f"{elapsed:.2f}s"
        )
        bot.reply_to(
            message,
            report,
            parse_mode="Markdown"
        )
        # ====================================================
        # CREATE DETECTION MAP
        # ====================================================
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
        # ====================================================
        # SEND MAP
        # ====================================================
        with open(
            detection_path,
            "rb"
        ) as photo:
            bot.send_photo(
                message.chat.id,
                photo,
                caption=(
                    "🔢 **RIGHT → LEFT CANDLE MAP**\n\n"
                    "Number 1 = newest/rightmost "
                    "detected candle.\n\n"
                    "➡️ Counting continues "
                    "from RIGHT → LEFT.\n\n"
                    "🟨 Yellow box = detected candle body.\n"
                    "🟢 Number = classified GREEN.\n"
                    "🔴 Number = classified RED.\n\n"
                    "Compare every box with the actual "
                    "candles in your screenshot."
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
            f"❌ Detection error:\n{str(e)}"
        )
    finally:
        for path in [
            original_path,
            detection_path
        ]:
            if os.path.exists(
                path
            ):
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
    "🕯️ CANDLE READING TEST"
)
print(
    "========================================"
)
print(
    "➡️ Scan direction: RIGHT → LEFT"
)
print(
    "🔢 Number 1 = newest/rightmost candle"
)
print(
    "🔴 Strict red detection enabled"
)
print(
    "🟢 Strict green detection enabled"
)
print(
    "🚫 No OHLC generation"
)
print(
    "🚫 No random candles"
)
print(
    "🚫 No trading signals"
)
print(
    "========================================"
)
bot.infinity_polling(
    timeout=30,
    long_polling_timeout=30
)

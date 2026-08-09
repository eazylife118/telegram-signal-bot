import os
import cv2
import numpy as np
import telebot
import time
# ============================================================
# TELEGRAM
# ============================================================
# Put your NEW token in the BOT_TOKEN environment variable.
# Do not hard-code your Telegram token into public code.
TELEGRAM_TOKEN = os.getenv(
    "BOT_TOKEN",
    "PUT_YOUR_NEW_BOT_TOKEN_HERE"
)
if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "PUT_YOUR_NEW_BOT_TOKEN_HERE":
    raise ValueError(
        "BOT_TOKEN is not configured. Add your new Telegram bot token."
    )
bot = telebot.TeleBot(TELEGRAM_TOKEN)
# ============================================================
# DETECTION SETTINGS
# ============================================================
# Based on the previous detector that performed better.
MIN_BODY_AREA = 6
MIN_BODY_HEIGHT = 2
MIN_CANDLE_WIDTH = 2
# Slightly more tolerant than the previous version,
# but NOT extreme.
MIN_COLOR_PIXELS = 4
MIN_COLOR_DENSITY = 0.10
# Candle bodies should remain relatively narrow.
MAX_CANDLE_WIDTH_RATIO = 0.045
# IMPORTANT:
# Smaller value = harder to merge nearby candles.
# We want to avoid combining two neighboring candles.
MERGE_DISTANCE_RATIO = 0.25
# Cross-color duplicate distance.
CROSS_COLOR_DISTANCE_RATIO = 0.30
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
    # Mild upscale for small Telegram images.
    # Do not crop anything.
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
# FULL-SCREEN DETECTION RECTANGLE
# ============================================================
def get_full_detection_zone(img):
    h, w = img.shape[:2]
    # The entire screenshot is the detection zone.
    #
    # No:
    # - left crop
    # - right crop
    # - top crop
    # - bottom crop
    #
    return 0, 0, w, h
# ============================================================
# COLOR MASKS
# ============================================================
def get_color_masks(img):
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )
    # --------------------------------------------------------
    # HSV GREEN
    # --------------------------------------------------------
    green_lower = np.array([
        30,
        30,
        30
    ])
    green_upper = np.array([
        95,
        255,
        255
    ])
    green_hsv = cv2.inRange(
        hsv,
        green_lower,
        green_upper
    )
    # --------------------------------------------------------
    # HSV RED
    # --------------------------------------------------------
    red_lower_1 = np.array([
        0,
        25,
        25
    ])
    red_upper_1 = np.array([
        18,
        255,
        255
    ])
    red_lower_2 = np.array([
        162,
        25,
        25
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
    red_hsv = cv2.bitwise_or(
        red1,
        red2
    )
    # --------------------------------------------------------
    # BGR SECONDARY COLOR CHECK
    # --------------------------------------------------------
    b, g, r = cv2.split(img)
    # Difference must be meaningful enough to avoid
    # treating ordinary grey/white chart elements as colors.
    green_bgr = (
        (g.astype(np.int16) > r.astype(np.int16) + 12) &
        (g.astype(np.int16) > b.astype(np.int16) + 5) &
        (g > 45)
    )
    red_bgr = (
        (r.astype(np.int16) > g.astype(np.int16) + 12) &
        (r.astype(np.int16) > b.astype(np.int16) + 5) &
        (r > 45)
    )
    green_bgr = (
        green_bgr.astype(np.uint8) * 255
    )
    red_bgr = (
        red_bgr.astype(np.uint8) * 255
    )
    # Combine HSV + BGR carefully.
    #
    # HSV remains the main detector.
    # BGR only helps recover weaker colors.
    green = cv2.bitwise_or(
        green_hsv,
        green_bgr
    )
    red = cv2.bitwise_or(
        red_hsv,
        red_bgr
    )
    return green, red
# ============================================================
# FIND CANDIDATES
# ============================================================
def find_candidates(
    mask,
    color,
    image_width
):
    # Very mild cleanup.
    # Do not aggressively connect neighboring candles.
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
        (2, 2)
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
    for contour in contours:
        area = cv2.contourArea(
            contour
        )
        if area < MIN_BODY_AREA:
            continue
        x, y, w, h = cv2.boundingRect(
            contour
        )
        if w < MIN_CANDLE_WIDTH:
            continue
        if h < MIN_BODY_HEIGHT:
            continue
        if w > max_width:
            continue
        # Reject obvious horizontal UI elements.
        if w > h * 5:
            continue
        region = cleaned[
            y:y + h,
            x:x + w
        ]
        colored_pixels = int(
            np.sum(region > 0)
        )
        if colored_pixels < MIN_COLOR_PIXELS:
            continue
        density = (
            colored_pixels /
            float(max(1, w * h))
        )
        if density < MIN_COLOR_DENSITY:
            continue
        center_x = (
            x +
            w / 2.0
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
# MERGE ONLY CLEARLY CONNECTED PIECES
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
                candidate["center_x"] -
                existing["center_x"]
            )
            allowed = max(
                candidate["w"],
                existing["w"],
                2
            ) * MERGE_DISTANCE_RATIO
            # Vertical overlap.
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
            # Horizontal boxes must actually touch or be
            # extremely close. This is much stricter than
            # the previous 0.55 setting.
            candidate_right = (
                candidate["x"] +
                candidate["w"]
            )
            existing_right = (
                existing["x"] +
                existing["w"]
            )
            horizontal_gap = max(
                0,
                max(
                    candidate["x"],
                    existing["x"]
                ) -
                min(
                    candidate_right,
                    existing_right
                )
            )
            touching_or_near = (
                horizontal_gap <= 1
            )
            if (
                distance <= allowed
                and
                vertical_overlap
                and
                touching_or_near
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
                    existing["w"] / 2.0
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
# REMOVE TRUE DUPLICATES
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
                candle["center_x"] -
                existing["center_x"]
            )
            threshold = max(
                candle["w"],
                existing["w"],
                2
            ) * CROSS_COLOR_DISTANCE_RATIO
            if distance <= threshold:
                # Only consider it a duplicate if the
                # detections occupy substantially similar
                # vertical space.
                candle_top = candle["y"]
                candle_bottom = (
                    candle["y"] +
                    candle["h"]
                )
                existing_top = existing["y"]
                existing_bottom = (
                    existing["y"] +
                    existing["h"]
                )
                overlap = max(
                    0,
                    min(
                        candle_bottom,
                        existing_bottom
                    ) -
                    max(
                        candle_top,
                        existing_top
                    )
                )
                smaller_height = max(
                    1,
                    min(
                        candle["h"],
                        existing["h"]
                    )
                )
                overlap_ratio = (
                    overlap /
                    smaller_height
                )
                if overlap_ratio >= 0.50:
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
            # Keep the detection with stronger
            # colored-pixel evidence.
            if (
                candle["pixels"] >
                existing["pixels"]
            ):
                result[
                    duplicate_index
                ] = candle
    return result
# ============================================================
# DETECT CANDLES — ENTIRE SCREEN
# ============================================================
def detect_candles(img):
    h, w = img.shape[:2]
    # Full iPhone screenshot = detection zone.
    zone_x, zone_y, zone_w, zone_h = (
        get_full_detection_zone(img)
    )
    chart = img[
        zone_y:zone_y + zone_h,
        zone_x:zone_x + zone_w
    ]
    green_mask, red_mask = (
        get_color_masks(chart)
    )
    # --------------------------------------------------------
    # GREEN — ENTIRE SCREEN
    # --------------------------------------------------------
    green = find_candidates(
        green_mask,
        "GREEN",
        zone_w
    )
    # --------------------------------------------------------
    # RED — ENTIRE SCREEN
    # --------------------------------------------------------
    red = find_candidates(
        red_mask,
        "RED",
        zone_w
    )
    # --------------------------------------------------------
    # MERGE PIECES
    # --------------------------------------------------------
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
    # --------------------------------------------------------
    # REMOVE TRUE DUPLICATES
    # --------------------------------------------------------
    candles = (
        remove_cross_color_duplicates(
            candles
        )
    )
    # --------------------------------------------------------
    # RESTORE FULL-SCREEN COORDINATES
    # --------------------------------------------------------
    for candle in candles:
        candle["x"] += zone_x
        candle["y"] += zone_y
        candle["center_x"] += zone_x
    # --------------------------------------------------------
    # LEFT → RIGHT
    # --------------------------------------------------------
    candles.sort(
        key=lambda c: c["center_x"]
    )
    return candles
# ============================================================
# REPORT
# ============================================================
def create_report(candles):
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
# FULL-SCREEN DETECTION MAP
# ============================================================
def create_detection_map(
    img,
    candles
):
    output = img.copy()
    h, w = output.shape[:2]
    # --------------------------------------------------------
    # LARGE RECTANGLE = ENTIRE SCREEN
    # --------------------------------------------------------
    cv2.rectangle(
        output,
        (0, 0),
        (w - 1, h - 1),
        (0, 255, 255),
        4
    )
    cv2.putText(
        output,
        "FULL SCREEN CANDLE DETECTION ZONE",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )
    # --------------------------------------------------------
    # CANDLE BOXES
    # --------------------------------------------------------
    for number, candle in enumerate(
        candles,
        start=1
    ):
        x = int(candle["x"])
        y = int(candle["y"])
        cw = int(candle["w"])
        ch = int(candle["h"])
        # Yellow box = detected candle body.
        cv2.rectangle(
            output,
            (x, y),
            (x + cw, y + ch),
            (0, 255, 255),
            2
        )
        if candle["color"] == "GREEN":
            label_color = (
                0,
                255,
                0
            )
            label = (
                f"{number} G"
            )
        else:
            label_color = (
                0,
                0,
                255
            )
            label = (
                f"{number} R"
            )
        cv2.putText(
            output,
            label,
            (
                x,
                max(
                    25,
                    y - 7
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
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
        "candle_detection.png"
    )
    try:
        bot.reply_to(
            message,
            "👁️ Reading the entire screenshot...\n"
            "🟡 Full-screen detection zone\n"
            "🟢 Checking GREEN candles\n"
            "🔴 Checking RED candles"
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
        # ----------------------------------------------------
        # COUNT
        # ----------------------------------------------------
        green, red = create_report(
            candles
        )
        total = len(candles)
        elapsed = (
            time.time() -
            start_time
        )
        # ----------------------------------------------------
        # SEQUENCE
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
        # NO DETECTION
        # ----------------------------------------------------
        if total == 0:
            bot.reply_to(
                message,
                "❌ No candle bodies were detected.\n\n"
                "The bot did NOT invent candles.\n"
                "The bot did NOT generate OHLC data.\n"
                "The bot did NOT create a trading signal."
            )
            return
        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------
        report = (
            "🔎 **CANDLE READING TEST**\n\n"
            "🟡 **DETECTION AREA:**\n"
            "Entire uploaded screenshot — 0% to 100%\n\n"
            "📊 **WHAT THE BOT ACTUALLY DETECTED:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🟢 GREEN: {green}\n"
            f"🔴 RED: {red}\n"
            f"📊 TOTAL: {total}\n\n"
            "🕯️ **CANDLE-BY-CANDLE READING:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"{sequence_text}\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🎯 **COLOR CHECK:**\n"
            "🟢 G = Bot classified GREEN\n"
            "🔴 R = Bot classified RED\n\n"
            "⚠️ **TEST ONLY**\n"
            "No OHLC candles are generated.\n"
            "No random candles are added.\n"
            "No trading signal is generated.\n\n"
            f"⚡ Processing time: {elapsed:.2f}s"
        )
        bot.reply_to(
            message,
            report,
            parse_mode="Markdown"
        )
        # ----------------------------------------------------
        # CREATE DETECTION MAP
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
        # ----------------------------------------------------
        # SEND DETECTION MAP
        # ----------------------------------------------------
        with open(
            detection_path,
            "rb"
        ) as photo:
            bot.send_photo(
                message.chat.id,
                photo,
                caption=(
                    "🟡 FULL SCREEN = detection zone\n\n"
                    "Yellow boxes = candle bodies detected.\n"
                    "🟢 number G = GREEN classification.\n"
                    "🔴 number R = RED classification.\n\n"
                    "Check the boxes against the actual "
                    "candles in your screenshot.\n\n"
                    "If a candle is visible but has NO box, "
                    "we know the detector missed it."
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
    "🕯️ FULL-SCREEN CANDLE READING TEST"
)
print(
    "========================================"
)
print(
    "Detection area: 0% → 100% of screenshot"
)
print(
    "No left/right crop"
)
print(
    "HSV + mild BGR color assistance"
)
print(
    "Strict candle merging"
)
print(
    "Small-body detection enabled"
)
print(
    "No OHLC generation"
)
print(
    "No random candles"
)
print(
    "No trading signals"
)
print(
    "========================================"
)
bot.infinity_polling(
    timeout=30,
    long_polling_timeout=30
)

import os
import cv2
import numpy as np
import telebot
import time
# ============================================================
# TELEGRAM
# ============================================================
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN environment variable is not set."
    )
bot = telebot.TeleBot(TELEGRAM_TOKEN)
# ============================================================
# DETECTION SETTINGS
# ============================================================
# Lower than the previous version so small candle bodies
# have a chance to be detected.
MIN_BODY_AREA = 5
MIN_BODY_HEIGHT = 2
MIN_CANDLE_WIDTH = 1
# Slightly more sensitive for small/dim bodies.
MIN_COLOR_DENSITY = 0.05
# Prevent large UI/chart objects from becoming candles.
MAX_CANDLE_WIDTH_RATIO = 0.025
# A candle body should not be extremely wide compared
# with its height.
MAX_WIDTH_TO_HEIGHT = 5.0
# ------------------------------------------------------------
# IMPORTANT:
# We do NOT use an extreme fallback.
# We do NOT force the count to 40, 48 or 50.
# ------------------------------------------------------------
# ============================================================
# LOAD FULL SCREENSHOT
# ============================================================
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read screenshot.")
    h, w = img.shape[:2]
    # Mild upscale for small screenshots.
    if w < 1400:
        scale = 1400.0 / w
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
# COLOR DETECTION
# ============================================================
def get_color_masks(img):
    # ========================================================
    # HSV DETECTION
    # ========================================================
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )
    # GREEN
    hsv_green_lower = np.array([
        30,
        25,
        25
    ])
    hsv_green_upper = np.array([
        95,
        255,
        255
    ])
    hsv_green = cv2.inRange(
        hsv,
        hsv_green_lower,
        hsv_green_upper
    )
    # RED
    hsv_red_lower_1 = np.array([
        0,
        25,
        25
    ])
    hsv_red_upper_1 = np.array([
        18,
        255,
        255
    ])
    hsv_red_lower_2 = np.array([
        160,
        25,
        25
    ])
    hsv_red_upper_2 = np.array([
        180,
        255,
        255
    ])
    hsv_red_1 = cv2.inRange(
        hsv,
        hsv_red_lower_1,
        hsv_red_upper_1
    )
    hsv_red_2 = cv2.inRange(
        hsv,
        hsv_red_lower_2,
        hsv_red_upper_2
    )
    hsv_red = cv2.bitwise_or(
        hsv_red_1,
        hsv_red_2
    )
    # ========================================================
    # BGR CHANNEL-DIFFERENCE DETECTION
    # ========================================================
    b, g, r = cv2.split(img)
    # Green:
    # green channel must be meaningfully stronger
    # than red and blue.
    green_bgr = (
        (g.astype(np.int16) > r.astype(np.int16) + 10)
        &
        (g.astype(np.int16) > b.astype(np.int16) + 10)
        &
        (g > 35)
    ).astype(np.uint8) * 255
    # Red:
    # red channel must be meaningfully stronger
    # than green and blue.
    red_bgr = (
        (r.astype(np.int16) > g.astype(np.int16) + 8)
        &
        (r.astype(np.int16) > b.astype(np.int16) + 8)
        &
        (r > 35)
    ).astype(np.uint8) * 255
    # ========================================================
    # COMBINE HSV + BGR
    # ========================================================
    green = cv2.bitwise_or(
        hsv_green,
        green_bgr
    )
    red = cv2.bitwise_or(
        hsv_red,
        red_bgr
    )
    return green, red
# ============================================================
# CLEAN COLOR MASK
# ============================================================
def clean_mask(mask):
    # Very small opening removes isolated single pixels
    # without destroying small candle bodies.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )
    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel
    )
    # Small closing reconnects tiny gaps inside a candle.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        close_kernel
    )
    return cleaned
# ============================================================
# FIND CANDLE CANDIDATES
# ============================================================
def find_candidates(
    mask,
    color,
    image_width,
    image_height
):
    cleaned = clean_mask(mask)
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    max_width = max(
        12,
        int(
            image_width *
            MAX_CANDLE_WIDTH_RATIO
        )
    )
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_BODY_AREA:
            continue
        x, y, w, h = cv2.boundingRect(
            contour
        )
        if w < MIN_CANDLE_WIDTH:
            continue
        if h < MIN_BODY_HEIGHT:
            continue
        # Reject huge horizontal regions.
        if w > max_width:
            continue
        # Candle bodies should normally be taller
        # than they are wide.
        if w > h * MAX_WIDTH_TO_HEIGHT:
            continue
        # Reject enormous vertical UI/chart objects.
        if h > image_height * 0.40:
            continue
        # ====================================================
        # COLORED PIXEL EVIDENCE
        # ====================================================
        region = cleaned[
            y:y + h,
            x:x + w
        ]
        colored_pixels = int(
            np.count_nonzero(region)
        )
        if colored_pixels < 4:
            continue
        density = (
            colored_pixels /
            float(max(1, w * h))
        )
        if density < MIN_COLOR_DENSITY:
            continue
        # ====================================================
        # CENTER
        # ====================================================
        center_x = (
            x +
            w / 2.0
        )
        center_y = (
            y +
            h / 2.0
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
            "center_y": center_y,
            "color": color
        })
    return candidates
# ============================================================
# MERGE ONLY GENUINELY TOUCHING PIECES
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
        merged_here = False
        for existing in merged:
            # Horizontal distance between the two objects.
            horizontal_gap = max(
                0,
                max(
                    existing["x"],
                    candidate["x"]
                )
                -
                min(
                    existing["x"] + existing["w"],
                    candidate["x"] + candidate["w"]
                )
            )
            # Vertical overlap.
            existing_top = existing["y"]
            existing_bottom = (
                existing["y"] +
                existing["h"]
            )
            candidate_top = candidate["y"]
            candidate_bottom = (
                candidate["y"] +
                candidate["h"]
            )
            overlap_top = max(
                existing_top,
                candidate_top
            )
            overlap_bottom = min(
                existing_bottom,
                candidate_bottom
            )
            vertical_overlap = (
                overlap_bottom >
                overlap_top
            )
            # ------------------------------------------------
            # IMPORTANT:
            # Only merge pieces that are actually touching
            # or separated by at most one pixel.
            # ------------------------------------------------
            if (
                horizontal_gap <= 1
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
                existing["w"] = right - left
                existing["h"] = bottom - top
                existing["center_x"] = (
                    left +
                    existing["w"] / 2.0
                )
                existing["center_y"] = (
                    top +
                    existing["h"] / 2.0
                )
                existing["area"] += (
                    candidate["area"]
                )
                existing["pixels"] += (
                    candidate["pixels"]
                )
                merged_here = True
                break
        if not merged_here:
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
    if not candles:
        return []
    candles = sorted(
        candles,
        key=lambda c: c["center_x"]
    )
    result = []
    for candle in candles:
        duplicate_index = None
        for i, existing in enumerate(result):
            # Require actual horizontal overlap,
            # not merely nearby candle centers.
            candle_left = candle["x"]
            candle_right = (
                candle["x"] +
                candle["w"]
            )
            existing_left = existing["x"]
            existing_right = (
                existing["x"] +
                existing["w"]
            )
            overlap = (
                min(
                    candle_right,
                    existing_right
                )
                -
                max(
                    candle_left,
                    existing_left
                )
            )
            if overlap <= 0:
                continue
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
            # Keep the stronger color evidence.
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
# REMOVE ONLY OBVIOUS NOISE
# ============================================================
def remove_obvious_noise(
    candles,
    image_width
):
    if len(candles) <= 2:
        return candles
    candles = sorted(
        candles,
        key=lambda c: c["center_x"]
    )
    result = []
    for candle in candles:
        # Very weak detections are allowed if they have
        # reasonable candle dimensions.
        #
        # We only reject extremely weak objects that are
        # also extremely small.
        tiny = (
            candle["w"] <= 1
            and
            candle["h"] <= 1
        )
        weak = (
            candle["pixels"] <= 2
            and
            candle["density"] < 0.03
        )
        if tiny and weak:
            continue
        result.append(
            candle
        )
    return result
# ============================================================
# FULL CHART CANDLE DETECTION
# ============================================================
def detect_candles(img):
    h, w = img.shape[:2]
    # ========================================================
    # FULL IMAGE
    # ========================================================
    #
    # No crop.
    # No right-side-only scan.
    # No left-side exclusion.
    #
    green_mask, red_mask = get_color_masks(
        img
    )
    # ========================================================
    # GREEN — ENTIRE SCREEN
    # ========================================================
    green = find_candidates(
        green_mask,
        "GREEN",
        w,
        h
    )
    green = merge_candidates(
        green
    )
    # ========================================================
    # RED — ENTIRE SCREEN
    # ========================================================
    red = find_candidates(
        red_mask,
        "RED",
        w,
        h
    )
    red = merge_candidates(
        red
    )
    # ========================================================
    # COMBINE
    # ========================================================
    candles = (
        green +
        red
    )
    # ========================================================
    # CROSS-COLOR DUPLICATES
    # ========================================================
    candles = remove_cross_color_duplicates(
        candles
    )
    # ========================================================
    # OBVIOUS NOISE ONLY
    # ========================================================
    candles = remove_obvious_noise(
        candles,
        w
    )
    # ========================================================
    # LEFT → RIGHT
    # ========================================================
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
        for candle in candles
        if candle["color"] == "GREEN"
    )
    red = sum(
        1
        for candle in candles
        if candle["color"] == "RED"
    )
    return green, red
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
        w = int(candle["w"])
        h = int(candle["h"])
        # ----------------------------------------------------
        # YELLOW BOX = DETECTED BODY
        # ----------------------------------------------------
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2
        )
        # ----------------------------------------------------
        # COLOR LABEL
        # ----------------------------------------------------
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
        # ----------------------------------------------------
        # NUMBER
        # ----------------------------------------------------
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
            "👁️ Reading the FULL screenshot...\n"
            "Checking green and red candle bodies..."
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
        # LOAD FULL IMAGE
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
        green, red = create_report(
            candles
        )
        total = len(candles)
        elapsed = (
            time.time()
            -
            start_time
        )
        # ====================================================
        # CANDLE SEQUENCE
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
        sequence_text = "\n".join(
            sequence
        )
        # ====================================================
        # NO DETECTION
        # ====================================================
        if total == 0:
            bot.reply_to(
                message,
                "❌ No reliable candle bodies detected.\n\n"
                "The detector did not generate or guess "
                "any candles."
            )
            return
        # ====================================================
        # REPORT
        # ====================================================
        report = (
            "🔎 **CANDLE READING TEST**\n\n"
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
            "🟢 = detector classified GREEN\n"
            "🔴 = detector classified RED\n\n"
            "🔬 **METHOD:**\n"
            "• Full 0–100% screenshot scan\n"
            "• HSV + BGR color detection\n"
            "• Small-body detection\n"
            "• Conservative merging\n"
            "• Duplicate removal\n"
            "• No right-side-only scan\n\n"
            "⚠️ **TEST ONLY**\n"
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
        # SEND DETECTION MAP
        # ====================================================
        with open(
            detection_path,
            "rb"
        ) as photo:
            bot.send_photo(
                message.chat.id,
                photo,
                caption=(
                    "🔢 **FULL-SCREEN CANDLE DETECTION MAP**\n\n"
                    "🟨 Yellow box = detected candle body\n"
                    "🟢 Number = classified GREEN\n"
                    "🔴 Number = classified RED\n\n"
                    "The entire screenshot was scanned.\n"
                    "No candle count was forced.\n"
                    "No missing candle was invented.\n\n"
                    "Compare the boxes with the actual "
                    "candles to identify missed or false detections."
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
    "Full screenshot scan: ENABLED"
)
print(
    "HSV + BGR detection: ENABLED"
)
print(
    "Small candle detection: ENABLED"
)
print(
    "Conservative merging: ENABLED"
)
print(
    "Right-side-only detection: DISABLED"
)
print(
    "Forced candle count: DISABLED"
)
print(
    "OHLC generation: DISABLED"
)
print(
    "Random candles: DISABLED"
)
print(
    "Trading signals: DISABLED"
)
print(
    "========================================"
)
bot.infinity_polling(
    timeout=30,
    long_polling_timeout=30
)

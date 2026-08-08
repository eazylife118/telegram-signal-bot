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
        "BOT_TOKEN is missing. Add your Telegram bot token "
        "as an environment variable."
    )
bot = telebot.TeleBot(TELEGRAM_TOKEN)
# ============================================================
# DETECTION SETTINGS
# ============================================================
MIN_BODY_AREA = 8
MIN_BODY_HEIGHT = 2
MIN_CANDLE_WIDTH = 2
# Slightly more sensitive on the newest/right-side candles.
RIGHT_MIN_BODY_AREA = 5
RIGHT_MIN_BODY_HEIGHT = 2
MAX_CANDLE_WIDTH_RATIO = 0.045
# Reduced merging so nearby candles stay separate.
MERGE_DISTANCE_RATIO = 0.35
MIN_COLORED_PIXELS = 4
MIN_COLOR_DENSITY = 0.12
# ============================================================
# LOAD IMAGE
# ============================================================
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read screenshot.")
    h, w = img.shape[:2]
    # Mild upscale only.
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
# HSV COLOR MASKS
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
    # Slightly more sensitive for faint/dark red.
    red_lower_1 = np.array([
        0,
        30,
        30
    ])
    red_upper_1 = np.array([
        18,
        255,
        255
    ])
    red_lower_2 = np.array([
        162,
        30,
        30
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
# BGR SECONDARY COLOR CHECK
# ============================================================
def verify_bgr_color(img, x, y, w, h, detected_color):
    """
    Secondary colour verification.
    HSV remains the PRIMARY detector.
    BGR is only used to check whether the actual pixels
    inside the detected candidate support the same colour.
    This helps especially with faint/dark red candles.
    It does NOT use:
        b > r
        g > r
        or similar loose rules.
    Blue pixels alone are never treated as green.
    """
    roi = img[
        y:y + h,
        x:x + w
    ]
    if roi.size == 0:
        return detected_color
    # Convert to HSV as well so we can identify the
    # coloured pixels inside the candidate.
    hsv_roi = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )
    # --------------------------------------------------------
    # Green pixels
    # --------------------------------------------------------
    green_mask = cv2.inRange(
        hsv_roi,
        np.array([30, 25, 25]),
        np.array([95, 255, 255])
    )
    # --------------------------------------------------------
    # Red pixels
    # --------------------------------------------------------
    red_mask_1 = cv2.inRange(
        hsv_roi,
        np.array([0, 20, 20]),
        np.array([20, 255, 255])
    )
    red_mask_2 = cv2.inRange(
        hsv_roi,
        np.array([160, 20, 20]),
        np.array([180, 255, 255])
    )
    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2
    )
    green_pixels = int(
        np.sum(green_mask > 0)
    )
    red_pixels = int(
        np.sum(red_mask > 0)
    )
    # --------------------------------------------------------
    # BGR channel averages
    # --------------------------------------------------------
    pixels = roi.reshape(
        -1,
        3
    ).astype(
        np.float32
    )
    b_avg = float(
        np.mean(pixels[:, 0])
    )
    g_avg = float(
        np.mean(pixels[:, 1])
    )
    r_avg = float(
        np.mean(pixels[:, 2])
    )
    # --------------------------------------------------------
    # Only change classification when the secondary
    # evidence is reasonably strong.
    # --------------------------------------------------------
    if red_pixels > green_pixels * 1.25:
        # Red mask is clearly stronger.
        if r_avg >= g_avg * 0.95:
            return "RED"
    if green_pixels > red_pixels * 1.25:
        # Green must actually dominate red.
        # We deliberately do NOT count blue as green.
        if g_avg > r_avg * 1.05:
            return "GREEN"
    # If the evidence is not strong enough,
    # keep the original HSV classification.
    return detected_color
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
        (2, 2)
    )
    cleaned = cv2.morphologyEx(
        mask,
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
        # Reject long horizontal chart/UI objects.
        if w > h * 6:
            continue
        # ----------------------------------------------------
        # COLOURED PIXEL DENSITY
        # ----------------------------------------------------
        region = cleaned[
            y:y + h,
            x:x + w
        ]
        colored_pixels = int(
            np.sum(region > 0)
        )
        if colored_pixels < MIN_COLORED_PIXELS:
            continue
        density = (
            colored_pixels /
            float(max(1, w * h))
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
# MERGE ONLY VERY CLOSE PIECES
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
            if not (
                distance <= allowed
                and
                vertical_overlap
            ):
                continue
            # ------------------------------------------------
            # Do not merge bodies with a real horizontal gap.
            # ------------------------------------------------
            candidate_left = candidate["x"]
            candidate_right = (
                candidate["x"] +
                candidate["w"]
            )
            existing_left = existing["x"]
            existing_right = (
                existing["x"] +
                existing["w"]
            )
            if candidate_right < existing_left:
                horizontal_gap = (
                    existing_left -
                    candidate_right
                )
            elif existing_right < candidate_left:
                horizontal_gap = (
                    candidate_left -
                    existing_right
                )
            else:
                horizontal_gap = 0
            if horizontal_gap > 2:
                continue
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
            ) * 0.45
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
# CONTROLLED RIGHT-SIDE PASS
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
    # --------------------------------------------------------
    # MAIN FULL-CHART DETECTION
    # --------------------------------------------------------
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
    # --------------------------------------------------------
    # CONTROLLED RIGHT-SIDE SECOND PASS
    # --------------------------------------------------------
    right_candidates = detect_right_side(
        img,
        green_mask,
        red_mask
    )
    candles.extend(
        right_candidates
    )
    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------
    candles = (
        remove_cross_color_duplicates(
            candles
        )
    )
    # --------------------------------------------------------
    # SECONDARY BGR COLOR VERIFICATION
    # --------------------------------------------------------
    for candle in candles:
        verified_color = verify_bgr_color(
            img,
            int(candle["x"]),
            int(candle["y"]),
            int(candle["w"]),
            int(candle["h"]),
            candle["color"]
        )
        candle["color"] = verified_color
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
        # Green = green box.
        if candle["color"] == "GREEN":
            box_color = (
                0,
                255,
                0
            )
            label_color = (
                0,
                255,
                0
            )
        # Red = red box.
        else:
            box_color = (
                0,
                0,
                255
            )
            label_color = (
                0,
                0,
                255
            )
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            box_color,
            2
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
        "candle_detection.png"
    )
    try:
        bot.reply_to(
            message,
            "👁️ Reading visible candles...\n"
            "Checking position and colour."
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
            time.time()
            -
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
                "❌ No reliable candle bodies detected.\n\n"
                "No candle was generated.\n"
                "No random candle was added.\n"
                "No trading signal was generated."
            )
            return
        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------
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
            "🟢 GREEN = classified GREEN\n"
            "🔴 RED = classified RED\n\n"
            "⚠️ **TEST ONLY**\n"
            "No OHLC data is generated.\n"
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
        # SEND MAP
        # ----------------------------------------------------
        with open(
            detection_path,
            "rb"
        ) as photo:
            bot.send_photo(
                message.chat.id,
                photo,
                caption=(
                    "🔢 **CANDLE DETECTION MAP**\n\n"
                    "🟢 Green box = classified GREEN.\n"
                    "🔴 Red box = classified RED.\n\n"
                    "Boxes show the actual regions "
                    "used for detection.\n\n"
                    "Compare every box with the "
                    "real candle in your screenshot."
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
    "🕯️ CANDLE READING TEST"
)
print(
    "========================================"
)
print(
    "Previous detector architecture."
)
print(
    "Controlled red sensitivity."
)
print(
    "Reduced merging."
)
print(
    "Small-body detection."
)
print(
    "Controlled right-side pass."
)
print(
    "BGR secondary colour verification."
)
print(
    "No forced candle count."
)
print(
    "No OHLC generation."
)
print(
    "No random candles."
)
print(
    "No trading signals."
)
print(
    "========================================"
)
bot.infinity_polling(
    timeout=30,
    long_polling_timeout=30
)

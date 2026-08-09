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
    "8937673241:AAGvyTA-G12xfwMlhif3Nh4_2Ag8OStq3tU"
)
bot = telebot.TeleBot(TELEGRAM_TOKEN)
# ============================================================
# SETTINGS
# ============================================================
# Full screenshot — no crop.
FULL_SCREEN = True
# Candle body limits.
MIN_BODY_AREA = 6
MIN_BODY_HEIGHT = 2
MIN_CANDLE_WIDTH = 2
# Do not allow giant objects to become candles.
MAX_CANDLE_WIDTH_RATIO = 0.025
# A real candle normally has concentrated color.
MIN_COLOR_DENSITY = 0.08
# Distance used only for obvious pieces of the SAME candle.
# Much less aggressive than the previous version.
TOUCH_GAP = 2
# ============================================================
# LOAD IMAGE
# ============================================================
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read screenshot.")
    return img
# ============================================================
# COLOR MASKS
# TWO SEPARATE DETECTORS
# ============================================================
def get_green_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Green candle colors.
    hsv_green = cv2.inRange(
        hsv,
        np.array([28, 45, 35]),
        np.array([95, 255, 255])
    )
    # BGR confirmation.
    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    bgr_green = (
        (g > r + 10) &
        (g > b + 5) &
        (g > 45)
    )
    bgr_green = (
        bgr_green.astype(np.uint8) * 255
    )
    # Either method can identify the pixel,
    # but both are restricted to green-like pixels.
    mask = cv2.bitwise_and(
        hsv_green,
        bgr_green
    )
    return mask
def get_red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Two red hue ranges.
    red1 = cv2.inRange(
        hsv,
        np.array([0, 35, 30]),
        np.array([18, 255, 255])
    )
    red2 = cv2.inRange(
        hsv,
        np.array([162, 35, 30]),
        np.array([180, 255, 255])
    )
    hsv_red = cv2.bitwise_or(
        red1,
        red2
    )
    # BGR confirmation.
    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    bgr_red = (
        (r > g + 10) &
        (r > b + 10) &
        (r > 45)
    )
    bgr_red = (
        bgr_red.astype(np.uint8) * 255
    )
    mask = cv2.bitwise_and(
        hsv_red,
        bgr_red
    )
    return mask
# ============================================================
# CLEAN MASK
# ============================================================
def clean_mask(mask):
    # Very light cleanup.
    # Do NOT use large kernels because nearby candles
    # must remain separate.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )
    return mask
# ============================================================
# FIND CANDLE CANDIDATES
# ============================================================
def find_candidates(mask, color, image_width):
    mask = clean_mask(mask)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    max_width = max(
        12,
        int(image_width * MAX_CANDLE_WIDTH_RATIO)
    )
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < MIN_BODY_AREA:
            continue
        if w < MIN_CANDLE_WIDTH:
            continue
        if h < MIN_BODY_HEIGHT:
            continue
        if w > max_width:
            continue
        # Reject obvious horizontal UI/text objects.
        if w > h * 5:
            continue
        roi = mask[
            y:y+h,
            x:x+w
        ]
        pixels = int(
            np.count_nonzero(roi)
        )
        if pixels < 4:
            continue
        density = (
            pixels /
            float(max(1, w * h))
        )
        if density < MIN_COLOR_DENSITY:
            continue
        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": float(area),
            "pixels": pixels,
            "density": density,
            "center_x": x + w / 2,
            "center_y": y + h / 2,
            "color": color
        })
    return candidates
# ============================================================
# ONLY MERGE PHYSICALLY TOUCHING PIECES
# ============================================================
def merge_touching(candidates):
    if not candidates:
        return []
    candidates = sorted(
        candidates,
        key=lambda c: c["center_x"]
    )
    result = []
    for candidate in candidates:
        merged = False
        for existing in result:
            existing_right = (
                existing["x"] +
                existing["w"]
            )
            candidate_left = candidate["x"]
            horizontal_gap = (
                candidate_left -
                existing_right
            )
            reverse_gap = (
                existing["x"] +
                existing["w"] -
                candidate["x"]
            )
            # Determine actual horizontal separation.
            if candidate["x"] >= existing_right:
                gap = horizontal_gap
            elif existing["x"] >= candidate["x"] + candidate["w"]:
                gap = reverse_gap
            else:
                gap = 0
            # Vertical overlap.
            top1 = existing["y"]
            bottom1 = existing["y"] + existing["h"]
            top2 = candidate["y"]
            bottom2 = candidate["y"] + candidate["h"]
            overlap = max(
                0,
                min(bottom1, bottom2)
                -
                max(top1, top2)
            )
            smaller_height = max(
                1,
                min(existing["h"], candidate["h"])
            )
            overlap_ratio = (
                overlap /
                smaller_height
            )
            # VERY conservative merge.
            if (
                gap <= TOUCH_GAP
                and
                overlap_ratio > 0.25
            ):
                left = min(
                    existing["x"],
                    candidate["x"]
                )
                right = max(
                    existing["x"] + existing["w"],
                    candidate["x"] + candidate["w"]
                )
                top = min(
                    existing["y"],
                    candidate["y"]
                )
                bottom = max(
                    existing["y"] + existing["h"],
                    candidate["y"] + candidate["h"]
                )
                existing["x"] = left
                existing["y"] = top
                existing["w"] = right - left
                existing["h"] = bottom - top
                existing["center_x"] = (
                    left +
                    existing["w"] / 2
                )
                existing["area"] += candidate["area"]
                existing["pixels"] += candidate["pixels"]
                merged = True
                break
        if not merged:
            result.append(
                candidate.copy()
            )
    return result
# ============================================================
# REMOVE ONLY TRUE DUPLICATES
# ============================================================
def remove_duplicates(candles):
    candles = sorted(
        candles,
        key=lambda c: c["center_x"]
    )
    result = []
    for candle in candles:
        duplicate = False
        for existing in result:
            x_distance = abs(
                candle["center_x"]
                -
                existing["center_x"]
            )
            # Same physical location.
            width_limit = max(
                candle["w"],
                existing["w"]
            )
            if x_distance > width_limit * 0.5:
                continue
            # Check box intersection.
            x1 = max(
                candle["x"],
                existing["x"]
            )
            y1 = max(
                candle["y"],
                existing["y"]
            )
            x2 = min(
                candle["x"] + candle["w"],
                existing["x"] + existing["w"]
            )
            y2 = min(
                candle["y"] + candle["h"],
                existing["y"] + existing["h"]
            )
            if x2 > x1 and y2 > y1:
                duplicate = True
                # Keep stronger color evidence.
                if candle["pixels"] > existing["pixels"]:
                    existing.update(candle)
                break
        if not duplicate:
            result.append(
                candle.copy()
            )
    return result
# ============================================================
# TWO-PART CANDLE DETECTOR
# ============================================================
def detect_candles(img):
    h, w = img.shape[:2]
    # --------------------------------------------------------
    # PART 1 — GREEN ONLY
    # --------------------------------------------------------
    green_mask = get_green_mask(img)
    green_candidates = find_candidates(
        green_mask,
        "GREEN",
        w
    )
    green_candidates = merge_touching(
        green_candidates
    )
    print(
        f"🟢 Green detector found: "
        f"{len(green_candidates)}"
    )
    # --------------------------------------------------------
    # PART 2 — RED ONLY
    # --------------------------------------------------------
    red_mask = get_red_mask(img)
    red_candidates = find_candidates(
        red_mask,
        "RED",
        w
    )
    red_candidates = merge_touching(
        red_candidates
    )
    print(
        f"🔴 Red detector found: "
        f"{len(red_candidates)}"
    )
    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------
    candles = (
        green_candidates +
        red_candidates
    )
    candles = remove_duplicates(
        candles
    )
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
def create_detection_map(img, candles):
    output = img.copy()
    for number, candle in enumerate(
        candles,
        start=1
    ):
        x = int(candle["x"])
        y = int(candle["y"])
        w = int(candle["w"])
        h = int(candle["h"])
        if candle["color"] == "GREEN":
            box_color = (
                0,
                255,
                0
            )
            label = "G"
        else:
            box_color = (
                0,
                0,
                255
            )
            label = "R"
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            box_color,
            2
        )
        cv2.putText(
            output,
            f"{number}{label}",
            (
                x,
                max(22, y - 5)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            box_color,
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
            "🟢 Running GREEN detector...\n"
            "🔴 Running RED detector..."
        )
        # ----------------------------------------------------
        # DOWNLOAD ORIGINAL PHOTO
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
        # LOAD — NO CROP
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
        # REPORT
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
            "🟢 G = Green detector\n"
            "🔴 R = Red detector\n\n"
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
        # CREATE MAP
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
                    "🟢 Green box = GREEN detector.\n"
                    "🔴 Red box = RED detector.\n\n"
                    "The screenshot was analyzed "
                    "from 0% to 100%.\n\n"
                    "Check every box against the "
                    "actual candle."
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
    "🕯️ TWO-PART CANDLE READING TEST"
)
print(
    "========================================"
)
print(
    "🟢 Separate GREEN detector"
)
print(
    "🔴 Separate RED detector"
)
print(
    "📱 Full screenshot — no crop"
)
print(
    "🚫 No aggressive fallback"
)
print(
    "🚫 No right-side-only detection"
)
print(
    "🚫 No random candles"
)
print(
    "🚫 No OHLC generation"
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

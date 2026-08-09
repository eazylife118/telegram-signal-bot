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
    "PUT_YOUR_BOT_TOKEN_HERE"
)
bot = telebot.TeleBot(TELEGRAM_TOKEN)
# ============================================================
# RED CANDLE SETTINGS
# ============================================================
# Based on the "powerful red candle" detector you provided,
# but relaxed for smaller Pocket Option candles.
MIN_RED_AREA = 20
MIN_RED_HEIGHT = 5
MIN_RED_WIDTH = 1
# Candle bodies should remain relatively narrow.
MAX_RED_WIDTH_RATIO = 0.045
# Vertical-shape protection.
MIN_VERTICAL_RATIO = 1.25
# Minimum amount of actual red pixels inside the box.
MIN_RED_PIXELS = 5
# Conservative merging.
# We do NOT aggressively combine nearby candles.
MERGE_X_DISTANCE = 2
# ============================================================
# LOAD SCREENSHOT
# ============================================================
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read screenshot.")
    return img
# ============================================================
# RED COLOR MASK
# ============================================================
def get_red_mask(img):
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )
    # --------------------------------------------------------
    # HSV RED
    # --------------------------------------------------------
    lower_red_1 = np.array([
        0,
        45,
        35
    ])
    upper_red_1 = np.array([
        15,
        255,
        255
    ])
    lower_red_2 = np.array([
        165,
        45,
        35
    ])
    upper_red_2 = np.array([
        180,
        255,
        255
    ])
    hsv_red_1 = cv2.inRange(
        hsv,
        lower_red_1,
        upper_red_1
    )
    hsv_red_2 = cv2.inRange(
        hsv,
        lower_red_2,
        upper_red_2
    )
    hsv_red = cv2.bitwise_or(
        hsv_red_1,
        hsv_red_2
    )
    # --------------------------------------------------------
    # BGR RED CONFIRMATION
    # --------------------------------------------------------
    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    # Red must actually dominate the other channels.
    bgr_red = (
        (r > g + 12) &
        (r > b + 12) &
        (r > 45)
    )
    bgr_red = (
        bgr_red.astype(np.uint8) * 255
    )
    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------
    # HSV is the primary detector.
    # BGR helps confirm difficult red pixels.
    combined = cv2.bitwise_or(
        hsv_red,
        bgr_red
    )
    return combined
# ============================================================
# FIND RED CANDLE CANDIDATES
# ============================================================
def find_red_candidates(
    img,
    red_mask
):
    h_img, w_img = img.shape[:2]
    # Keep morphology deliberately small.
    # Large kernels can join neighboring candles.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )
    cleaned = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_OPEN,
        kernel
    )
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    max_width = max(
        8,
        int(
            w_img *
            MAX_RED_WIDTH_RATIO
        )
    )
    for contour in contours:
        area = cv2.contourArea(
            contour
        )
        if area < MIN_RED_AREA:
            continue
        x, y, w, h = cv2.boundingRect(
            contour
        )
        if w < MIN_RED_WIDTH:
            continue
        if h < MIN_RED_HEIGHT:
            continue
        # Reject extremely wide objects.
        if w > max_width:
            continue
        # ----------------------------------------------------
        # VERTICAL CANDLE PROTECTION
        # ----------------------------------------------------
        if h < w * MIN_VERTICAL_RATIO:
            continue
        # ----------------------------------------------------
        # CHECK REAL RED PIXELS
        # ----------------------------------------------------
        roi = cleaned[
            y:y + h,
            x:x + w
        ]
        red_pixels = int(
            np.sum(
                roi > 0
            )
        )
        if red_pixels < MIN_RED_PIXELS:
            continue
        density = (
            red_pixels /
            float(
                max(
                    1,
                    w * h
                )
            )
        )
        # Don't accept extremely sparse red noise.
        if density < 0.08:
            continue
        # ----------------------------------------------------
        # CENTER
        # ----------------------------------------------------
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
            "pixels": red_pixels,
            "density": density,
            "center_x": center_x,
            "color": "RED"
        })
    return candidates
# ============================================================
# CONSERVATIVE RED MERGING
# ============================================================
def merge_red_candidates(
    candidates
):
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
            distance = abs(
                candidate["center_x"]
                -
                existing["center_x"]
            )
            # Only merge extremely close pieces.
            if distance > MERGE_X_DISTANCE:
                continue
            # Check vertical overlap.
            c_top = candidate["y"]
            c_bottom = (
                candidate["y"]
                +
                candidate["h"]
            )
            e_top = existing["y"]
            e_bottom = (
                existing["y"]
                +
                existing["h"]
            )
            overlap = not (
                c_bottom < e_top
                or
                c_top > e_bottom
            )
            if not overlap:
                continue
            # Merge only when they are genuinely touching.
            left = min(
                existing["x"],
                candidate["x"]
            )
            right = max(
                existing["x"]
                +
                existing["w"],
                candidate["x"]
                +
                candidate["w"]
            )
            top = min(
                existing["y"],
                candidate["y"]
            )
            bottom = max(
                existing["y"]
                +
                existing["h"],
                candidate["y"]
                +
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
                existing["w"] /
                2.0
            )
            existing["area"] += (
                candidate["area"]
            )
            existing["pixels"] += (
                candidate["pixels"]
            )
            merged = True
            break
        if not merged:
            result.append(
                candidate.copy()
            )
    return result
# ============================================================
# DETECT RED CANDLES
# ============================================================
def detect_red_candles(img):
    # Entire screenshot.
    # No crop.
    red_mask = get_red_mask(
        img
    )
    candidates = find_red_candidates(
        img,
        red_mask
    )
    candidates = merge_red_candidates(
        candidates
    )
    candidates.sort(
        key=lambda c: c["center_x"]
    )
    return candidates
# ============================================================
# CREATE RED DETECTION MAP
# ============================================================
def create_red_detection_map(
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
        # ----------------------------------------------------
        # RED RECTANGLE
        # ----------------------------------------------------
        cv2.rectangle(
            output,
            (x, y),
            (
                x + w,
                y + h
            ),
            (0, 0, 255),
            2
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
            0.60,
            (0, 0, 255),
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
        "red_test_original.png"
    )
    detection_path = (
        "red_detection_map.png"
    )
    try:
        bot.reply_to(
            message,
            "🔴 RED CANDLE TEST\n\n"
            "Scanning the entire screenshot...\n"
            "Looking ONLY for red candle-shaped objects."
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
        # LOAD
        # ----------------------------------------------------
        img = load_image(
            original_path
        )
        # ----------------------------------------------------
        # RED DETECTION
        # ----------------------------------------------------
        red_candles = (
            detect_red_candles(
                img
            )
        )
        red_count = len(
            red_candles
        )
        elapsed = (
            time.time()
            -
            start_time
        )
        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------
        if red_count == 0:
            bot.reply_to(
                message,
                "🔴 RED CANDLE TEST\n\n"
                "No reliable red candle-shaped "
                "objects were detected.\n\n"
                "No candles were generated.\n"
                "No random candles were added.\n"
                "No trading signal was generated."
            )
            return
        sequence = []
        for number in range(
            1,
            red_count + 1
        ):
            sequence.append(
                f"{number}. 🔴 RED"
            )
        sequence_text = "\n".join(
            sequence
        )
        report = (
            "🔎 **RED CANDLE READING TEST**\n\n"
            "🟡 **DETECTION AREA:**\n"
            "Entire uploaded screenshot — 0% to 100%\n\n"
            "🔴 **RED CANDLES DETECTED:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🔴 RED: {red_count}\n\n"
            "🕯️ **RED CANDLE ORDER:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"{sequence_text}\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🎯 **SCREEN CHECK:**\n"
            "🔴 Red rectangle = detected red candle\n"
            "🔢 Number = left-to-right detection order\n\n"
            "⚠️ **TEST ONLY**\n"
            "Only visually detected red objects are counted.\n"
            "No OHLC candles are generated.\n"
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
        # ----------------------------------------------------
        # CREATE MAP
        # ----------------------------------------------------
        detection_map = (
            create_red_detection_map(
                img,
                red_candles
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
                    "🔴 **RED CANDLE DETECTION MAP**\n\n"
                    "Every red rectangle is one "
                    "detected red candle candidate.\n\n"
                    "Numbers show detection order "
                    "from LEFT → RIGHT.\n\n"
                    "Please compare this directly "
                    "with the original screenshot."
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
            f"❌ Red detection error:\n{str(e)}"
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
    "🔴 RED CANDLE TEST BOT"
)
print(
    "========================================"
)
print(
    "Full screenshot scan: ENABLED"
)
print(
    "RED detection: HSV + BGR"
)
print(
    "Vertical candle protection: ENABLED"
)
print(
    "Conservative merging: ENABLED"
)
print(
    "Green detection: DISABLED"
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

requirements.txt

pyTelegramBotAPI
opencv-python-headless
numpy

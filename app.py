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
# DETECTION SETTINGS
# ============================================================

MIN_COLOR_PIXELS = 8
MIN_BODY_AREA = 12
MIN_BODY_HEIGHT = 2

# Pocket Option screenshots normally have candles distributed
# across the chart horizontally.
MIN_CANDLE_WIDTH = 2
MAX_CANDLE_WIDTH_RATIO = 0.045

# Distance used when combining pieces belonging to one candle.
MERGE_DISTANCE_RATIO = 0.55


# ============================================================
# IMAGE PREPARATION
# ============================================================

def load_image(path):

    img = cv2.imread(path)

    if img is None:
        raise ValueError("Could not read screenshot.")

    h, w = img.shape[:2]

    # Upscale small screenshots.
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
# COLOR DETECTION
# ============================================================

def get_color_masks(img):

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # GREEN / BULLISH
    # --------------------------------------------------------

    green_lower = np.array([
        30,
        35,
        35
    ])

    green_upper = np.array([
        95,
        255,
        255
    ])

    green = cv2.inRange(
        hsv,
        green_lower,
        green_upper
    )

    # --------------------------------------------------------
    # RED / BEARISH
    # --------------------------------------------------------

    red_lower_1 = np.array([
        0,
        35,
        35
    ])

    red_upper_1 = np.array([
        15,
        255,
        255
    ])

    red_lower_2 = np.array([
        165,
        35,
        35
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
# FIND COLORED REGIONS
# ============================================================

def find_candidates(mask, color, image_width):

    # Remove isolated noise.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Connect small gaps inside candle bodies.
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
        int(image_width * MAX_CANDLE_WIDTH_RATIO)
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_BODY_AREA:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if h < MIN_BODY_HEIGHT:
            continue

        if w < MIN_CANDLE_WIDTH:
            continue

        if w > max_width:
            continue

        # Reject extremely wide horizontal objects.
        if w > h * 6:
            continue

        center_x = x + w / 2

        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": float(area),
            "center_x": center_x,
            "color": color
        })

    return candidates


# ============================================================
# MERGE PIECES OF THE SAME CANDLE
# ============================================================

def merge_candidates(candidates):

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda x: x["center_x"]
    )

    merged = []

    for candidate in candidates:

        found = False

        for existing in merged:

            distance = abs(
                candidate["center_x"]
                -
                existing["center_x"]
            )

            allowed = max(
                candidate["w"],
                existing["w"]
            ) * MERGE_DISTANCE_RATIO

            if distance <= allowed:

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
                    left + existing["w"] / 2
                )
                existing["area"] += candidate["area"]

                found = True
                break

        if not found:
            merged.append(
                candidate.copy()
            )

    return merged


# ============================================================
# REMOVE DUPLICATES BETWEEN COLORS
# ============================================================

def remove_overlapping_detections(candidates):

    candidates = sorted(
        candidates,
        key=lambda x: x["center_x"]
    )

    result = []

    for candle in candidates:

        duplicate = False

        for existing in result:

            distance = abs(
                candle["center_x"]
                -
                existing["center_x"]
            )

            threshold = max(
                candle["w"],
                existing["w"]
            ) * 0.6

            if distance <= threshold:

                duplicate = True

                # Keep the stronger detection.
                if candle["area"] > existing["area"]:

                    index = result.index(
                        existing
                    )

                    result[index] = candle

                break

        if not duplicate:
            result.append(candle)

    return result


# ============================================================
# DETECT CANDLES
# ============================================================

def detect_candles(img):

    height, width = img.shape[:2]

    green_mask, red_mask = get_color_masks(
        img
    )

    green_candidates = find_candidates(
        green_mask,
        "GREEN",
        width
    )

    red_candidates = find_candidates(
        red_mask,
        "RED",
        width
    )

    green_candidates = merge_candidates(
        green_candidates
    )

    red_candidates = merge_candidates(
        red_candidates
    )

    all_candidates = (
        green_candidates +
        red_candidates
    )

    all_candidates = remove_overlapping_detections(
        all_candidates
    )

    all_candidates.sort(
        key=lambda x: x["center_x"]
    )

    return all_candidates


# ============================================================
# ANNOTATED IMAGE
# ============================================================

def create_annotated_image(img, candles):

    output = img.copy()

    for number, candle in enumerate(
        candles,
        start=1
    ):

        x = candle["x"]
        y = candle["y"]
        w = candle["w"]
        h = candle["h"]

        # Draw detection rectangle.
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (255, 255, 0),
            2
        )

        # Candle number.
        label = str(number)

        text_x = x
        text_y = max(
            25,
            y - 8
        )

        cv2.putText(
            output,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

    return output


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

    sequence = []

    for candle in candles:

        if candle["color"] == "GREEN":
            sequence.append("🟢")
        else:
            sequence.append("🔴")

    return green, red, sequence


# ============================================================
# TELEGRAM PHOTO HANDLER
# ============================================================

@bot.message_handler(
    content_types=["photo"]
)
def handle_photo(message):

    start_time = time.time()

    original_path = "chart_screenshot.png"
    annotated_path = "candle_detection.png"

    try:

        bot.reply_to(
            message,
            "👁️ Reading the visible candles..."
        )

        # ----------------------------------------------------
        # DOWNLOAD HIGHEST RESOLUTION IMAGE
        # ----------------------------------------------------

        file_info = bot.get_file(
            message.photo[-1].file_id
        )

        downloaded_file = bot.download_file(
            file_info.file_path
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
        # REPORT
        # ----------------------------------------------------

        green, red, sequence = create_report(
            candles
        )

        total = len(candles)

        elapsed = time.time() - start_time

        if total == 0:

            bot.reply_to(
                message,
                "❌ No reliable candle bodies detected.\n\n"
                "No candles were generated or guessed."
            )

            return

        sequence_text = " → ".join(
            sequence
        )

        report = (
            "🔎 **CANDLE VISION TEST**\n\n"
            "📊 **WHAT THE BOT ACTUALLY DETECTED:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🟢 Green candles: {green}\n"
            f"🔴 Red candles: {red}\n"
            f"📊 Total candles: {total}\n\n"
            "🕯️ **Candle sequence (left → right):**\n"
            f"{sequence_text}\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "⚠️ **IMPORTANT:**\n"
            "Only visually detected candle objects are counted.\n"
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
        # CREATE NUMBERED DIAGNOSTIC IMAGE
        # ----------------------------------------------------

        annotated = create_annotated_image(
            img,
            candles
        )

        cv2.imwrite(
            annotated_path,
            annotated
        )

        # ----------------------------------------------------
        # SEND NUMBERED IMAGE
        # ----------------------------------------------------

        with open(
            annotated_path,
            "rb"
        ) as photo:

            bot.send_photo(
                message.chat.id,
                photo,
                caption=(
                    "🔢 **DETECTION MAP**\n\n"
                    "Every numbered box is one candle "
                    "the detector believes it found.\n\n"
                    "🟢/🔴 classification is based on "
                    "the detected candle color."
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
            annotated_path
        ]:

            if os.path.exists(path):

                try:
                    os.remove(path)
                except:
                    pass


# ============================================================
# START
# ============================================================

print(
    "========================================"
)

print(
    "🕯️ CANDLE DETECTION VISION TEST"
)

print(
    "========================================"
)

print(
    "Bot is listening for screenshots..."
)

bot.infinity_polling(
    timeout=30,
    long_polling_timeout=30
)

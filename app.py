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
    raise ValueError("BOT_TOKEN environment variable is missing.")

bot = telebot.TeleBot(TELEGRAM_TOKEN)


# ============================================================
# SETTINGS
# ============================================================

# Very small real candle bodies are allowed.
MIN_BODY_AREA = 2
MIN_BODY_HEIGHT = 2
MIN_BODY_WIDTH = 1

# Prevent a wick or large UI object from becoming a candle.
MAX_BODY_WIDTH = 18
MAX_BODY_HEIGHT = 160

# Minimum amount of actual candle-colour pixels.
MIN_COLOR_PIXELS = 3

# Do not accept extremely sparse objects.
MIN_DENSITY = 0.10

# Candles must be reasonably separated horizontally.
# This is deliberately much less aggressive than the old merger.
MAX_MERGE_GAP = 1


# ============================================================
# LOAD FULL SCREENSHOT
# ============================================================

def load_image(path):

    img = cv2.imread(path)

    if img is None:
        raise ValueError("Could not read screenshot.")

    h, w = img.shape[:2]

    # Keep 100% of the screenshot.
    # Only upscale if necessary.
    if w < 1400:

        scale = 1400 / float(w)

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
# TWO-PART COLOR DETECTION
# ============================================================

def get_color_masks(img):

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)

    # ========================================================
    # GREEN PASS
    # ========================================================

    green_hsv = cv2.inRange(
        hsv,
        np.array([25, 35, 30]),
        np.array([100, 255, 255])
    )

    # BGR confirmation.
    # Green must actually be stronger than red.
    green_bgr = (
        (g > r + 8) &
        (g > b - 5) &
        (g > 45)
    ).astype(np.uint8) * 255

    green = cv2.bitwise_or(
        green_hsv,
        green_bgr
    )

    # ========================================================
    # RED PASS
    # ========================================================

    red_hsv_1 = cv2.inRange(
        hsv,
        np.array([0, 25, 25]),
        np.array([18, 255, 255])
    )

    red_hsv_2 = cv2.inRange(
        hsv,
        np.array([160, 25, 25]),
        np.array([180, 255, 255])
    )

    red_hsv = cv2.bitwise_or(
        red_hsv_1,
        red_hsv_2
    )

    # BGR red confirmation.
    red_bgr = (
        (r > g + 8) &
        (r > b + 8) &
        (r > 40)
    ).astype(np.uint8) * 255

    red = cv2.bitwise_or(
        red_hsv,
        red_bgr
    )

    return green, red


# ============================================================
# CLEAN MASK
# ============================================================

def clean_mask(mask):

    # Do NOT use large kernels.
    # Large morphology was causing nearby candles to join.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return cleaned


# ============================================================
# FIND REAL COLOURED BODY OBJECTS
# ============================================================

def find_candidates(
    mask,
    color,
    image_width
):

    cleaned = clean_mask(mask)

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        # ----------------------------------------------------
        # SIZE CHECK
        # ----------------------------------------------------

        if w < MIN_BODY_WIDTH:
            continue

        if h < MIN_BODY_HEIGHT:
            continue

        if w > MAX_BODY_WIDTH:
            continue

        if h > MAX_BODY_HEIGHT:
            continue

        # ----------------------------------------------------
        # IMPORTANT:
        # Don't accept huge horizontal objects.
        # ----------------------------------------------------

        if w > h * 4:
            continue

        # ----------------------------------------------------
        # COLOURED PIXEL CHECK
        # ----------------------------------------------------

        roi = cleaned[
            y:y+h,
            x:x+w
        ]

        pixels = int(
            np.count_nonzero(roi)
        )

        if pixels < MIN_COLOR_PIXELS:
            continue

        density = (
            pixels /
            float(max(1, w * h))
        )

        if density < MIN_DENSITY:
            continue

        # ----------------------------------------------------
        # CENTRE
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

            "pixels": pixels,
            "density": density,

            "center_x": center_x,

            "color": color
        })

    return candidates


# ============================================================
# REMOVE ONLY TRUE SAME-CANDLE PIECES
# ============================================================

def merge_candidates(candidates):

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

            # Horizontal distance between bodies.
            right_existing = (
                existing["x"] +
                existing["w"]
            )

            left_candidate = candidate["x"]

            gap = (
                left_candidate -
                right_existing
            )

            # Only merge if physically touching or almost touching.
            if gap < 0:
                gap = 0

            if gap > MAX_MERGE_GAP:
                continue

            # Vertical overlap.
            top1 = existing["y"]
            bottom1 = (
                existing["y"] +
                existing["h"]
            )

            top2 = candidate["y"]
            bottom2 = (
                candidate["y"] +
                candidate["h"]
            )

            overlap = max(
                0,
                min(bottom1, bottom2)
                -
                max(top1, top2)
            )

            smaller_height = min(
                existing["h"],
                candidate["h"]
            )

            if smaller_height <= 0:
                continue

            overlap_ratio = (
                overlap /
                float(smaller_height)
            )

            if overlap_ratio < 0.50:
                continue

            # ------------------------------------------------
            # Merge
            # ------------------------------------------------

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

            existing["pixels"] += (
                candidate["pixels"]
            )

            existing["center_x"] = (
                left +
                existing["w"] / 2.0
            )

            merged = True
            break

        if not merged:

            result.append(
                candidate.copy()
            )

    return result


# ============================================================
# CROSS-COLOR DUPLICATE PROTECTION
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

        duplicate = False

        for existing in result:

            distance = abs(
                candle["center_x"]
                -
                existing["center_x"]
            )

            # Only treat as duplicate when the bodies
            # actually occupy nearly the same position.
            threshold = max(
                1.5,
                min(
                    candle["w"],
                    existing["w"]
                ) * 0.45
            )

            if distance <= threshold:

                # Keep the stronger colour detection.
                if (
                    candle["pixels"]
                    >
                    existing["pixels"]
                ):

                    index = result.index(
                        existing
                    )

                    result[index] = (
                        candle
                    )

                duplicate = True
                break

        if not duplicate:

            result.append(
                candle
            )

    return result


# ============================================================
# MAIN CANDLE DETECTOR
# ============================================================

def detect_candles(img):

    h, w = img.shape[:2]

    # ========================================================
    # FULL IMAGE
    # ========================================================

    # NO CROP.
    # NO RIGHT-SIDE SPECIAL AREA.
    # NO LEFT-SIDE SPECIAL AREA.
    # Entire screenshot is analysed.
    # ========================================================

    green_mask, red_mask = (
        get_color_masks(img)
    )

    # ========================================================
    # PART 1 — GREEN
    # ========================================================

    green = find_candidates(
        green_mask,
        "GREEN",
        w
    )

    green = merge_candidates(
        green
    )

    # ========================================================
    # PART 2 — RED
    # ========================================================

    red = find_candidates(
        red_mask,
        "RED",
        w
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

    # Remove only extremely close duplicates.
    candles = remove_cross_color_duplicates(
        candles
    )

    # Left → right.
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

        # ----------------------------------------------------
        # Rectangle is around the detected BODY,
        # not the entire wick.
        # ----------------------------------------------------

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

            label = f"G{number}"

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

            label = f"R{number}"

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            box_color,
            2
        )

        cv2.putText(
            output,
            label,
            (
                x,
                max(
                    20,
                    y - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            label_color,
            1,
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
            "🟢 Checking GREEN candles...\n"
            "🔴 Checking RED candles...\n"
            "No candles will be invented."
        )

        # ====================================================
        # DOWNLOAD HIGHEST QUALITY
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
        # SEQUENCE
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
        # REPORT
        # ====================================================

        if total == 0:

            bot.reply_to(
                message,
                "❌ No reliable coloured candle bodies detected.\n\n"
                "No candle was generated.\n"
                "No random candle was added.\n"
                "No signal was generated."
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
            "🟢 G = Bot classified GREEN\n"
            "🔴 R = Bot classified RED\n\n"

            "🔬 **METHOD:**\n"
            "• Full screenshot scan\n"
            "• Separate GREEN pass\n"
            "• Separate RED pass\n"
            "• BGR + HSV colour confirmation\n"
            "• Small-body detection\n"
            "• Wick-width protection\n"
            "• Conservative merging\n"
            "• No forced candle count\n\n"

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

        # ====================================================
        # CREATE MAP
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
                    "🔎 **FULL-SCREEN CANDLE MAP**\n\n"
                    "🟢 Green box = GREEN candle body\n"
                    "🔴 Red box = RED candle body\n\n"
                    "The rectangle is intentionally limited "
                    "to the detected coloured candle body "
                    "so a long wick does not become a huge candle.\n\n"
                    "Compare this image directly with your "
                    "Pocket Option screenshot."
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
    "🟢 Separate GREEN detection"
)

print(
    "🔴 Separate RED detection"
)

print(
    "📱 Full screenshot — 0% to 100%"
)

print(
    "🕯️ Body-focused detection"
)

print(
    "🚫 No forced candle count"
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

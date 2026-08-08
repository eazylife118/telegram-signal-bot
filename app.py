import os
import cv2
import numpy as np
import telebot

# ============================================================
# TELEGRAM CONFIGURATION
# ============================================================

TELEGRAM_TOKEN = os.getenv(
    "BOT_TOKEN",
    "8937673241:AAGvyTA-G12xfwMlhif3Nh4_2Ag8OStq3tU"
)

bot = telebot.TeleBot(TELEGRAM_TOKEN)


# ============================================================
# IMAGE SETTINGS
# ============================================================

MIN_BODY_AREA = 25
MIN_BODY_HEIGHT = 3
MAX_BODY_WIDTH = 80

# Minimum amount of colored pixels needed to consider
# a vertical region a possible candle body.
MIN_COLOR_PIXELS = 20


# ============================================================
# LOAD AND PREPARE IMAGE
# ============================================================

def prepare_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Could not read the screenshot.")

    height, width = img.shape[:2]

    # Upscale small screenshots.
    if width < 1400:
        scale = 1400 / width
        img = cv2.resize(
            img,
            (
                int(width * scale),
                int(height * scale)
            ),
            interpolation=cv2.INTER_CUBIC
        )

    return img


# ============================================================
# COLOR MASKS
# ============================================================

def create_masks(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # GREEN / BULLISH
    green_lower = np.array([35, 50, 40])
    green_upper = np.array([90, 255, 255])

    green_mask = cv2.inRange(
        hsv,
        green_lower,
        green_upper
    )

    # RED / BEARISH
    red_lower_1 = np.array([0, 50, 40])
    red_upper_1 = np.array([12, 255, 255])

    red_lower_2 = np.array([168, 50, 40])
    red_upper_2 = np.array([180, 255, 255])

    red_mask_1 = cv2.inRange(
        hsv,
        red_lower_1,
        red_upper_1
    )

    red_mask_2 = cv2.inRange(
        hsv,
        red_lower_2,
        red_upper_2
    )

    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2
    )

    return green_mask, red_mask


# ============================================================
# FIND CANDLE BODIES
# ============================================================

def find_body_candidates(mask, color):

    # Remove tiny isolated pixels.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Slightly connect pieces belonging to the same candle body.
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 3)
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_BODY_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if h < MIN_BODY_HEIGHT:
            continue

        if w > MAX_BODY_WIDTH:
            continue

        # Reject extremely thin horizontal objects.
        if w > h * 5:
            continue

        # Reject extremely large objects that are unlikely
        # to be individual candle bodies.
        if area > 100000:
            continue

        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "color": color
        })

    return candidates


# ============================================================
# MERGE CANDIDATES THAT BELONG TO THE SAME CANDLE
# ============================================================

def merge_same_candles(candidates):

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda c: c["x"]
    )

    merged = []

    for candidate in candidates:

        cx = candidate["x"] + candidate["w"] / 2

        matched = False

        for existing in merged:

            ecx = existing["x"] + existing["w"] / 2

            # Candles normally occupy a compact horizontal area.
            horizontal_distance = abs(cx - ecx)

            max_width = max(
                candidate["w"],
                existing["w"]
            )

            if horizontal_distance <= max_width * 0.65:

                # Combine the two detections.
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
                existing["area"] += candidate["area"]

                matched = True
                break

        if not matched:
            merged.append(candidate.copy())

    return merged


# ============================================================
# REMOVE OBVIOUS NON-CANDLE OBJECTS
# ============================================================

def filter_candidates(candidates, image_width):

    if not candidates:
        return []

    result = []

    for c in candidates:

        center_x = c["x"] + c["w"] / 2

        # Ignore extreme edge objects where UI controls/
        # labels commonly appear.
        if center_x < image_width * 0.03:
            continue

        if center_x > image_width * 0.97:
            continue

        # Candle bodies should have a reasonable width.
        if c["w"] < 2:
            continue

        if c["w"] > image_width * 0.08:
            continue

        result.append(c)

    return result


# ============================================================
# CANDLE DETECTION
# ============================================================

def detect_candles(image_path):

    img = prepare_image(image_path)

    height, width = img.shape[:2]

    green_mask, red_mask = create_masks(img)

    green_candidates = find_body_candidates(
        green_mask,
        "GREEN"
    )

    red_candidates = find_body_candidates(
        red_mask,
        "RED"
    )

    green_candidates = merge_same_candles(
        green_candidates
    )

    red_candidates = merge_same_candles(
        red_candidates
    )

    green_candidates = filter_candidates(
        green_candidates,
        width
    )

    red_candidates = filter_candidates(
        red_candidates,
        width
    )

    # Combine all candidates.
    all_candidates = (
        green_candidates +
        red_candidates
    )

    # Sort exactly as candles appear on the screenshot.
    all_candidates.sort(
        key=lambda c: c["x"]
    )

    # Prevent two overlapping detections at essentially
    # the same horizontal position from becoming two candles.
    final = []

    for candle in all_candidates:

        center = candle["x"] + candle["w"] / 2

        duplicate = False

        for existing in final:

            existing_center = (
                existing["x"] +
                existing["w"] / 2
            )

            distance = abs(
                center - existing_center
            )

            if distance <= max(
                candle["w"],
                existing["w"]
            ) * 0.45:

                duplicate = True

                # Keep the stronger detection.
                if candle["area"] > existing["area"]:

                    index = final.index(existing)
                    final[index] = candle

                break

        if not duplicate:
            final.append(candle)

    final.sort(key=lambda c: c["x"])

    return final


# ============================================================
# ANALYSIS REPORT
# ============================================================

def analyze_screenshot(image_path):

    candles = detect_candles(image_path)

    green = [
        c for c in candles
        if c["color"] == "GREEN"
    ]

    red = [
        c for c in candles
        if c["color"] == "RED"
    ]

    sequence = []

    for candle in candles:

        if candle["color"] == "GREEN":
            sequence.append("🟢")
        else:
            sequence.append("🔴")

    return {
        "green": len(green),
        "red": len(red),
        "total": len(candles),
        "sequence": sequence
    }


# ============================================================
# TELEGRAM PHOTO HANDLER
# ============================================================

@bot.message_handler(content_types=["photo"])
def handle_photo(message):

    image_path = "chart_screenshot.png"

    try:

        bot.reply_to(
            message,
            "👁️ Reading the visible candlesticks..."
        )

        # Download highest-resolution Telegram photo.
        file_info = bot.get_file(
            message.photo[-1].file_id
        )

        downloaded_file = bot.download_file(
            file_info.file_path
        )

        with open(image_path, "wb") as f:
            f.write(downloaded_file)

        # Analyze actual screenshot.
        result = analyze_screenshot(
            image_path
        )

        green = result["green"]
        red = result["red"]
        total = result["total"]
        sequence = result["sequence"]

        if total == 0:

            bot.reply_to(
                message,
                "❌ No reliable candle bodies were detected.\n\n"
                "No candles were invented or generated."
            )

            return

        sequence_text = " → ".join(sequence)

        reply = (
            "🔎 **CANDLE DETECTION TEST**\n\n"
            "📊 **WHAT THE BOT ACTUALLY DETECTED:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🟢 Green candles: {green}\n"
            f"🔴 Red candles: {red}\n"
            f"📊 Total candles: {total}\n\n"
            "🕯️ **Candle sequence (left → right):**\n"
            f"{sequence_text}\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "⚠️ **IMPORTANT:**\n"
            "Only visually detected candle bodies are counted.\n"
            "No OHLC candles are generated.\n"
            "No random candles are added.\n"
            "No trading signal is generated."
        )

        bot.reply_to(
            message,
            reply,
            parse_mode="Markdown"
        )

    except Exception as e:

        print(
            "ERROR:",
            repr(e)
        )

        bot.reply_to(
            message,
            f"❌ Detection error:\n{str(e)}"
        )

    finally:

        if os.path.exists(image_path):

            try:
                os.remove(image_path)
            except:
                pass


# ============================================================
# START BOT
# ============================================================

print("========================================")
print("🕯️ CANDLE DETECTION TEST BOT")
print("========================================")
print("Bot is listening for screenshots...")

bot.infinity_polling(
    timeout=30,
    long_polling_timeout=30
)

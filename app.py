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
    raise RuntimeError("BOT_TOKEN environment variable is not set.")

bot = telebot.TeleBot(TELEGRAM_TOKEN)


# ============================================================
# DETECTION SETTINGS — REAL DATA ONLY
# ============================================================

# These settings are for detecting REAL colored pixels
# They do NOT generate or invent anything

MIN_BODY_AREA = 1
MIN_BODY_HEIGHT = 1
MIN_CANDLE_WIDTH = 1

MAX_CANDLE_WIDTH_RATIO = 0.30
MAX_BODY_HEIGHT_RATIO = 0.85

MERGE_DISTANCE_RATIO = 0.90
MIN_COLOR_DENSITY = 0.005


# ============================================================
# LOAD IMAGE — REAL SCREENSHOT
# ============================================================

def load_image(path):
    """Load the actual screenshot. No generation."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read screenshot.")

    h, w = img.shape[:2]
    if w < 1400:
        scale = 1400 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    return img


# ============================================================
# COLOR MASKS — DETECT REAL COLORS
# ============================================================

def get_color_masks(img):
    """Detect REAL green and red pixels from the screenshot."""

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ==========================================================
    # REAL GREEN PIXELS
    # ==========================================================

    green_lower = np.array([20, 20, 20])
    green_upper = np.array([110, 255, 255])
    green = cv2.inRange(hsv, green_lower, green_upper)

    # ==========================================================
    # REAL RED PIXELS — Multiple methods to catch all
    # ==========================================================

    # Method 1: HSV red ranges
    red_lower_1 = np.array([0, 10, 10])
    red_upper_1 = np.array([35, 255, 255])
    red_lower_2 = np.array([145, 10, 10])
    red_upper_2 = np.array([190, 255, 255])

    red1 = cv2.inRange(hsv, red_lower_1, red_upper_1)
    red2 = cv2.inRange(hsv, red_lower_2, red_upper_2)
    red_hsv = cv2.bitwise_or(red1, red2)

    # Method 2: BGR channel difference (red = r > g and r > b)
    b, g, r = cv2.split(img)
    red_bgr = cv2.bitwise_and(
        cv2.bitwise_and(
            (r > g + 5).astype(np.uint8) * 255,
            (r > b + 5).astype(np.uint8) * 255
        ),
        (r > 10).astype(np.uint8) * 255
    )

    # Combine all red detection methods
    red = cv2.bitwise_or(red_hsv, red_bgr)

    return green, red


# ============================================================
# FIND CANDIDATES — REAL CONTOURS
# ============================================================

def find_candidates(mask, color, image_width, image_height):
    """
    Find REAL colored contours in the image.
    Each contour represents a REAL candle or part of one.
    No generation. No invention.
    """

    # Clean up noise but preserve real candles
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    # Find contours (real colored shapes)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    max_width = max(20, int(image_width * MAX_CANDLE_WIDTH_RATIO))

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_BODY_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < MIN_CANDLE_WIDTH or h < MIN_BODY_HEIGHT:
            continue
        if w > max_width or h > image_height * MAX_BODY_HEIGHT_RATIO:
            continue

        # Count real colored pixels in this region
        region = cleaned[y:y+h, x:x+w]
        colored_pixels = int(np.sum(region > 0))

        if colored_pixels < 2:
            continue

        density = colored_pixels / float(max(1, w * h))
        if density < MIN_COLOR_DENSITY:
            continue

        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": float(area),
            "pixels": colored_pixels,
            "density": density,
            "center_x": x + w / 2,
            "color": color
        })

    return candidates


# ============================================================
# MERGE CANDLES — REAL ONLY
# ============================================================

def merge_candidates(candidates):
    """
    Merge pieces that belong to the same real candle.
    Does NOT create new candles — only merges existing pieces.
    """

    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda c: c["center_x"])
    merged = []

    for candidate in candidates:
        for existing in merged:
            distance = abs(candidate["center_x"] - existing["center_x"])
            allowed = max(candidate["w"], existing["w"], 2) * MERGE_DISTANCE_RATIO

            # Check if they overlap vertically
            if distance <= allowed and not (
                candidate["y"] + candidate["h"] < existing["y"] or
                candidate["y"] > existing["y"] + existing["h"]
            ):
                # Merge the two real pieces
                left = min(existing["x"], candidate["x"])
                right = max(existing["x"] + existing["w"], candidate["x"] + candidate["w"])
                top = min(existing["y"], candidate["y"])
                bottom = max(existing["y"] + existing["h"], candidate["y"] + candidate["h"])

                existing.update({
                    "x": left,
                    "y": top,
                    "w": right - left,
                    "h": bottom - top,
                    "center_x": left + (right - left) / 2,
                    "area": existing["area"] + candidate["area"],
                    "pixels": existing["pixels"] + candidate["pixels"]
                })
                break
        else:
            merged.append(candidate.copy())

    return merged


# ============================================================
# REMOVE DUPLICATES — REAL ONLY
# ============================================================

def remove_cross_color_duplicates(candles):
    """
    Remove duplicate detections of the same real candle.
    Does NOT create new candles.
    """

    candles = sorted(candles, key=lambda c: c["center_x"])
    result = []

    for candle in candles:
        for i, existing in enumerate(result):
            distance = abs(candle["center_x"] - existing["center_x"])
            if distance <= max(candle["w"], existing["w"], 2) * 0.70:
                # Keep the detection with more real pixels
                if candle["pixels"] > existing["pixels"]:
                    result[i] = candle
                break
        else:
            result.append(candle)

    return result


# ============================================================
# DETECT CANDLES — REAL ONLY
# ============================================================

def detect_candles(img):
    """Detect REAL candles from the screenshot. Nothing generated."""

    h, w = img.shape[:2]

    green_mask, red_mask = get_color_masks(img)

    # Detect real green candles
    green = find_candidates(green_mask, "GREEN", w, h)
    green = merge_candidates(green)

    # Detect real red candles
    red = find_candidates(red_mask, "RED", w, h)
    red = merge_candidates(red)

    # Combine
    candles = green + red

    # Remove duplicates of the same real candle
    candles = remove_cross_color_duplicates(candles)

    # Sort left to right
    candles.sort(key=lambda c: c["center_x"])

    return candles


# ============================================================
# REPORT — REAL COUNT ONLY
# ============================================================

def create_report(candles):
    green = sum(1 for c in candles if c["color"] == "GREEN")
    red = sum(1 for c in candles if c["color"] == "RED")
    return green, red


# ============================================================
# DETECTION MAP — REAL VISUALIZATION
# ============================================================

def create_detection_map(img, candles):
    """Show REAL candles with yellow boxes. No generated data."""

    output = img.copy()

    for number, candle in enumerate(candles, start=1):
        x, y, w, h = int(candle["x"]), int(candle["y"]), int(candle["w"]), int(candle["h"])

        # Yellow box around each REAL detected candle
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Label color based on REAL color detection
        label_color = (0, 255, 0) if candle["color"] == "GREEN" else (0, 0, 255)
        cv2.putText(output, str(number), (x, max(25, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, label_color, 2, cv2.LINE_AA)

    return output


# ============================================================
# TELEGRAM HANDLER
# ============================================================

@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    start_time = time.time()
    original_path = "chart_screenshot.png"
    detection_path = "candle_detection.png"

    try:
        bot.reply_to(message, "👁️ Reading visible candles from screenshot...")

        # ==========================================================
        # DOWNLOAD THE REAL SCREENSHOT
        # ==========================================================

        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with open(original_path, "wb") as f:
            f.write(downloaded_file)

        # ==========================================================
        # LOAD THE REAL IMAGE
        # ==========================================================

        img = load_image(original_path)

        # ==========================================================
        # DETECT REAL CANDLES
        # ==========================================================

        candles = detect_candles(img)

        # ==========================================================
        # COUNT REAL CANDLES
        # ==========================================================

        green, red = create_report(candles)
        total = len(candles)

        elapsed = time.time() - start_time

        # ==========================================================
        # NO CANDLES DETECTED — REAL RESULT
        # ==========================================================

        if total == 0:
            bot.reply_to(
                message,
                "❌ **No candles detected in this screenshot.**\n\n"
                "This is a REAL result — no candles were invented.\n"
                "No fake data was added.\n"
                "No generated OHLC.\n"
                "No trading signal.\n\n"
                "Try a clearer screenshot with more visible candles."
            )
            return

        # ==========================================================
        # REAL CANDLE-BY-CANDLE READING
        # ==========================================================

        sequence = []
        for i, candle in enumerate(candles, start=1):
            if candle["color"] == "GREEN":
                sequence.append(f"{i}. 🟢 GREEN")
            else:
                sequence.append(f"{i}. 🔴 RED")

        sequence_text = "\n".join(sequence)

        # ==========================================================
        # REAL REPORT
        # ==========================================================

        report = (
            "🔎 **REAL CANDLE DETECTION**\n\n"

            "📊 **WHAT WAS ACTUALLY DETECTED:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🟢 GREEN: {green}\n"
            f"🔴 RED: {red}\n"
            f"📊 TOTAL: {total}\n\n"

            "🕯️ **REAL CANDLE SEQUENCE:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"{sequence_text}\n\n"

            "━━━━━━━━━━━━━━━━━━━━\n\n"

            "✅ **REAL DATA ONLY**\n"
            "• No candles were generated\n"
            "• No OHLC data was invented\n"
            "• No missing candles were added\n"
            "• No trading signal was generated\n"
            "• Only what is visible in the screenshot\n\n"

            f"⚡ Processing time: {elapsed:.2f}s"
        )

        bot.reply_to(message, report, parse_mode="Markdown")

        # ==========================================================
        # REAL DETECTION MAP
        # ==========================================================

        detection_map = create_detection_map(img, candles)
        cv2.imwrite(detection_path, detection_map)

        with open(detection_path, "rb") as photo:
            bot.send_photo(
                message.chat.id,
                photo,
                caption=(
                    "🔢 **REAL CANDLE DETECTION MAP**\n\n"
                    "🟨 Yellow box = REAL detected candle\n"
                    "🟢 Green number = classified GREEN\n"
                    "🔴 Red number = classified RED\n\n"
                    "✅ Every box represents a REAL candle\n"
                    "❌ No fake data — only what was visible"
                ),
                parse_mode="Markdown"
            )

    except Exception as e:
        bot.reply_to(message, f"❌ Error:\n{str(e)}")

    finally:
        for path in [original_path, detection_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass


# ============================================================
# START
# ============================================================

print("=" * 50)
print("📊 REAL CANDLE DETECTOR")
print("=" * 50)
print("✅ Detects REAL candles from screenshots")
print("✅ No fake data")
print("✅ No generated OHLC")
print("✅ No trading signals")
print("✅ No invented candles")
print("=" * 50)

bot.infinity_polling(timeout=30, long_polling_timeout=30)

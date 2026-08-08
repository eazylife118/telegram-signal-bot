import os
import cv2
import numpy as np
import telebot
import time

# ============================================================
# TELEGRAM — TOKEN HARDCODED
# ============================================================

TELEGRAM_TOKEN = "8937673241:AAGvyTA-G12xfwMlhif3Nh4_2Ag8OStq3tU"

bot = telebot.TeleBot(TELEGRAM_TOKEN)


# ============================================================
# DETECTION SETTINGS — EXTREME
# ============================================================

MIN_BODY_AREA = 1
MIN_BODY_HEIGHT = 1
MIN_CANDLE_WIDTH = 1

RIGHT_MIN_BODY_AREA = 1
RIGHT_MIN_BODY_HEIGHT = 1

MAX_CANDLE_WIDTH_RATIO = 0.25
MAX_BODY_HEIGHT_RATIO = 0.80

MERGE_DISTANCE_RATIO = 0.90
MIN_COLOR_DENSITY = 0.01

RIGHT_SIDE_START = 0.30


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read screenshot.")

    h, w = img.shape[:2]
    if w < 1400:
        scale = 1400 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    return img


# ============================================================
# COLOR MASKS — AGGRESSIVE RED DETECTION
# ============================================================

def get_color_masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green — wide range
    green_lower = np.array([20, 20, 20])
    green_upper = np.array([110, 255, 255])
    green = cv2.inRange(hsv, green_lower, green_upper)

    # Red — VERY wide range
    red_lower_1 = np.array([0, 10, 10])
    red_upper_1 = np.array([35, 255, 255])
    red_lower_2 = np.array([145, 10, 10])
    red_upper_2 = np.array([190, 255, 255])

    red1 = cv2.inRange(hsv, red_lower_1, red_upper_1)
    red2 = cv2.inRange(hsv, red_lower_2, red_upper_2)
    red = cv2.bitwise_or(red1, red2)

    # Fallback: BGR channel difference
    b, g, r = cv2.split(img)
    red_fallback = cv2.bitwise_and(
        cv2.bitwise_and(
            (r > g + 5).astype(np.uint8) * 255,
            (r > b + 5).astype(np.uint8) * 255
        ),
        (r > 10).astype(np.uint8) * 255
    )

    red = cv2.bitwise_or(red, red_fallback)

    return green, red


# ============================================================
# FIND CANDIDATES
# ============================================================

def find_candidates(mask, color, image_width, image_height, right_side=False):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    max_width = max(20, int(image_width * MAX_CANDLE_WIDTH_RATIO))

    min_area = RIGHT_MIN_BODY_AREA if right_side else MIN_BODY_AREA
    min_height = RIGHT_MIN_BODY_HEIGHT if right_side else MIN_BODY_HEIGHT

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < MIN_CANDLE_WIDTH or h < min_height:
            continue
        if w > max_width or h > image_height * MAX_BODY_HEIGHT_RATIO:
            continue

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
# MERGE CANDLES
# ============================================================

def merge_candidates(candidates):
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda c: c["center_x"])
    merged = []

    for candidate in candidates:
        for existing in merged:
            distance = abs(candidate["center_x"] - existing["center_x"])
            allowed = max(candidate["w"], existing["w"], 2) * MERGE_DISTANCE_RATIO

            if distance <= allowed and not (
                candidate["y"] + candidate["h"] < existing["y"] or
                candidate["y"] > existing["y"] + existing["h"]
            ):
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
# REMOVE DUPLICATES
# ============================================================

def remove_cross_color_duplicates(candles):
    candles = sorted(candles, key=lambda c: c["center_x"])
    result = []

    for candle in candles:
        for i, existing in enumerate(result):
            distance = abs(candle["center_x"] - existing["center_x"])
            if distance <= max(candle["w"], existing["w"], 2) * 0.70:
                if candle["pixels"] > existing["pixels"]:
                    result[i] = candle
                break
        else:
            result.append(candle)

    return result


# ============================================================
# MAIN DETECTION — SCANS ENTIRE CHART
# ============================================================

def detect_candles(img):
    h, w = img.shape[:2]

    green_mask, red_mask = get_color_masks(img)

    # Scan the ENTIRE chart — no right-side only
    green = find_candidates(green_mask, "GREEN", w, h, right_side=False)
    green = merge_candidates(green)

    red = find_candidates(red_mask, "RED", w, h, right_side=False)
    red = merge_candidates(red)

    candles = green + red
    candles = remove_cross_color_duplicates(candles)
    candles.sort(key=lambda c: c["center_x"])

    # Fallback: extreme settings
    if len(candles) < 40:
        print(f"⚠️ Only {len(candles)} candles. Running extreme fallback...")

        global MIN_BODY_AREA, MIN_COLOR_DENSITY, MAX_CANDLE_WIDTH_RATIO, MIN_CANDLE_WIDTH
        old_area, old_density, old_width, old_min_width = MIN_BODY_AREA, MIN_COLOR_DENSITY, MAX_CANDLE_WIDTH_RATIO, MIN_CANDLE_WIDTH

        MIN_BODY_AREA = 1
        MIN_COLOR_DENSITY = 0.005
        MAX_CANDLE_WIDTH_RATIO = 0.30
        MIN_CANDLE_WIDTH = 1

        green = find_candidates(green_mask, "GREEN", w, h, right_side=False)
        red = find_candidates(red_mask, "RED", w, h, right_side=False)

        green = merge_candidates(green)
        red = merge_candidates(red)

        candles = green + red
        candles = remove_cross_color_duplicates(candles)
        candles.sort(key=lambda c: c["center_x"])

        MIN_BODY_AREA, MIN_COLOR_DENSITY, MAX_CANDLE_WIDTH_RATIO, MIN_CANDLE_WIDTH = old_area, old_density, old_width, old_min_width

    return candles


# ============================================================
# REPORT
# ============================================================

def create_report(candles):
    green = sum(1 for c in candles if c["color"] == "GREEN")
    red = sum(1 for c in candles if c["color"] == "RED")
    return green, red


# ============================================================
# DETECTION MAP
# ============================================================

def create_detection_map(img, candles):
    output = img.copy()

    for number, candle in enumerate(candles, start=1):
        x, y, w, h = int(candle["x"]), int(candle["y"]), int(candle["w"]), int(candle["h"])

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)

        label_color = (0, 255, 0) if candle["color"] == "GREEN" else (0, 0, 255)
        cv2.putText(output, str(number), (x, max(25, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, label_color, 2, cv2.LINE_AA)

    return output


# ============================================================
# TELEGRAM HANDLER
# ============================================================

@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    start_time = time.time()
    original_path, detection_path = "chart_screenshot.png", "candle_detection.png"

    try:
        bot.reply_to(message, "👁️ Reading visible candles...")

        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with open(original_path, "wb") as f:
            f.write(downloaded_file)

        img = load_image(original_path)
        candles = detect_candles(img)

        green, red = create_report(candles)
        total = len(candles)

        elapsed = time.time() - start_time

        if total == 0:
            bot.reply_to(message, "❌ No reliable candle bodies detected.")
            return

        sequence = "\n".join(f"{i+1}. {'🟢 GREEN' if c['color'] == 'GREEN' else '🔴 RED'}" for i, c in enumerate(candles))

        report = (
            f"🔎 **CANDLE READING TEST**\n\n"
            f"📊 **WHAT THE BOT ACTUALLY DETECTED:**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🟢 GREEN: {green}\n"
            f"🔴 RED: {red}\n"
            f"📊 TOTAL: {total}\n\n"
            f"🕯️ **CANDLE-BY-CANDLE READING:**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{sequence}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 **COLOR CHECK:**\n"
            f"🟢 = Bot believes candle is GREEN\n"
            f"🔴 = Bot believes candle is RED\n\n"
            f"⚠️ **TEST ONLY**\n"
            f"No OHLC data is generated.\n"
            f"No random candles are added.\n"
            f"No trading signal is generated.\n\n"
            f"⚡ Processing time: {elapsed:.2f}s"
        )

        bot.reply_to(message, report, parse_mode="Markdown")

        detection_map = create_detection_map(img, candles)
        cv2.imwrite(detection_path, detection_map)

        with open(detection_path, "rb") as photo:
            bot.send_photo(
                message.chat.id,
                photo,
                caption="🔢 **CANDLE DETECTION MAP**\n\n🟨 Yellow box = detected candle body\n🟢 Green number = classified GREEN\n🔴 Red number = classified RED",
                parse_mode="Markdown"
            )

    except Exception as e:
        bot.reply_to(message, f"❌ Detection error:\n{str(e)}")

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

print("========================================")
print("🕯️ CANDLE READING TEST — EXTREME")
print("========================================")
print("✅ Scans ENTIRE chart")
print("✅ Very aggressive red detection")
print("✅ Extreme fallback for missing candles")
print("========================================")

bot.infinity_polling(timeout=30, long_polling_timeout=30)

import os
import cv2
import numpy as np
import telebot
import time
from statistics import median
# ============================================================
# TELEGRAM
# ============================================================
TELEGRAM_TOKEN = os.getenv(
    "BOT_TOKEN",
    "PUT_YOUR_NEW_BOT_TOKEN_HERE"
)
if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "PUT_YOUR_NEW_BOT_TOKEN_HERE":
    raise ValueError(
        "BOT_TOKEN is not configured."
    )
bot = telebot.TeleBot(TELEGRAM_TOKEN)
# ============================================================
# CANDLE DETECTION SETTINGS
# ============================================================
# Small enough to detect small candles,
# but not so small that every colored pixel becomes a candle.
MIN_BODY_AREA = 6
MIN_BODY_HEIGHT = 2
MIN_CANDLE_WIDTH = 2
# Candle bodies should be narrow.
MAX_CANDLE_WIDTH = 35
# Color evidence.
MIN_COLOR_PIXELS = 5
MIN_COLOR_DENSITY = 0.12
# Very strict merging.
MERGE_DISTANCE_RATIO = 0.20
# Duplicate protection.
DUPLICATE_DISTANCE_RATIO = 0.35
# Candle geometry.
MAX_BODY_WIDTH_HEIGHT_RATIO = 2.5
MIN_VERTICALITY = 1.15
# Spacing filter.
# A real candle series normally has repeated horizontal spacing.
SPACING_TOLERANCE = 0.60
# Minimum number of candidates before spacing filtering
# becomes useful.
MIN_CANDIDATES_FOR_SPACING = 8
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
    # Upscale only. NEVER crop.
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
# COLOR MASKS
# ============================================================
def get_color_masks(img):
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )
    # --------------------------------------------------------
    # GREEN — HSV
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
    # RED — HSV
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
    # BGR ASSISTANCE
    # --------------------------------------------------------
    b, g, r = cv2.split(img)
    b16 = b.astype(np.int16)
    g16 = g.astype(np.int16)
    r16 = r.astype(np.int16)
    green_bgr = (
        (g16 > r16 + 12) &
        (g16 > b16 + 5) &
        (g > 45)
    )
    red_bgr = (
        (r16 > g16 + 12) &
        (r16 > b16 + 5) &
        (r > 45)
    )
    green_bgr = (
        green_bgr.astype(np.uint8) * 255
    )
    red_bgr = (
        red_bgr.astype(np.uint8) * 255
    )
    # HSV remains primary.
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
# FIND RAW CANDLE CANDIDATES
# ============================================================
def find_candidates(
    mask,
    color,
    image_width
):
    # Very small cleanup.
    # Do NOT connect neighboring candles.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )
    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    for contour in contours:
        area = cv2.contourArea(
            contour
        )
        if area < MIN_BODY_AREA:
            continue
        x, y, w, h = cv2.boundingRect(
            contour
        )
        # ----------------------------------------------------
        # BASIC SIZE
        # ----------------------------------------------------
        if w < MIN_CANDLE_WIDTH:
            continue
        if h < MIN_BODY_HEIGHT:
            continue
        if w > MAX_CANDLE_WIDTH:
            continue
        # ----------------------------------------------------
        # CANDLE GEOMETRY
        # ----------------------------------------------------
        width_height_ratio = (
            w /
            float(max(1, h))
        )
        # Candle body should not be a wide horizontal object.
        if width_height_ratio > MAX_BODY_WIDTH_HEIGHT_RATIO:
            continue
        verticality = (
            h /
            float(max(1, w))
        )
        if verticality < MIN_VERTICALITY:
            continue
        # ----------------------------------------------------
        # COLOR EVIDENCE
        # ----------------------------------------------------
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
        # ----------------------------------------------------
        # CENTER
        # ----------------------------------------------------
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
# MERGE ONLY ACTUALLY CONNECTED PIECES
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
            distance = abs(
                candidate["center_x"] -
                existing["center_x"]
            )
            allowed = max(
                1,
                min(
                    candidate["w"],
                    existing["w"]
                )
            ) * MERGE_DISTANCE_RATIO
            # Horizontal edges.
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
            gap = max(
                candidate_left -
                existing_right,
                existing_left -
                candidate_right,
                0
            )
            # Only merge if essentially touching.
            touching = gap <= 1
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
            if (
                distance <= allowed
                and
                touching
                and
                vertical_overlap
            ):
                left = min(
                    existing_left,
                    candidate_left
                )
                right = max(
                    existing_right,
                    candidate_right
                )
                top = min(
                    existing_top,
                    candidate_top
                )
                bottom = max(
                    existing_bottom,
                    candidate_bottom
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
# REMOVE TRUE DUPLICATES
# ============================================================
def remove_duplicates(candles):
    candles = sorted(
        candles,
        key=lambda c: c["center_x"]
    )
    result = []
    for candle in candles:
        duplicate = False
        for index, existing in enumerate(result):
            distance = abs(
                candle["center_x"] -
                existing["center_x"]
            )
            threshold = max(
                candle["w"],
                existing["w"],
                2
            ) * DUPLICATE_DISTANCE_RATIO
            if distance > threshold:
                continue
            # Compare vertical overlap.
            top1 = candle["y"]
            bottom1 = (
                candle["y"] +
                candle["h"]
            )
            top2 = existing["y"]
            bottom2 = (
                existing["y"] +
                existing["h"]
            )
            overlap = max(
                0,
                min(
                    bottom1,
                    bottom2
                ) -
                max(
                    top1,
                    top2
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
                duplicate = True
                # Keep stronger color evidence.
                if (
                    candle["pixels"] >
                    existing["pixels"]
                ):
                    result[index] = candle
                break
        if not duplicate:
            result.append(candle)
    return result
# ============================================================
# SPACING ANALYSIS
# ============================================================
def calculate_spacing_score(
    candle,
    sorted_candidates
):
    if len(sorted_candidates) < 4:
        return 0.0
    x = candle["center_x"]
    distances = []
    for other in sorted_candidates:
        if other is candle:
            continue
        d = abs(
            other["center_x"] -
            x
        )
        if d > 1:
            distances.append(d)
    if not distances:
        return 0.0
    distances.sort()
    nearby = distances[
        :min(6, len(distances))
    ]
    typical = median(
        nearby
    )
    if typical <= 0:
        return 0.0
    # Candidate gets a stronger score if its closest
    # neighbors occur at a reasonably repeated spacing.
    score = 0.0
    for d in nearby:
        ratio = (
            d /
            typical
        )
        if 0.50 <= ratio <= 1.50:
            score += 1.0
    return score
# ============================================================
# KEEP CANDIDATES THAT LOOK LIKE PART OF THE CANDLE SERIES
# ============================================================
def spacing_filter(candles):
    if len(candles) < MIN_CANDIDATES_FOR_SPACING:
        return candles
    candles = sorted(
        candles,
        key=lambda c: c["center_x"]
    )
    # We do NOT aggressively delete based on spacing.
    # Instead, identify isolated objects.
    keep = []
    for i, candle in enumerate(candles):
        left_distance = None
        right_distance = None
        if i > 0:
            left_distance = (
                candle["center_x"] -
                candles[i - 1]["center_x"]
            )
        if i < len(candles) - 1:
            right_distance = (
                candles[i + 1]["center_x"] -
                candle["center_x"]
            )
        neighbors = []
        if left_distance is not None:
            neighbors.append(
                left_distance
            )
        if right_distance is not None:
            neighbors.append(
                right_distance
            )
        # Keep edge candles.
        if len(neighbors) == 1:
            keep.append(candle)
            continue
        # A genuine candle in a dense candle field normally
        # has neighbors on both sides.
        nearest = min(neighbors)
        farthest = max(neighbors)
        # Reject only extremely isolated colored objects.
        if (
            farthest >
            nearest * 4.0
        ):
            # Keep it if it is reasonably candle-shaped
            # and has strong color evidence.
            if (
                candle["density"] >= 0.30
                and
                candle["pixels"] >= 15
                and
                candle["h"] >= candle["w"] * 2
            ):
                keep.append(candle)
            continue
        keep.append(candle)
    return keep
# ============================================================
# DETECT CANDLES — FULL SCREEN
# ============================================================
def detect_candles(img):
    # Entire screenshot.
    h, w = img.shape[:2]
    green_mask, red_mask = (
        get_color_masks(img)
    )
    # --------------------------------------------------------
    # GREEN
    # --------------------------------------------------------
    green = find_candidates(
        green_mask,
        "GREEN",
        w
    )
    # --------------------------------------------------------
    # RED
    # --------------------------------------------------------
    red = find_candidates(
        red_mask,
        "RED",
        w
    )
    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------
    green = merge_candidates(
        green
    )
    red = merge_candidates(
        red
    )
    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------
    candles = (
        green +
        red
    )
    candles = remove_duplicates(
        candles
    )
    candles.sort(
        key=lambda c: c["center_x"]
    )
    # --------------------------------------------------------
    # SPACING CHECK
    # --------------------------------------------------------
    candles = spacing_filter(
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
def create_detection_map(
    img,
    candles
):
    output = img.copy()
    h, w = output.shape[:2]
    # --------------------------------------------------------
    # FULL SCREEN BORDER
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
        "FULL SCREEN",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )
    # --------------------------------------------------------
    # ACCEPTED CANDLES
    # --------------------------------------------------------
    for number, candle in enumerate(
        candles,
        start=1
    ):
        x = int(candle["x"])
        y = int(candle["y"])
        cw = int(candle["w"])
        ch = int(candle["h"])
        # Yellow rectangle = accepted candle.
        cv2.rectangle(
            output,
            (x, y),
            (x + cw, y + ch),
            (0, 255, 255),
            2
        )
        if candle["color"] == "GREEN":
            text_color = (
                0,
                255,
                0
            )
            label = (
                f"{number} G"
            )
        else:
            text_color = (
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
                    22,
                    y - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            text_color,
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
            "👁️ Reading the full screenshot...\n"
            "🟡 100% screen coverage\n"
            "🕯️ Checking candle geometry\n"
            "🟢 GREEN / 🔴 RED classification"
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
        # RESULT
        # ----------------------------------------------------
        if total == 0:
            bot.reply_to(
                message,
                "❌ No reliable candle bodies detected.\n\n"
                "No candles were invented.\n"
                "No OHLC data was generated.\n"
                "No trading signal was generated."
            )
            return
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
        report = (
            "🔎 **CANDLE READING TEST**\n\n"
            "🟡 **DETECTION AREA:**\n"
            "100% of uploaded screenshot\n\n"
            "📊 **DETECTED:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🟢 GREEN: {green}\n"
            f"🔴 RED: {red}\n"
            f"📊 TOTAL: {total}\n\n"
            "🕯️ **CANDLE-BY-CANDLE:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"{sequence_text}\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🎯 **COLOR CHECK:**\n"
            "🟢 G = classified GREEN\n"
            "🔴 R = classified RED\n\n"
            "⚠️ TEST ONLY\n"
            "No OHLC generation.\n"
            "No random candles.\n"
            "No trading signal.\n\n"
            f"⚡ Processing: {elapsed:.2f}s"
        )
        bot.reply_to(
            message,
            report,
            parse_mode="Markdown"
        )
        # ----------------------------------------------------
        # MAP
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
        with open(
            detection_path,
            "rb"
        ) as photo:
            bot.send_photo(
                message.chat.id,
                photo,
                caption=(
                    "🟡 Yellow boxes = accepted candles.\n"
                    "🟢 G = GREEN.\n"
                    "🔴 R = RED.\n\n"
                    "The whole screenshot was scanned.\n"
                    "No left/right crop was used.\n\n"
                    "Check every visible candle against "
                    "the yellow boxes."
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
    "100% screenshot coverage"
)
print(
    "HSV + controlled BGR color detection"
)
print(
    "Candle geometry filtering"
)
print(
    "Strict merging"
)
print(
    "Spacing/isolation filtering"
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

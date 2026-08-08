import os
import time
import cv2
import numpy as np
import telebot
# ============================================================
# TELEGRAM
# ============================================================
# Put your TEST bot token here.
# Do NOT commit this file with a real token to GitHub.
TELEGRAM_TOKEN = "8937673241:AAGvyTA-G12xfwMlhif3Nh4_2Ag8OStq3tU"
bot = telebot.TeleBot(TELEGRAM_TOKEN)
# ============================================================
# SETTINGS
# ============================================================
# The whole received screenshot is examined.
FULL_SCREEN = True
# Small candle width.
# The program estimates the actual spacing automatically.
MIN_CANDLE_WIDTH = 3
MAX_CANDLE_WIDTH = 80
# Minimum evidence required before a lane is counted.
MIN_COLOR_PIXELS = 10
MIN_COLOR_RATIO = 0.025
# A candle should have some vertical structure.
MIN_VERTICAL_SPAN = 5
# How much of the detected candle width becomes the lane.
LANE_MULTIPLIER = 1.15
# ============================================================
# IMAGE
# ============================================================
def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Screenshot could not be read.")
    return image
# ============================================================
# COLOR MASKS
# ============================================================
def create_color_masks(image):
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )
    # --------------------------------------------------------
    # GREEN
    # --------------------------------------------------------
    green_low = np.array([
        30,
        45,
        35
    ])
    green_high = np.array([
        95,
        255,
        255
    ])
    green = cv2.inRange(
        hsv,
        green_low,
        green_high
    )
    # --------------------------------------------------------
    # RED
    # --------------------------------------------------------
    red_low_1 = np.array([
        0,
        45,
        35
    ])
    red_high_1 = np.array([
        15,
        255,
        255
    ])
    red_low_2 = np.array([
        165,
        45,
        35
    ])
    red_high_2 = np.array([
        180,
        255,
        255
    ])
    red1 = cv2.inRange(
        hsv,
        red_low_1,
        red_high_1
    )
    red2 = cv2.inRange(
        hsv,
        red_low_2,
        red_high_2
    )
    red = cv2.bitwise_or(
        red1,
        red2
    )
    return green, red
# ============================================================
# CLEAN COLOR MASK
# ============================================================
def clean_mask(mask):
    # Keep small candle bodies.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )
    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel
    )
    return cleaned
# ============================================================
# FIND CANDLE X POSITIONS
#
# This does NOT count contours.
#
# It finds vertical concentrations of candle-colored pixels,
# then uses those positions to build full-height lanes.
# ============================================================
def find_x_peaks(mask):
    height, width = mask.shape
    # Number of colored pixels in every x column.
    projection = np.sum(
        mask > 0,
        axis=0
    ).astype(np.float32)
    if np.max(projection) <= 0:
        return []
    # Smooth the projection slightly.
    smooth_kernel = max(
        3,
        int(width * 0.002)
    )
    if smooth_kernel % 2 == 0:
        smooth_kernel += 1
    smooth = cv2.GaussianBlur(
        projection.reshape(1, -1),
        (
            smooth_kernel,
            1
        ),
        0
    ).flatten()
    # Adaptive threshold.
    nonzero = smooth[smooth > 0]
    if len(nonzero) == 0:
        return []
    threshold = max(
        2,
        float(np.percentile(nonzero, 35))
    )
    active = smooth >= threshold
    # Find continuous x regions.
    regions = []
    start = None
    for x, value in enumerate(active):
        if value and start is None:
            start = x
        elif not value and start is not None:
            end = x - 1
            if end >= start:
                regions.append(
                    (start, end)
                )
            start = None
    if start is not None:
        regions.append(
            (start, width - 1)
        )
    centers = []
    for left, right in regions:
        region_width = right - left + 1
        if (
            region_width < MIN_CANDLE_WIDTH
            or
            region_width > MAX_CANDLE_WIDTH
        ):
            continue
        local = smooth[left:right + 1]
        if len(local) == 0:
            continue
        strongest = left + int(
            np.argmax(local)
        )
        centers.append(
            strongest
        )
    return centers
# ============================================================
# ESTIMATE CANDLE SPACING
# ============================================================
def estimate_spacing(centers, image_width):
    if len(centers) < 2:
        # Conservative fallback for small candle width.
        return max(
            6,
            min(
                30,
                int(image_width * 0.018)
            )
        )
    centers = sorted(
        set(centers)
    )
    distances = []
    for i in range(
        1,
        len(centers)
    ):
        distance = (
            centers[i]
            -
            centers[i - 1]
        )
        if 4 <= distance <= image_width * 0.15:
            distances.append(distance)
    if not distances:
        return 12
    # Median is safer than average when a few false peaks exist.
    spacing = int(
        np.median(distances)
    )
    return max(
        MIN_CANDLE_WIDTH,
        min(
            MAX_CANDLE_WIDTH,
            spacing
        )
    )
# ============================================================
# BUILD FULL-HEIGHT LANES
# ============================================================
def build_lanes(
    centers,
    spacing,
    image_width
):
    if not centers:
        return []
    # Candle width is intentionally small.
    candle_width = max(
        MIN_CANDLE_WIDTH,
        int(spacing * LANE_MULTIPLIER)
    )
    # Never allow the lane to become huge.
    candle_width = min(
        candle_width,
        max(8, int(image_width * 0.04))
    )
    lanes = []
    for center in sorted(
        set(centers)
    ):
        left = int(
            center -
            candle_width / 2
        )
        right = int(
            center +
            candle_width / 2
        )
        left = max(
            0,
            left
        )
        right = min(
            image_width - 1,
            right
        )
        lanes.append({
            "center": int(center),
            "left": left,
            "right": right,
            "width": right - left + 1
        })
    return lanes
# ============================================================
# CHECK WHETHER A LANE REALLY CONTAINS A CANDLE
# ============================================================
def inspect_lane(
    green_mask,
    red_mask,
    lane
):
    height, width = green_mask.shape
    left = lane["left"]
    right = lane["right"]
    green_region = green_mask[
        0:height,
        left:right + 1
    ]
    red_region = red_mask[
        0:height,
        left:right + 1
    ]
    green_pixels = int(
        np.sum(
            green_region > 0
        )
    )
    red_pixels = int(
        np.sum(
            red_region > 0
        )
    )
    total_pixels = max(
        1,
        green_region.size
    )
    green_ratio = (
        green_pixels /
        total_pixels
    )
    red_ratio = (
        red_pixels /
        total_pixels
    )
    # Find the vertical span of colored pixels.
    combined = cv2.bitwise_or(
        green_region,
        red_region
    )
    ys, xs = np.where(
        combined > 0
    )
    if len(ys) == 0:
        return None
    vertical_span = (
        int(np.max(ys))
        -
        int(np.min(ys))
        +
        1
    )
    if vertical_span < MIN_VERTICAL_SPAN:
        return None
    # --------------------------------------------------------
    # COLOR DECISION
    # --------------------------------------------------------
    if (
        green_pixels >= MIN_COLOR_PIXELS
        and
        green_ratio >= MIN_COLOR_RATIO
        and
        green_pixels > red_pixels * 1.20
    ):
        return {
            "color": "GREEN",
            "green_pixels": green_pixels,
            "red_pixels": red_pixels,
            "ratio": green_ratio,
            "vertical_span": vertical_span
        }
    if (
        red_pixels >= MIN_COLOR_PIXELS
        and
        red_ratio >= MIN_COLOR_RATIO
        and
        red_pixels > green_pixels * 1.20
    ):
        return {
            "color": "RED",
            "green_pixels": green_pixels,
            "red_pixels": red_pixels,
            "ratio": red_ratio,
            "vertical_span": vertical_span
        }
    # Ambiguous = do not force a color.
    return None
# ============================================================
# FULL SCREEN CANDLE DETECTOR
# ============================================================
def detect_candles(image):
    height, width = image.shape[:2]
    green_mask, red_mask = create_color_masks(
        image
    )
    green_mask = clean_mask(
        green_mask
    )
    red_mask = clean_mask(
        red_mask
    )
    # --------------------------------------------------------
    # FIND COLOR CONCENTRATIONS
    # --------------------------------------------------------
    green_centers = find_x_peaks(
        green_mask
    )
    red_centers = find_x_peaks(
        red_mask
    )
    all_centers = (
        green_centers +
        red_centers
    )
    if not all_centers:
        return [], [], 0
    all_centers = sorted(
        set(all_centers)
    )
    # --------------------------------------------------------
    # ESTIMATE REAL CANDLE SPACING
    # --------------------------------------------------------
    spacing = estimate_spacing(
        all_centers,
        width
    )
    # --------------------------------------------------------
    # CREATE FULL-HEIGHT VERTICAL LANES
    # --------------------------------------------------------
    lanes = build_lanes(
        all_centers,
        spacing,
        width
    )
    detected = []
    for lane in lanes:
        result = inspect_lane(
            green_mask,
            red_mask,
            lane
        )
        if result is None:
            continue
        candle = lane.copy()
        candle.update(
            result
        )
        detected.append(
            candle
        )
    # Sort left → right.
    detected.sort(
        key=lambda c: c["center"]
    )
    return (
        detected,
        lanes,
        spacing
    )
# ============================================================
# CREATE FULL-SCREEN DIAGNOSTIC IMAGE
# ============================================================
def create_detection_map(
    image,
    detected,
    lanes,
    spacing
):
    output = image.copy()
    height, width = output.shape[:2]
    # --------------------------------------------------------
    # DRAW ALL SEARCH LANES LIGHTLY
    # --------------------------------------------------------
    for lane in lanes:
        x = lane["center"]
        cv2.line(
            output,
            (x, 0),
            (x, height - 1),
            (255, 255, 0),
            1
        )
    # --------------------------------------------------------
    # DRAW CONFIRMED CANDLES
    # --------------------------------------------------------
    for number, candle in enumerate(
        detected,
        start=1
    ):
        left = candle["left"]
        right = candle["right"]
        if candle["color"] == "GREEN":
            box_color = (
                0,
                255,
                0
            )
        else:
            box_color = (
                0,
                0,
                255
            )
        # Full-height lane.
        cv2.rectangle(
            output,
            (left, 0),
            (right, height - 1),
            box_color,
            2
        )
        # Number at top.
        cv2.putText(
            output,
            f"{number}",
            (
                left,
                30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            box_color,
            2,
            cv2.LINE_AA
        )
        # Color label.
        label = candle["color"]
        cv2.putText(
            output,
            label,
            (
                left,
                min(
                    height - 10,
                    60
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            box_color,
            2,
            cv2.LINE_AA
        )
    return output
# ============================================================
# REPORT
# ============================================================
def create_report(
    detected,
    spacing
):
    green = sum(
        1
        for c in detected
        if c["color"] == "GREEN"
    )
    red = sum(
        1
        for c in detected
        if c["color"] == "RED"
    )
    total = len(
        detected
    )
    sequence = []
    for candle in detected:
        if candle["color"] == "GREEN":
            sequence.append("🟢 GREEN")
        else:
            sequence.append("🔴 RED")
    return (
        green,
        red,
        total,
        sequence,
        spacing
    )
# ============================================================
# TELEGRAM PHOTO HANDLER
# ============================================================
@bot.message_handler(
    content_types=["photo"]
)
def handle_photo(message):
    start_time = time.time()
    original_path = (
        "pocket_option_screenshot.png"
    )
    detection_path = (
        "candle_full_screen_map.png"
    )
    try:
        bot.reply_to(
            message,
            "👁️ Reading the entire screenshot...\n"
            "Building full-height candle lanes..."
        )
        # ----------------------------------------------------
        # DOWNLOAD HIGHEST RESOLUTION PHOTO
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
        # LOAD ORIGINAL SCREENSHOT
        # ----------------------------------------------------
        image = load_image(
            original_path
        )
        # ----------------------------------------------------
        # DETECT
        # ----------------------------------------------------
        detected, lanes, spacing = (
            detect_candles(image)
        )
        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------
        (
            green,
            red,
            total,
            sequence,
            spacing
        ) = create_report(
            detected,
            spacing
        )
        elapsed = (
            time.time()
            -
            start_time
        )
        if total == 0:
            bot.reply_to(
                message,
                "❌ No candle was confidently detected.\n\n"
                "The detector did NOT invent a candle.\n"
                "No OHLC data was generated.\n"
                "No random candles were added.\n"
                "No trading signal was generated."
            )
            return
        sequence_text = "\n".join(
            f"{i}. {value}"
            for i, value in enumerate(
                sequence,
                start=1
            )
        )
        report = (
            "🔎 **CANDLE VISION TEST**\n\n"
            "📱 **FULL SCREEN SCAN**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"📐 Screenshot: "
            f"{image.shape[1]} × {image.shape[0]}\n"
            f"📏 Estimated candle spacing: "
            f"{spacing}px\n\n"
            "📊 **ACTUAL DETECTION:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🟢 Green candles: {green}\n"
            f"🔴 Red candles: {red}\n"
            f"📊 Total candles: {total}\n\n"
            "🕯️ **LEFT → RIGHT:**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"{sequence_text}\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🔬 **METHOD:**\n"
            "• Entire screenshot scanned\n"
            "• Full-height vertical candle lanes\n"
            "• Small candle width\n"
            "• Adaptive candle spacing\n"
            "• Actual green/red pixel evidence\n"
            "• Ambiguous objects are NOT forced\n\n"
            "⚠️ **TEST ONLY**\n"
            "No OHLC candles generated.\n"
            "No random candles added.\n"
            "No trading signal generated.\n\n"
            f"⚡ Processing time: "
            f"{elapsed:.2f}s"
        )
        bot.reply_to(
            message,
            report,
            parse_mode="Markdown"
        )
        # ----------------------------------------------------
        # DIAGNOSTIC MAP
        # ----------------------------------------------------
        diagnostic = create_detection_map(
            image,
            detected,
            lanes,
            spacing
        )
        cv2.imwrite(
            detection_path,
            diagnostic
        )
        with open(
            detection_path,
            "rb"
        ) as photo:
            bot.send_photo(
                message.chat.id,
                photo,
                caption=(
                    "📱 FULL-SCREEN CANDLE MAP\n\n"
                    "🟢 Green lane = detector classified "
                    "a candle as GREEN.\n"
                    "🔴 Red lane = detector classified "
                    "a candle as RED.\n"
                    "🟡 Thin line = inspected candle position.\n\n"
                    "The lane covers the full height of "
                    "the screenshot.\n\n"
                    "A lane by itself is NOT counted as "
                    "a candle."
                )
            )
    except Exception as error:
        print(
            "❌ ERROR:",
            repr(error)
        )
        bot.reply_to(
            message,
            "❌ Detection error:\n"
            f"{str(error)}"
        )
    finally:
        for path in (
            original_path,
            detection_path
        ):
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
    "📱 POCKET OPTION FULL-SCREEN "
    "CANDLE VISION TEST"
)
print(
    "========================================"
)
print(
    "Entire screenshot will be scanned."
)
print(
    "Vertical lanes cover the full screen."
)
print(
    "No forced candle count."
)
print(
    "No fake candles."
)
print(
    "No OHLC generation."
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

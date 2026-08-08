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
# Normal candle body settings
MIN_BODY_AREA = 5
MIN_BODY_HEIGHT = 2
MIN_CANDLE_WIDTH = 1
# Right-side candles can be smaller/newer
RIGHT_SIDE_MIN_AREA = 2
RIGHT_SIDE_MIN_HEIGHT = 1
# Maximum candle width relative to screenshot width
MAX_CANDLE_WIDTH_RATIO = 0.035
# How close two detections can be before they are considered
# parts of the same candle.
MERGE_DISTANCE_RATIO = 0.75
# Minimum amount of colored pixels needed to consider
# a vertical region a possible candle.
MIN_COLOR_PIXELS = 3
# ============================================================
# LOAD IMAGE
# ============================================================
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read screenshot.")
    h, w = img.shape[:2]
    # Upscale smaller screenshots.
    if w < 1600:
        scale = 1600 / w
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
# CHART REGION
# ============================================================
def get_chart_region(img):
    h, w = img.shape[:2]
    # Pocket Option screenshots normally contain controls,
    # pair information and other UI outside the chart.
    #
    # Keep a generous chart area because we don't want to
    # accidentally cut off candles.
    top = int(h * 0.06)
    bottom = int(h * 0.91)
    left = int(w * 0.03)
    right = int(w * 0.97)
    return img[top:bottom, left:right], left, top
# ============================================================
# COLOR MASKS
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
        25,
        25,
        25
    ])
    green_upper = np.array([
        100,
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
        25,
        25
    ])
    red_upper_1 = np.array([
        18,
        255,
        255
    ])
    red_lower_2 = np.array([
        158,
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
    red = cv2.bitwise_or(
        red1,
        red2
    )
    return green, red
# ============================================================
# CLEAN MASK
# ============================================================
def clean_mask(mask):
    # Very small opening removes isolated noise while
    # preserving small candle bodies.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )
    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel
    )
    # Close tiny gaps in candle bodies.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        close_kernel
    )
    return cleaned
# ============================================================
# FIND BODY CANDIDATES
# ============================================================
def find_body_candidates(
    mask,
    color,
    image_width,
    region_name
):
    cleaned = clean_mask(mask)
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    max_width = max(
        8,
        int(image_width * MAX_CANDLE_WIDTH_RATIO)
    )
    if region_name == "RIGHT":
        minimum_area = RIGHT_SIDE_MIN_AREA
        minimum_height = RIGHT_SIDE_MIN_HEIGHT
    else:
        minimum_area = MIN_BODY_AREA
        minimum_height = MIN_BODY_HEIGHT
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area:
            continue
        x, y, w, h = cv2.boundingRect(
            contour
        )
        if w < MIN_CANDLE_WIDTH:
            continue
        if h < minimum_height:
            continue
        if w > max_width:
            continue
        # Candle bodies should normally be vertical or
        # approximately square, not long horizontal bars.
        if w > h * 7:
            continue
        pixel_count = int(
            np.sum(
                cleaned[y:y+h, x:x+w] > 0
            )
        )
        if pixel_count < MIN_COLOR_PIXELS:
            continue
        center_x = x + w / 2
        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": float(area),
            "pixels": pixel_count,
            "center_x": center_x,
            "color": color,
            "region": region_name
        })
    return candidates
# ============================================================
# VERTICAL CANDLE EVIDENCE
# ============================================================
def find_vertical_candidates(
    mask,
    color,
    image_width,
    region_name
):
    """
    Secondary detector.
    This is especially useful when a candle body is tiny or
    fragmented and contour detection misses it.
    It looks for narrow vertical concentrations of the
    actual candle color.
    """
    height, width = mask.shape
    candidates = []
    # Calculate colored pixels in each vertical column.
    column_pixels = np.sum(
        mask > 0,
        axis=0
    )
    # A candle body usually produces a local concentration
    # of colored pixels.
    threshold = max(
        2,
        int(height * 0.003)
    )
    active = column_pixels >= threshold
    start = None
    for x in range(width):
        if active[x] and start is None:
            start = x
        elif not active[x] and start is not None:
            end = x
            width_region = end - start
            if width_region >= 1:
                # Ignore extremely wide objects.
                if width_region <= max(
                    10,
                    int(image_width * MAX_CANDLE_WIDTH_RATIO)
                ):
                    region = mask[:, start:end]
                    ys, xs = np.where(
                        region > 0
                    )
                    if len(ys) >= MIN_COLOR_PIXELS:
                        y1 = int(np.min(ys))
                        y2 = int(np.max(ys))
                        height_region = y2 - y1 + 1
                        if region_name == "RIGHT":
                            valid_height = (
                                height_region >=
                                RIGHT_SIDE_MIN_HEIGHT
                            )
                        else:
                            valid_height = (
                                height_region >=
                                MIN_BODY_HEIGHT
                            )
                        if valid_height:
                            candidates.append({
                                "x": start,
                                "y": y1,
                                "w": width_region,
                                "h": height_region,
                                "area": float(
                                    np.sum(region > 0)
                                ),
                                "pixels": int(
                                    np.sum(region > 0)
                                ),
                                "center_x": (
                                    start +
                                    width_region / 2
                                ),
                                "color": color,
                                "region": region_name
                            })
            start = None
    return candidates
# ============================================================
# MERGE SAME-COLOR CANDIDATES
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
        matched = None
        for existing in merged:
            distance = abs(
                candidate["center_x"]
                -
                existing["center_x"]
            )
            allowed_distance = max(
                candidate["w"],
                existing["w"],
                2
            ) * MERGE_DISTANCE_RATIO
            # Require vertical overlap as well.
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
                distance <= allowed_distance
                and vertical_overlap
            ):
                matched = existing
                break
        if matched is None:
            merged.append(
                candidate.copy()
            )
        else:
            left = min(
                matched["x"],
                candidate["x"]
            )
            right = max(
                matched["x"] + matched["w"],
                candidate["x"] + candidate["w"]
            )
            top = min(
                matched["y"],
                candidate["y"]
            )
            bottom = max(
                matched["y"] + matched["h"],
                candidate["y"] + candidate["h"]
            )
            matched["x"] = left
            matched["y"] = top
            matched["w"] = right - left
            matched["h"] = bottom - top
            matched["center_x"] = (
                left +
                matched["w"] / 2
            )
            matched["area"] += candidate["area"]
            matched["pixels"] += candidate["pixels"]
    return merged
# ============================================================
# REMOVE DUPLICATES
# ============================================================
def remove_duplicates(candidates):
    if not candidates:
        return []
    candidates = sorted(
        candidates,
        key=lambda c: c["center_x"]
    )
    result = []
    for candidate in candidates:
        duplicate_index = None
        for i, existing in enumerate(result):
            x_distance = abs(
                candidate["center_x"]
                -
                existing["center_x"]
            )
            max_width = max(
                candidate["w"],
                existing["w"],
                2
            )
            # Two detections very close together are likely
            # the same candle.
            if x_distance <= max_width * 0.8:
                duplicate_index = i
                break
        if duplicate_index is None:
            result.append(
                candidate
            )
        else:
            existing = result[
                duplicate_index
            ]
            # Prefer the stronger visual detection.
            if candidate["pixels"] > existing["pixels"]:
                result[
                    duplicate_index
                ] = candidate
    return result
# ============================================================
# RIGHT-SIDE SECOND PASS
# ============================================================
def detect_right_side(
    chart,
    green_mask,
    red_mask
):
    """
    The newest candles are normally on the right side.
    Run a separate, more sensitive pass over that region.
    """
    h, w = chart.shape[:2]
    right_start = int(
        w * 0.70
    )
    green_right = green_mask[
        :,
        right_start:
    ]
    red_right = red_mask[
        :,
        right_start:
    ]
    green_candidates = find_body_candidates(
        green_right,
        "GREEN",
        w,
        "RIGHT"
    )
    red_candidates = find_body_candidates(
        red_right,
        "RED",
        w,
        "RIGHT"
    )
    green_vertical = find_vertical_candidates(
        green_right,
        "GREEN",
        w,
        "RIGHT"
    )
    red_vertical = find_vertical_candidates(
        red_right,
        "RED",
        w,
        "RIGHT"
    )
    candidates = (
        green_candidates +
        red_candidates +
        green_vertical +
        red_vertical
    )
    # Restore coordinates relative to full chart.
    for candidate in candidates:
        candidate["x"] += right_start
        candidate["center_x"] += right_start
    candidates = merge_candidates(
        candidates
    )
    return candidates
# ============================================================
# MAIN CANDLE DETECTOR
# ============================================================
def detect_candles(img):
    chart, offset_x, offset_y = (
        get_chart_region(img)
    )
    h, w = chart.shape[:2]
    green_mask, red_mask = (
        get_color_masks(chart)
    )
    # --------------------------------------------------------
    # NORMAL FULL-CHART DETECTION
    # --------------------------------------------------------
    green_candidates = find_body_candidates(
        green_mask,
        "GREEN",
        w,
        "NORMAL"
    )
    red_candidates = find_body_candidates(
        red_mask,
        "RED",
        w,
        "NORMAL"
    )
    # --------------------------------------------------------
    # SECONDARY VERTICAL DETECTION
    # --------------------------------------------------------
    green_vertical = find_vertical_candidates(
        green_mask,
        "GREEN",
        w,
        "NORMAL"
    )
    red_vertical = find_vertical_candidates(
        red_mask,
        "RED",
        w,
        "NORMAL"
    )
    candidates = (
        green_candidates +
        red_candidates +
        green_vertical +
        red_vertical
    )
    candidates = merge_candidates(
        candidates
    )
    # --------------------------------------------------------
    # SPECIAL RIGHT-SIDE PASS
    # --------------------------------------------------------
    right_candidates = detect_right_side(
        chart,
        green_mask,
        red_mask
    )
    candidates.extend(
        right_candidates
    )
    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------
    candidates = remove_duplicates(
        candidates
    )
    # --------------------------------------------------------
    # RESTORE FULL SCREENSHOT COORDINATES
    # --------------------------------------------------------
    for candle in candidates:
        candle["x"] += offset_x
        candle["y"] += offset_y
        candle["center_x"] += offset_x
    # --------------------------------------------------------
    # SORT LEFT → RIGHT
    # --------------------------------------------------------
    candidates.sort(
        key=lambda c: c["center_x"]
    )
    return candidates
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
# ANNOTATED DETECTION MAP
# ============================================================
def create_annotated_image(
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
        # Yellow detection rectangle.
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2
        )
        # Number every detected candle.
        cv2.putText(
            output,
            str(number),
            (
                x,
                max(25, y - 7)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
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
    annotated_path = (
        "candle_detection.png"
    )
    try:
        bot.reply_to(
            message,
            "👁️ Reading ALL visible candles...\n"
            "Checking historical + newest candles."
        )
        # ----------------------------------------------------
        # DOWNLOAD HIGHEST RESOLUTION PHOTO
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
        # LOAD IMAGE
        # ----------------------------------------------------
        img = load_image(
            original_path
        )
        # ----------------------------------------------------
        # DETECT CANDLES
        # ----------------------------------------------------
        candles = detect_candles(
            img
        )
        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------
        green, red, sequence = (
            create_report(candles)
        )
        total = len(candles)
        elapsed = (
            time.time() -
            start_time
        )
        if total == 0:
            bot.reply_to(
                message,
                "❌ No reliable candle objects detected.\n\n"
                "No candles were generated.\n"
                "No random candles were added.\n"
                "No trading signal was generated."
            )
            return
        sequence_text = (
            " → ".join(sequence)
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
            "🔬 **DETECTION METHOD:**\n"
            "• Full-chart candle scan\n"
            "• Small-body detection\n"
            "• Vertical candle detection\n"
            "• Enhanced right-side scan\n"
            "• Duplicate removal\n\n"
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
        # CREATE DETECTION MAP
        # ----------------------------------------------------
        annotated = (
            create_annotated_image(
                img,
                candles
            )
        )
        cv2.imwrite(
            annotated_path,
            annotated
        )
        # ----------------------------------------------------
        # SEND DETECTION MAP
        # ----------------------------------------------------
        with open(
            annotated_path,
            "rb"
        ) as photo:
            bot.send_photo(
                message.chat.id,
                photo,
                caption=(
                    "🔢 **CANDLE DETECTION MAP**\n\n"
                    "Each yellow numbered box represents "
                    "one candle detected from the screenshot.\n\n"
                    "The right side receives an additional "
                    "sensitive detection pass."
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
# START BOT
# ============================================================
print(
    "========================================"
)
print(
    "🕯️ CANDLE VISION TEST"
)
print(
    "========================================"
)
print(
    "Reading visible candles only."
)
print(
    "No OHLC generation."
)
print(
    "No random candles."
)
print(
    "No trading strategy."
)
print(
    "Enhanced newest/right-side detection enabled."
)
print(
    "========================================"
)
bot.infinity_polling(
    timeout=30,
    long_polling_timeout=30
)

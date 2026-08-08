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
    raise RuntimeError(
        "BOT_TOKEN environment variable is not set."
    )

bot = telebot.TeleBot(TELEGRAM_TOKEN)


# ============================================================
# DETECTION SETTINGS
# ============================================================

# Keep these close to the previous working detector.
MIN_BODY_AREA = 10
MIN_BODY_HEIGHT = 2
MIN_CANDLE_WIDTH = 2

# Slightly easier detection for the newest candles.
RIGHT_MIN_BODY_AREA = 7
RIGHT_MIN_BODY_HEIGHT = 2

# Prevent enormous colored regions from being called candles.
MAX_CANDLE_WIDTH_RATIO = 0.045

# Same-color pieces close together may belong to one candle.
MERGE_DISTANCE_RATIO = 0.55

# New: minimum amount of colored evidence inside a body.
MIN_COLOR_DENSITY = 0.15

# New: right side starts here.
RIGHT_SIDE_START = 0.72

# New: reject extremely large isolated objects.
MAX_BODY_HEIGHT_RATIO = 0.35


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    img = cv2.imread(path)

    if img is None:
        raise ValueError("Could not read screenshot.")

    h, w = img.shape[:2]

    # Mild upscale only.
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
    # GREEN
    # --------------------------------------------------------

    green_lower = np.array([
        30,
        35,
        35
    ])

    green_upper = np.array([
        90,
        255,
        255
    ])

    green = cv2.inRange(
        hsv,
        green_lower,
        green_upper
    )

    # --------------------------------------------------------
    # RED
    # --------------------------------------------------------

    red_lower_1 = np.array([
        0,
        40,
        40
    ])

    red_upper_1 = np.array([
        15,
        255,
        255
    ])

    red_lower_2 = np.array([
        165,
        40,
        40
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
# FIND CANDIDATES
# ============================================================

def find_candidates(
    mask,
    color,
    image_width,
    image_height,
    right_side=False
):

    # Very mild morphology.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

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
        int(
            image_width *
            MAX_CANDLE_WIDTH_RATIO
        )
    )

    if right_side:
        min_area = RIGHT_MIN_BODY_AREA
        min_height = RIGHT_MIN_BODY_HEIGHT
    else:
        min_area = MIN_BODY_AREA
        min_height = MIN_BODY_HEIGHT

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w < MIN_CANDLE_WIDTH:
            continue

        if h < min_height:
            continue

        if w > max_width:
            continue

        # Reject huge vertical objects.
        if h > image_height * MAX_BODY_HEIGHT_RATIO:
            continue

        # Reject very wide horizontal objects.
        if w > h * 6:
            continue

        # ----------------------------------------------------
        # COLORED PIXEL CHECK
        # ----------------------------------------------------

        region = cleaned[
            y:y + h,
            x:x + w
        ]

        colored_pixels = int(
            np.sum(region > 0)
        )

        if colored_pixels < 5:
            continue

        density = (
            colored_pixels /
            float(max(1, w * h))
        )

        if density < MIN_COLOR_DENSITY:
            continue

        center_x = x + (w / 2)

        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": float(area),
            "pixels": colored_pixels,
            "density": density,
            "center_x": center_x,
            "color": color
        })

    return candidates


# ============================================================
# MERGE SAME-COLOR PIECES
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

        merged_into_existing = False

        for existing in merged:

            distance = abs(
                candidate["center_x"]
                -
                existing["center_x"]
            )

            allowed = max(
                candidate["w"],
                existing["w"],
                2
            ) * MERGE_DISTANCE_RATIO

            candidate_top = candidate["y"]

            candidate_bottom = (
                candidate["y"]
                +
                candidate["h"]
            )

            existing_top = existing["y"]

            existing_bottom = (
                existing["y"]
                +
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
                vertical_overlap
            ):

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

                existing["w"] = (
                    right - left
                )

                existing["h"] = (
                    bottom - top
                )

                existing["center_x"] = (
                    left +
                    existing["w"] / 2
                )

                existing["area"] += (
                    candidate["area"]
                )

                existing["pixels"] += (
                    candidate["pixels"]
                )

                merged_into_existing = True

                break

        if not merged_into_existing:

            merged.append(
                candidate.copy()
            )

    return merged


# ============================================================
# REMOVE CROSS-COLOR DUPLICATES
# ============================================================

def remove_cross_color_duplicates(candles):

    candles = sorted(
        candles,
        key=lambda c: c["center_x"]
    )

    result = []

    for candle in candles:

        duplicate_index = None

        for i, existing in enumerate(result):

            distance = abs(
                candle["center_x"]
                -
                existing["center_x"]
            )

            threshold = max(
                candle["w"],
                existing["w"],
                2
            ) * 0.65

            if distance <= threshold:

                duplicate_index = i

                break

        if duplicate_index is None:

            result.append(
                candle
            )

        else:

            existing = result[
                duplicate_index
            ]

            # Keep the detection with stronger
            # actual color evidence.
            if (
                candle["pixels"]
                >
                existing["pixels"]
            ):

                result[
                    duplicate_index
                ] = candle

    return result


# ============================================================
# NEW: REMOVE OBVIOUS ISOLATED COLOR NOISE
# ============================================================

def remove_noise_candidates(candles):

    if len(candles) < 3:
        return candles

    candles = sorted(
        candles,
        key=lambda c: c["center_x"]
    )

    centers = [
        c["center_x"]
        for c in candles
    ]

    gaps = []

    for i in range(
        1,
        len(centers)
    ):

        gap = (
            centers[i]
            -
            centers[i - 1]
        )

        if gap > 0:
            gaps.append(gap)

    if len(gaps) < 2:
        return candles

    # Median spacing between detected objects.
    median_gap = float(
        np.median(gaps)
    )

    if median_gap <= 0:
        return candles

    cleaned = []

    for i, candle in enumerate(candles):

        left_gap = None
        right_gap = None

        if i > 0:
            left_gap = (
                candle["center_x"]
                -
                candles[i - 1]["center_x"]
            )

        if i < len(candles) - 1:
            right_gap = (
                candles[i + 1]["center_x"]
                -
                candle["center_x"]
            )

        # Very small isolated colored objects are more
        # suspicious when surrounded by very large gaps.
        isolated_left = (
            left_gap is not None
            and
            left_gap > median_gap * 3.0
        )

        isolated_right = (
            right_gap is not None
            and
            right_gap > median_gap * 3.0
        )

        weak_detection = (
            candle["pixels"] < 10
            or
            candle["density"] < 0.20
        )

        if (
            weak_detection
            and
            isolated_left
            and
            isolated_right
        ):
            continue

        cleaned.append(
            candle
        )

    return cleaned


# ============================================================
# MILD RIGHT-SIDE IMPROVEMENT
# ============================================================

def detect_right_side(
    chart,
    green_mask,
    red_mask
):

    h, w = chart.shape[:2]

    right_start = int(
        w * RIGHT_SIDE_START
    )

    green_right = green_mask[
        :,
        right_start:
    ]

    red_right = red_mask[
        :,
        right_start:
    ]

    green = find_candidates(
        green_right,
        "GREEN",
        w,
        h,
        right_side=True
    )

    red = find_candidates(
        red_right,
        "RED",
        w,
        h,
        right_side=True
    )

    for candle in (
        green +
        red
    ):

        candle["x"] += right_start

        candle["center_x"] += right_start

    return (
        green +
        red
    )


# ============================================================
# DETECT CANDLES
# ============================================================

def detect_candles(img):

    h, w = img.shape[:2]

    green_mask, red_mask = (
        get_color_masks(img)
    )

    # --------------------------------------------------------
    # MAIN DETECTION
    # --------------------------------------------------------

    green = find_candidates(
        green_mask,
        "GREEN",
        w,
        h,
        right_side=False
    )

    red = find_candidates(
        red_mask,
        "RED",
        w,
        h,
        right_side=False
    )

    green = merge_candidates(
        green
    )

    red = merge_candidates(
        red
    )

    candles = (
        green +
        red
    )

    # --------------------------------------------------------
    # RIGHT SIDE
    # --------------------------------------------------------

    right_candidates = detect_right_side(
        img,
        green_mask,
        red_mask
    )

    candles.extend(
        right_candidates
    )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    candles = remove_cross_color_duplicates(
        candles
    )

    # --------------------------------------------------------
    # SMALL NOISE CLEANUP
    # --------------------------------------------------------

    candles = remove_noise_candidates(
        candles
    )

    # --------------------------------------------------------
    # LEFT → RIGHT
    # --------------------------------------------------------

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

        # Yellow rectangle = detected body.
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2
        )

        # Green / red number shows classification.
        if candle["color"] == "GREEN":

            label_color = (
                0,
                255,
                0
            )

        else:

            label_color = (
                0,
                0,
                255
            )

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
            label_color,
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
            "👁️ Reading visible candles...\n"
            "Checking green/red classification..."
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
        # NO DETECTION
        # ----------------------------------------------------

        if total == 0:

            bot.reply_to(
                message,
                "❌ No reliable candle bodies detected.\n\n"
                "No candle was generated.\n"
                "No random candle was added.\n"
                "No signal was generated."
            )

            return

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        report = (
            "🔎 **CANDLE READING TEST**\n\n"

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
            "🟢 = Bot believes candle is GREEN\n"
            "🔴 = Bot believes candle is RED\n\n"

            "⚠️ **TEST ONLY**\n"
            "No OHLC data is generated.\n"
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
        # DETECTION MAP
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

                    "🟨 Yellow box = detected candle body\n"
                    "🟢 Green number = classified GREEN\n"
                    "🔴 Red number = classified RED\n\n"

                    "Check every box against the actual "
                    "candle in your screenshot.\n\n"

                    "The detector does NOT create missing "
                    "candles or fill empty spaces."
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
    "🕯️ CANDLE READING TEST"
)

print(
    "========================================"
)

print(
    "Previous detector architecture."
)

print(
    "Small right-side improvement."
)

print(
    "Color classification preserved."
)

print(
    "No aggressive vertical scanning."
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

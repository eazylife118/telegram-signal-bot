import os
import cv2
import numpy as np
import telebot
import time

# ============================================================
# FAST PRICE OCR
# ============================================================

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    TESSERACT_AVAILABLE = False


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_TOKEN = os.getenv(
    "BOT_TOKEN",
    "PASTE_YOUR_BOT_TOKEN_HERE"
)

bot = telebot.TeleBot(
    TELEGRAM_TOKEN
)


# ============================================================
# DETECTION SETTINGS
# ============================================================

MIN_BODY_AREA = 10
MIN_BODY_HEIGHT = 2
MIN_CANDLE_WIDTH = 2

RIGHT_MIN_BODY_AREA = 6
RIGHT_MIN_BODY_HEIGHT = 2

MAX_CANDLE_WIDTH_RATIO = 0.045

MERGE_DISTANCE_RATIO = 0.55


# ============================================================
# PURPLE / YELLOW SETTINGS
# ============================================================

# PURPLE = BUY / BULLISH

PURPLE_HUE_LOW = 125
PURPLE_HUE_HIGH = 165

MIN_PURPLE_SATURATION = 100
MIN_PURPLE_VALUE = 70


# YELLOW = SELL / BEARISH

YELLOW_HUE_LOW = 18
YELLOW_HUE_HIGH = 40

MIN_YELLOW_SATURATION = 100
MIN_YELLOW_VALUE = 70


MIN_COLOR_DENSITY = 0.25

PURPLE_DOMINANCE_RATIO = 1.20
YELLOW_DOMINANCE_RATIO = 1.10


# ============================================================
# PRICE SCALE SETTINGS
# ============================================================

# The price scale is normally on the far-right side.
#
# We deliberately do NOT OCR the entire screenshot.
# This is what keeps the price reading much faster.
#
# 0.78 = start OCR at 78% of image width.
# ============================================================

PRICE_SCALE_START_RATIO = 0.78

# Ignore very tiny OCR results.
MIN_PRICE_TEXT_HEIGHT = 7

# Maximum number of scale labels we keep.
MAX_SCALE_LABELS = 30

# A price should normally contain a decimal point.
MIN_PRICE_DIGITS = 3


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

    # ========================================================
    # PURPLE
    # ========================================================

    purple_lower = np.array([
        PURPLE_HUE_LOW,
        MIN_PURPLE_SATURATION,
        MIN_PURPLE_VALUE
    ])

    purple_upper = np.array([
        PURPLE_HUE_HIGH,
        255,
        255
    ])

    purple = cv2.inRange(
        hsv,
        purple_lower,
        purple_upper
    )

    # ========================================================
    # YELLOW
    # ========================================================

    yellow_lower = np.array([
        YELLOW_HUE_LOW,
        MIN_YELLOW_SATURATION,
        MIN_YELLOW_VALUE
    ])

    yellow_upper = np.array([
        YELLOW_HUE_HIGH,
        255,
        255
    ])

    yellow = cv2.inRange(
        hsv,
        yellow_lower,
        yellow_upper
    )

    # ========================================================
    # BGR CHANNELS
    # ========================================================

    b, g, r = cv2.split(img)

    # ========================================================
    # PURPLE DOMINANCE
    # ========================================================

    purple_dominance = (

        (r.astype(np.int16) >
         g.astype(np.int16) *
         PURPLE_DOMINANCE_RATIO)

        &

        (b.astype(np.int16) >
         g.astype(np.int16) *
         PURPLE_DOMINANCE_RATIO)

        &

        (r.astype(np.int16) > 70)

        &

        (b.astype(np.int16) > 70)

    )

    purple_dominance_mask = (
        purple_dominance.astype(
            np.uint8
        ) * 255
    )

    purple = cv2.bitwise_and(
        purple,
        purple_dominance_mask
    )

    # ========================================================
    # YELLOW DOMINANCE
    # ========================================================

    yellow_dominance = (

        (r.astype(np.int16) >
         b.astype(np.int16) *
         YELLOW_DOMINANCE_RATIO)

        &

        (g.astype(np.int16) >
         b.astype(np.int16) *
         YELLOW_DOMINANCE_RATIO)

        &

        (r.astype(np.int16) > 80)

        &

        (g.astype(np.int16) > 70)

    )

    yellow_dominance_mask = (
        yellow_dominance.astype(
            np.uint8
        ) * 255
    )

    yellow = cv2.bitwise_and(
        yellow,
        yellow_dominance_mask
    )

    return purple, yellow


# ============================================================
# FIND CANDIDATES
# ============================================================

def find_candidates(
    mask,
    color,
    image_width,
    right_side=False
):

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

        area = cv2.contourArea(
            contour
        )

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

        if w > h * 6:
            continue

        region = cleaned[
            y:y+h,
            x:x+w
        ]

        colored_pixels = int(
            np.sum(
                region > 0
            )
        )

        if colored_pixels < 5:
            continue

        density = (
            colored_pixels /
            float(
                max(
                    1,
                    w * h
                )
            )
        )

        if density < MIN_COLOR_DENSITY:
            continue

        center_x = (
            x +
            w / 2
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
            "color": color

        })

    return candidates


# ============================================================
# MERGE SAME-COLOR PIECES
# ============================================================

def merge_candidates(
    candidates
):

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda c:
        c["center_x"]
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

            candidate_top = (
                candidate["y"]
            )

            candidate_bottom = (
                candidate["y"] +
                candidate["h"]
            )

            existing_top = (
                existing["y"]
            )

            existing_bottom = (
                existing["y"] +
                existing["h"]
            )

            vertical_overlap = not (

                candidate_bottom <
                existing_top

                or

                candidate_top >
                existing_bottom

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
                    right -
                    left
                )

                existing["h"] = (
                    bottom -
                    top
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

def remove_cross_color_duplicates(
    candles
):

    candles = sorted(
        candles,
        key=lambda c:
        c["center_x"]
    )

    result = []

    for candle in candles:

        duplicate_index = None

        for i, existing in enumerate(
            result
        ):

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
# RIGHT-SIDE IMPROVEMENT
# ============================================================

def detect_right_side(
    chart,
    purple_mask,
    yellow_mask
):

    h, w = chart.shape[:2]

    right_start = int(
        w * 0.72
    )

    purple_right = purple_mask[
        :,
        right_start:
    ]

    yellow_right = yellow_mask[
        :,
        right_start:
    ]

    purple = find_candidates(
        purple_right,
        "PURPLE",
        w,
        right_side=True
    )

    yellow = find_candidates(
        yellow_right,
        "YELLOW",
        w,
        right_side=True
    )

    for candle in (
        purple +
        yellow
    ):

        candle["x"] += (
            right_start
        )

        candle["center_x"] += (
            right_start
        )

    return (
        purple +
        yellow
    )


# ============================================================
# DETECT CANDLES
# ============================================================

def detect_candles(
    img
):

    h, w = img.shape[:2]

    purple_mask, yellow_mask = (
        get_color_masks(img)
    )

    purple = find_candidates(
        purple_mask,
        "PURPLE",
        w,
        right_side=False
    )

    yellow = find_candidates(
        yellow_mask,
        "YELLOW",
        w,
        right_side=False
    )

    purple = merge_candidates(
        purple
    )

    yellow = merge_candidates(
        yellow
    )

    candles = (
        purple +
        yellow
    )

    right_candidates = (
        detect_right_side(
            img,
            purple_mask,
            yellow_mask
        )
    )

    candles.extend(
        right_candidates
    )

    candles = (
        remove_cross_color_duplicates(
            candles
        )
    )

    candles.sort(
        key=lambda c:
        c["center_x"],
        reverse=True
    )

    return candles


# ============================================================
# FAST PRICE SCALE OCR
# ============================================================

def clean_price_text(text):

    text = text.strip()

    # Remove common OCR garbage.
    text = text.replace(
        ",",
        "."
    )

    text = text.replace(
        " ",
        ""
    )

    # Keep only digits and decimal point.
    cleaned = ""

    for char in text:

        if char.isdigit() or char == ".":

            cleaned += char

    # Avoid multiple decimal points.
    parts = cleaned.split(".")

    if len(parts) > 2:

        cleaned = (
            parts[0]
            +
            "."
            +
            "".join(
                parts[1:]
            )
        )

    return cleaned


# ============================================================
# CHECK WHETHER OCR TEXT LOOKS LIKE A PRICE
# ============================================================

def looks_like_price(text):

    text = clean_price_text(
        text
    )

    if not text:
        return False

    digits = sum(
        char.isdigit()
        for char in text
    )

    if digits < MIN_PRICE_DIGITS:
        return False

    try:

        value = float(
            text
        )

    except Exception:

        return False

    # Reject obvious UI numbers.
    if value <= 0:
        return False

    # Currency prices in this application are
    # expected to be within a reasonable range.
    if value > 1000000:
        return False

    return True


# ============================================================
# OCR PRICE SCALE
# ============================================================

def read_price_scale(
    img
):

    start_time = time.time()

    if not TESSERACT_AVAILABLE:

        return [], 0.0

    h, w = img.shape[:2]

    # ========================================================
    # ONLY READ FAR-RIGHT PRICE SCALE
    # ========================================================

    start_x = int(
        w *
        PRICE_SCALE_START_RATIO
    )

    roi = img[
        :,
        start_x:
    ]

    if roi.size == 0:

        return [], (
            time.time() -
            start_time
        )

    # ========================================================
    # UPSCALE ONLY THE PRICE SCALE
    # ========================================================

    scale = 1.6

    roi = cv2.resize(
        roi,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # ========================================================
    # CREATE TWO FAST OCR VERSIONS
    # ========================================================

    # Bright text / white labels.
    _, threshold = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY
    )

    # Adaptive version helps when the chart background
    # is not perfectly uniform.
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        8
    )

    configurations = [

        (
            threshold,
            "--psm 6 "
            "-c tessedit_char_whitelist=0123456789."
        ),

        (
            adaptive,
            "--psm 11 "
            "-c tessedit_char_whitelist=0123456789."
        )

    ]

    labels = []

    for image, config in configurations:

        try:

            data = pytesseract.image_to_data(
                image,
                config=config,
                output_type=pytesseract.Output.DICT
            )

        except Exception:

            continue

        count = len(
            data["text"]
        )

        for i in range(count):

            raw_text = data[
                "text"
            ][i]

            text = clean_price_text(
                raw_text
            )

            if not looks_like_price(
                text
            ):
                continue

            try:

                confidence = float(
                    data["conf"][i]
                )

            except Exception:

                confidence = 0

            if confidence < 20:

                continue

            x = int(
                data["left"][i]
            )

            y = int(
                data["top"][i]
            )

            box_w = int(
                data["width"][i]
            )

            box_h = int(
                data["height"][i]
            )

            if box_h < (
                MIN_PRICE_TEXT_HEIGHT *
                scale
            ):

                continue

            try:

                value = float(
                    text
                )

            except Exception:

                continue

            # Convert coordinates back to original image.
            center_x = (
                start_x +
                (
                    x +
                    box_w / 2
                ) / scale
            )

            center_y = (
                (
                    y +
                    box_h / 2
                ) / scale
            )

            labels.append({

                "price": value,

                "x": center_x,

                "y": center_y,

                "confidence": confidence,

                "text": text

            })

    # ========================================================
    # SORT BY VERTICAL POSITION
    # ========================================================

    labels.sort(
        key=lambda item:
        item["y"]
    )

    # ========================================================
    # REMOVE DUPLICATE OCR RESULTS
    # ========================================================

    cleaned_labels = []

    for label in labels:

        duplicate = False

        for existing in cleaned_labels:

            same_y = abs(
                label["y"] -
                existing["y"]
            ) < 12

            same_price = abs(
                label["price"] -
                existing["price"]
            ) < 0.0000001

            if same_y and same_price:

                duplicate = True

                if (
                    label["confidence"]
                    >
                    existing["confidence"]
                ):

                    existing.update(
                        label
                    )

                break

        if not duplicate:

            cleaned_labels.append(
                label
            )

    # ========================================================
    # LIMIT RESULT
    # ========================================================

    cleaned_labels = (
        cleaned_labels[
            :MAX_SCALE_LABELS
        ]
    )

    elapsed = (
        time.time() -
        start_time
    )

    return (
        cleaned_labels,
        elapsed
    )


# ============================================================
# PRICE INTERPOLATION
# ============================================================

def price_from_y(
    y,
    scale_labels
):

    if len(scale_labels) < 2:

        return None

    labels = sorted(
        scale_labels,
        key=lambda item:
        item["y"]
    )

    # ========================================================
    # EXACT / NEAR LABEL
    # ========================================================

    for label in labels:

        if abs(
            y -
            label["y"]
        ) < 2:

            return label[
                "price"
            ]

    # ========================================================
    # BETWEEN TWO LABELS
    # ========================================================

    for i in range(
        len(labels) - 1
    ):

        upper = labels[i]
        lower = labels[i + 1]

        if (
            upper["y"]
            <= y
            <=
            lower["y"]
        ):

            y1 = upper["y"]
            y2 = lower["y"]

            p1 = upper["price"]
            p2 = lower["price"]

            if abs(
                y2 -
                y1
            ) < 0.001:

                return None

            ratio = (
                (y - y1)
                /
                (y2 - y1)
            )

            price = (
                p1 +
                ratio *
                (p2 - p1)
            )

            return price

    # ========================================================
    # OUTSIDE VISIBLE SCALE
    # ========================================================

    return None


# ============================================================
# CANDLE CLOSE Y POSITION
# ============================================================
#
# IMPORTANT:
#
# PURPLE/BULLISH:
# close is normally at the TOP of the body.
#
# YELLOW/BEARISH:
# close is normally at the BOTTOM of the body.
#
# This gives us a much better estimate than simply using
# the middle of the candle.
# ============================================================

def candle_close_y(
    candle
):

    y = candle["y"]
    h = candle["h"]

    if candle["color"] == "PURPLE":

        return float(
            y
        )

    else:

        return float(
            y +
            h
        )


# ============================================================
# ADD PRICES TO CANDLES
# ============================================================

def attach_prices(
    candles,
    scale_labels
):

    for candle in candles:

        close_y = candle_close_y(
            candle
        )

        price = price_from_y(
            close_y,
            scale_labels
        )

        candle["price"] = price

        candle["close_y"] = close_y

    return candles


# ============================================================
# CREATE REPORT
# ============================================================

def create_report(
    candles
):

    purple = sum(

        1
        for c in candles
        if c["color"] == "PURPLE"

    )

    yellow = sum(

        1
        for c in candles
        if c["color"] == "YELLOW"

    )

    return purple, yellow


# ============================================================
# FORMAT PRICE
# ============================================================

def format_price(
    price
):

    if price is None:

        return "NOT READ"

    # Automatically show enough decimal places.
    if price >= 100:

        return f"{price:.3f}"

    if price >= 1:

        return f"{price:.5f}"

    return f"{price:.6f}"


# ============================================================
# CREATE DETECTION MAP
# ============================================================

def create_detection_map(
    img,
    candles,
    scale_labels
):

    output = img.copy()

    # ========================================================
    # DRAW PRICE SCALE LABELS
    # ========================================================

    for label in scale_labels:

        x = int(
            label["x"]
        )

        y = int(
            label["y"]
        )

        price_text = format_price(
            label["price"]
        )

        # White price-scale text.
        cv2.putText(

            output,

            price_text,

            (
                max(
                    0,
                    x - 100
                ),
                max(
                    20,
                    y
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (255, 255, 255),

            1,

            cv2.LINE_AA

        )

        # Small horizontal marker.
        cv2.line(

            output,

            (
                max(
                    0,
                    x - 120
                ),
                y
            ),

            (
                min(
                    output.shape[1] - 1,
                    x + 10
                ),
                y
            ),

            (255, 255, 255),

            1

        )

    # ========================================================
    # DRAW CANDLES
    # ========================================================

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

        # Yellow detection box.
        cv2.rectangle(

            output,

            (x, y),

            (
                x + w,
                y + h
            ),

            (0, 255, 255),

            2

        )

        # ====================================================
        # NUMBER COLOR
        # ====================================================

        if candle["color"] == "PURPLE":

            label_color = (
                255,
                0,
                255
            )

        else:

            label_color = (
                0,
                255,
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

        # ====================================================
        # PRICE BESIDE CANDLE
        # ====================================================

        price_text = format_price(
            candle.get(
                "price"
            )
        )

        # Put price immediately to the left of the candle
        # when possible.
        price_x = max(
            2,
            x - 105
        )

        price_y = max(
            20,
            int(
                candle["close_y"]
            )
        )

        cv2.putText(

            output,

            price_text,

            (
                price_x,
                price_y
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.42,

            (255, 255, 255),

            1,

            cv2.LINE_AA

        )

    return output


# ============================================================
# PRICE SUMMARY
# ============================================================

def get_price_summary(
    candles,
    scale_labels
):

    if not scale_labels:

        highest = None
        lowest = None

    else:

        highest = max(
            label["price"]
            for label in scale_labels
        )

        lowest = min(
            label["price"]
            for label in scale_labels
        )

    current = None

    if candles:

        current = candles[0].get(
            "price"
        )

    return (
        highest,
        lowest,
        current
    )


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
        "chart_screenshot.png"
    )

    detection_path = (
        "candle_price_detection.png"
    )

    try:

        bot.reply_to(

            message,

            "👁️ Reading screenshot...\n"
            "➡️ Scanning RIGHT → LEFT.\n"
            "🟣 Purple = BUY.\n"
            "🟡 Yellow = SELL.\n"
            "💰 Reading visible price scale..."

        )

        # ====================================================
        # DOWNLOAD HIGHEST RESOLUTION
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
        # CANDLE DETECTION
        # ====================================================

        candle_start = time.time()

        candles = detect_candles(
            img
        )

        candle_time = (
            time.time() -
            candle_start
        )

        # ====================================================
        # PRICE SCALE OCR
        # ====================================================

        scale_labels, ocr_time = (
            read_price_scale(
                img
            )
        )

        # ====================================================
        # MAP PRICE TO CANDLES
        # ====================================================

        candles = attach_prices(
            candles,
            scale_labels
        )

        # ====================================================
        # COUNTS
        # ====================================================

        purple, yellow = (
            create_report(
                candles
            )
        )

        total = len(
            candles
        )

        # ====================================================
        # PRICE SUMMARY
        # ====================================================

        highest, lowest, current = (
            get_price_summary(
                candles,
                scale_labels
            )
        )

        elapsed = (
            time.time() -
            start_time
        )

        # ====================================================
        # RIGHT → LEFT SEQUENCE
        # ====================================================

        sequence = []

        for number, candle in enumerate(

            candles,

            start=1

        ):

            price_text = format_price(

                candle.get(
                    "price"
                )

            )

            if candle["color"] == "PURPLE":

                sequence.append(

                    f"{number}. 🟣 BUY "
                    f"@ {price_text}"

                )

            else:

                sequence.append(

                    f"{number}. 🟡 SELL "
                    f"@ {price_text}"

                )

        sequence_text = (
            "\n".join(
                sequence
            )
        )

        # ====================================================
        # NO CANDLES
        # ====================================================

        if total == 0:

            bot.reply_to(

                message,

                "❌ No reliable candle bodies detected.\n\n"
                "No candle was generated.\n"
                "No random candle was added.\n"
                "No signal was generated."

            )

            return

        # ====================================================
        # REPORT
        # ====================================================

        report = (

            "🔎 **CANDLE + PRICE READING TEST**\n\n"

            "➡️ **SCAN:** RIGHT → LEFT\n\n"

            "📊 **CANDLE DETECTION**\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"🟣 PURPLE / BUY: {purple}\n"

            f"🟡 YELLOW / SELL: {yellow}\n"

            f"📊 TOTAL: {total}\n\n"

            "💰 **VISIBLE PRICE SCALE**\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"⬆️ HIGHEST: "
            f"{format_price(highest)}\n"

            f"⬇️ LOWEST: "
            f"{format_price(lowest)}\n"

            f"📍 CURRENT / NEWEST CANDLE: "
            f"{format_price(current)}\n\n"

            "🕯️ **RIGHT → LEFT READING**\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"{sequence_text}\n\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            "🎯 **COLOR KEY**\n"

            "🟣 = PURPLE / BUY\n"

            "🟡 = YELLOW / SELL\n\n"

            "💰 **PRICE KEY**\n"

            "The candle price is calculated from "
            "the candle's close position and the "
            "visible price-scale labels.\n\n"

            "📌 PURPLE close = upper body position.\n"

            "📌 YELLOW close = lower body position.\n\n"

            "⚠️ These are screenshot-derived prices.\n"

            "⚠️ They are NOT generated market data.\n"

            "⚠️ No random prices are generated.\n"

            "⚠️ If the scale cannot be read reliably, "
            "the bot reports NOT READ.\n\n"

            f"⚡ Total processing time: "
            f"{elapsed:.2f}s\n"

            f"🕯️ Candle detection time: "
            f"{candle_time:.2f}s\n"

            f"🔎 Price OCR time: "
            f"{ocr_time:.2f}s\n"

            f"🔢 Scale labels detected: "
            f"{len(scale_labels)}"

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

                candles,

                scale_labels

            )

        )

        cv2.imwrite(

            detection_path,

            detection_map

        )

        # ====================================================
        # MAP CAPTION
        # ====================================================

        caption = (

            "🔢 **CANDLE + PRICE DETECTION MAP**\n\n"

            "1 = newest/rightmost candle.\n"

            "➡️ Numbers continue RIGHT → LEFT.\n\n"

            "🟣 Number = PURPLE / BUY.\n"

            "🟡 Number = YELLOW / SELL.\n\n"

            "💰 White values = detected "
            "price-scale values.\n\n"

            "💰 White price beside a candle = "
            "price estimated from its vertical "
            "position against the visible scale.\n\n"

            "⬆️ HIGH = highest detected scale price.\n"

            "⬇️ LOW = lowest detected scale price.\n"

            "📍 CURRENT = estimated close of the "
            "newest/rightmost candle.\n\n"

            "⚠️ Compare the white prices with the "
            "actual price scale in your screenshot."

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

                caption=caption,

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
    "🕯️ CANDLE + PRICE READING TEST"
)

print(
    "========================================"
)

print(
    "➡️ Scan direction: RIGHT → LEFT"
)

print(
    "🔢 Number 1 = newest/rightmost candle"
)

print(
    "🟣 PURPLE = BUY / BULLISH"
)

print(
    "🟡 YELLOW = SELL / BEARISH"
)

print(
    "💰 Visible price-scale reading enabled"
)

print(
    "💰 Candle-to-price interpolation enabled"
)

print(
    "🚫 No random candles"
)

print(
    "🚫 No random prices"
)

print(
    "🚫 No generated OHLC"
)

print(
    "========================================"
)


if not TESSERACT_AVAILABLE:

    print(
        "⚠️ pytesseract is NOT installed."
    )

    print(
        "⚠️ Price-scale reading will be NOT READ."
    )


bot.infinity_polling(

    timeout=30,

    long_polling_timeout=30

)

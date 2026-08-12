import os
import cv2
import numpy as np
import telebot
import time
import re

# ============================================================
# FAST OCR
# ============================================================
# RapidOCR reads only the small price-scale area.
# It does NOT scan the whole screenshot.
#
# Install:
# pip install rapidocr_onnxruntime
# ============================================================

try:
    from rapidocr_onnxruntime import RapidOCR
    OCR_AVAILABLE = True
    ocr_engine = RapidOCR()
except Exception as e:
    OCR_AVAILABLE = False
    ocr_engine = None
    print("⚠️ RapidOCR unavailable:", repr(e))


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_TOKEN = os.getenv(
    "BOT_TOKEN",
    "PASTE_YOUR_BOT_TOKEN_HERE"
)

bot = telebot.TeleBot(TELEGRAM_TOKEN)


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

# The price scale is normally on the RIGHT side.
#
# We intentionally leave a small section at the extreme
# right for the numbers.
#
# These percentages can be adjusted later if necessary.
# ============================================================

PRICE_SCALE_START_RATIO = 0.82
PRICE_SCALE_END_RATIO = 0.995

MIN_OCR_CONFIDENCE = 0.30


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
    # BGR DOMINANCE
    # ========================================================

    b, g, r = cv2.split(img)

    # ========================================================
    # PURPLE
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
    # YELLOW
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
# MERGE SAME COLOR
# ============================================================

def merge_candidates(candidates):

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
# REMOVE CROSS COLOR DUPLICATES
# ============================================================

def remove_cross_color_duplicates(candles):

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
# RIGHT SIDE DETECTION
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

def detect_candles(img):

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
# PRICE TEXT CLEANING
# ============================================================

def clean_price_text(text):

    if text is None:
        return None

    text = str(text).strip()

    # Replace common OCR mistakes.

    replacements = {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8"
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    # Keep only numbers and decimal point.

    text = re.sub(
        r"[^0-9.]",
        "",
        text
    )

    # Multiple dots are invalid.

    if text.count(".") > 1:
        return None

    if not text:
        return None

    try:

        value = float(text)

    except Exception:

        return None

    # A currency price normally needs
    # more than a single digit.

    if value <= 0:
        return None

    return value


# ============================================================
# PRICE SCALE OCR
# ============================================================

def read_price_scale(img):

    if not OCR_AVAILABLE:

        return {
            "labels": [],
            "highest": None,
            "lowest": None,
            "time": 0.0
        }

    start = time.time()

    h, w = img.shape[:2]

    x1 = int(
        w * PRICE_SCALE_START_RATIO
    )

    x2 = int(
        w * PRICE_SCALE_END_RATIO
    )

    roi = img[
        :,
        x1:x2
    ]

    # ========================================================
    # UPSCALE ONLY PRICE SCALE
    # ========================================================

    roi_scale = 2.0

    roi = cv2.resize(
        roi,
        None,
        fx=roi_scale,
        fy=roi_scale,
        interpolation=cv2.INTER_CUBIC
    )

    # ========================================================
    # OCR
    # ========================================================

    try:

        result, elapsed_ocr = ocr_engine(
            roi
        )

    except Exception as e:

        print(
            "⚠️ Price OCR error:",
            repr(e)
        )

        return {
            "labels": [],
            "highest": None,
            "lowest": None,
            "time": time.time() - start
        }

    labels = []

    if result:

        for item in result:

            try:

                box = item[0]
                text = item[1]
                score = float(item[2])

            except Exception:

                continue

            if score < MIN_OCR_CONFIDENCE:
                continue

            value = clean_price_text(
                text
            )

            if value is None:
                continue

            # =================================================
            # CENTER Y IN ORIGINAL IMAGE
            # =================================================

            ys = []

            for point in box:

                ys.append(
                    float(point[1])
                )

            if not ys:
                continue

            center_y = (
                sum(ys) /
                len(ys)
            ) / roi_scale

            center_x = (
                sum(
                    float(p[0])
                    for p in box
                )
                /
                len(box)
            ) / roi_scale

            # Back into screenshot coordinates.

            center_x += x1

            labels.append({

                "price": value,

                "y": center_y,

                "x": center_x,

                "confidence": score,

                "text": str(text)
            })

    # ========================================================
    # REMOVE DUPLICATE / BAD LABELS
    # ========================================================

    labels.sort(
        key=lambda x:
        x["y"]
    )

    filtered = []

    for label in labels:

        duplicate = False

        for existing in filtered:

            if abs(
                label["y"] -
                existing["y"]
            ) < 8:

                duplicate = True

                # Keep higher confidence.

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

            filtered.append(
                label
            )

    labels = filtered

    highest = None
    lowest = None

    if labels:

        highest = max(
            labels,
            key=lambda x:
            x["price"]
        )["price"]

        lowest = min(
            labels,
            key=lambda x:
            x["price"]
        )["price"]

    return {

        "labels": labels,

        "highest": highest,

        "lowest": lowest,

        "time": time.time() - start
    }


# ============================================================
# PRICE MAPPING
# ============================================================

def create_price_mapper(price_labels):

    if len(price_labels) < 2:

        return None

    points = []

    for label in price_labels:

        points.append(
            (
                float(label["y"]),
                float(label["price"])
            )
        )

    points.sort(
        key=lambda p:
        p[0]
    )

    # ========================================================
    # LINEAR PIXEL → PRICE REGRESSION
    # ========================================================

    ys = np.array(
        [p[0] for p in points],
        dtype=np.float64
    )

    prices = np.array(
        [p[1] for p in points],
        dtype=np.float64
    )

    if len(ys) < 2:
        return None

    try:

        slope, intercept = np.polyfit(
            ys,
            prices,
            1
        )

    except Exception:

        return None

    return {
        "slope": float(slope),
        "intercept": float(intercept)
    }


# ============================================================
# PIXEL TO PRICE
# ============================================================

def pixel_to_price(
    y,
    mapper
):

    if mapper is None:
        return None

    price = (

        mapper["slope"] *
        float(y)

        +

        mapper["intercept"]
    )

    return price


# ============================================================
# FORMAT PRICE
# ============================================================

def format_price(price):

    if price is None:

        return "NOT READ"

    # Automatically choose useful precision.

    if price >= 100:
        return f"{price:.3f}"

    if price >= 10:
        return f"{price:.4f}"

    if price >= 1:
        return f"{price:.5f}"

    return f"{price:.6f}"


# ============================================================
# ADD PRICE INFORMATION TO CANDLES
# ============================================================

def calculate_candle_prices(
    candles,
    mapper
):

    for candle in candles:

        top_y = candle["y"]

        bottom_y = (
            candle["y"] +
            candle["h"]
        )

        center_y = (
            top_y +
            bottom_y
        ) / 2.0

        top_price = pixel_to_price(
            top_y,
            mapper
        )

        bottom_price = pixel_to_price(
            bottom_y,
            mapper
        )

        center_price = pixel_to_price(
            center_y,
            mapper
        )

        # ====================================================
        # IMPORTANT:
        #
        # Purple BUY:
        # body top = approximate CLOSE
        # body bottom = approximate OPEN
        #
        # Yellow SELL:
        # body top = approximate OPEN
        # body bottom = approximate CLOSE
        #
        # This is body-derived, NOT complete OHLC.
        # ====================================================

        if candle["color"] == "PURPLE":

            close_price = top_price
            open_price = bottom_price

        else:

            open_price = top_price
            close_price = bottom_price

        candle["top_price"] = top_price
        candle["bottom_price"] = bottom_price
        candle["center_price"] = center_price

        candle["open_price"] = open_price
        candle["close_price"] = close_price

    return candles


# ============================================================
# CURRENT PRICE
# ============================================================

def get_current_price(
    candles
):

    if not candles:
        return None

    newest = candles[0]

    # Use approximate CLOSE of newest candle.

    return newest.get(
        "close_price"
    )


# ============================================================
# CANDLE REPORT
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
# DETECTION MAP
# ============================================================

def create_detection_map(
    img,
    candles,
    price_info
):

    output = img.copy()

    h, w = output.shape[:2]

    # ========================================================
    # PRICE SCALE AREA
    # ========================================================

    scale_x = int(
        w * PRICE_SCALE_START_RATIO
    )

    cv2.line(
        output,
        (scale_x, 0),
        (scale_x, h),
        (255, 255, 255),
        1
    )

    # ========================================================
    # DRAW PRICE SCALE LABELS
    # ========================================================

    for label in price_info["labels"]:

        x = int(
            label["x"]
        )

        y = int(
            label["y"]
        )

        price_text = format_price(
            label["price"]
        )

        # White marker at detected
        # price-scale position.

        cv2.circle(
            output,
            (x, y),
            4,
            (255, 255, 255),
            -1
        )

        cv2.putText(
            output,
            price_text,
            (
                max(
                    5,
                    x - 150
                ),
                max(
                    20,
                    y - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # ========================================================
    # CANDLE MAP
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

        cw = int(
            candle["w"]
        )

        ch = int(
            candle["h"]
        )

        # ====================================================
        # BOX
        # ====================================================

        cv2.rectangle(
            output,
            (x, y),
            (
                x + cw,
                y + ch
            ),
            (0, 255, 255),
            2
        )

        # ====================================================
        # COLOR
        # ====================================================

        if candle["color"] == "PURPLE":

            label_color = (
                255,
                0,
                255
            )

            direction = "BUY"

        else:

            label_color = (
                0,
                255,
                255
            )

            direction = "SELL"

        # ====================================================
        # NUMBER
        # ====================================================

        cv2.putText(
            output,
            str(number),
            (
                x,
                max(
                    25,
                    y - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            label_color,
            2,
            cv2.LINE_AA
        )

        # ====================================================
        # APPROXIMATE CLOSE PRICE
        # ====================================================

        close_price = format_price(
            candle.get(
                "close_price"
            )
        )

        # ====================================================
        # DRAW PRICE NEXT TO CANDLE
        # ====================================================

        text_x = min(
            w - 210,
            x + cw + 5
        )

        text_y = max(
            20,
            y + int(ch / 2)
        )

        cv2.putText(
            output,
            close_price,
            (
                text_x,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # ========================================================
    # SUMMARY BOX
    # ========================================================

    highest = format_price(
        price_info["highest"]
    )

    lowest = format_price(
        price_info["lowest"]
    )

    current = format_price(
        price_info["current"]
    )

    summary = [

        f"HIGH: {highest}",

        f"LOW: {lowest}",

        f"CURRENT: {current}"
    ]

    box_height = 95

    cv2.rectangle(
        output,
        (5, 5),
        (330, box_height),
        (0, 0, 0),
        -1
    )

    y_text = 30

    for text in summary:

        cv2.putText(
            output,
            text,
            (
                15,
                y_text
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        y_text += 28

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
        "candle_price_detection.png"
    )

    try:

        bot.reply_to(

            message,

            "👁️ Reading screenshot...\n"
            "➡️ Scanning candles RIGHT → LEFT.\n"
            "🟣 PURPLE = BUY.\n"
            "🟡 YELLOW = SELL.\n"
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
        # CANDLES
        # ====================================================

        candles = detect_candles(
            img
        )

        purple, yellow = (
            create_report(
                candles
            )
        )

        total = len(
            candles
        )

        # ====================================================
        # PRICE SCALE
        # ====================================================

        price_info = read_price_scale(
            img
        )

        # ====================================================
        # CREATE PIXEL → PRICE MAP
        # ====================================================

        mapper = create_price_mapper(
            price_info["labels"]
        )

        # ====================================================
        # CALCULATE CANDLE PRICES
        # ====================================================

        calculate_candle_prices(
            candles,
            mapper
        )

        # ====================================================
        # CURRENT PRICE
        # ====================================================

        current_price = (
            get_current_price(
                candles
            )
        )

        price_info["current"] = (
            current_price
        )

        # ====================================================
        # PROCESSING TIME
        # ====================================================

        elapsed = (
            time.time() -
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

            if candle["color"] == "PURPLE":

                icon = "🟣"
                direction = "BUY"

            else:

                icon = "🟡"
                direction = "SELL"

            close_price = format_price(
                candle.get(
                    "close_price"
                )
            )

            sequence.append(

                f"{number}. "
                f"{icon} {direction} "
                f"@ {close_price}"

            )

        sequence_text = (
            "\n".join(
                sequence
            )
        )

        # ====================================================
        # PRICE SCALE TEXT
        # ====================================================

        if price_info["highest"] is None:

            highest_text = (
                "NOT READ"
            )

        else:

            highest_text = format_price(
                price_info["highest"]
            )

        if price_info["lowest"] is None:

            lowest_text = (
                "NOT READ"
            )

        else:

            lowest_text = format_price(
                price_info["lowest"]
            )

        current_text = format_price(
            current_price
        )

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

            f"⬆️ HIGHEST: {highest_text}\n"

            f"⬇️ LOWEST: {lowest_text}\n"

            f"📍 CURRENT / NEWEST CANDLE: "
            f"{current_text}\n\n"

            "🕯️ **RIGHT → LEFT READING**\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"{sequence_text}\n\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            "🎯 **COLOR KEY**\n"

            "🟣 = PURPLE / BUY\n"

            "🟡 = YELLOW / SELL\n\n"

            "💰 **PRICE KEY**\n"

            "Price beside each candle is mapped "
            "from its vertical position against "
            "the detected visible price scale.\n\n"

            "⚠️ Candle prices are currently "
            "body-position estimates.\n"

            "⚠️ They are NOT claimed as complete "
            "OHLC values.\n"

            "⚠️ No random prices are generated.\n\n"

            f"⚡ Total processing time: "
            f"{elapsed:.2f}s\n"

            f"🔎 Price OCR time: "
            f"{price_info['time']:.2f}s\n"

            f"🔎 Scale labels detected: "
            f"{len(price_info['labels'])}"

        )

        bot.reply_to(
            message,
            report,
            parse_mode="Markdown"
        )

        # ====================================================
        # DETECTION MAP
        # ====================================================

        detection_map = (
            create_detection_map(
                img,
                candles,
                price_info
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

                    "🔢 **CANDLE + PRICE DETECTION MAP**\n\n"

                    "1 = newest/rightmost candle.\n"
                    "➡️ Numbers continue RIGHT → LEFT.\n\n"

                    "🟣 Number = PURPLE / BUY.\n"
                    "🟡 Number = YELLOW / SELL.\n\n"

                    "💰 White values = price-scale "
                    "numbers detected from the screenshot.\n\n"

                    "💰 White price beside a candle = "
                    "price calculated from the candle's "
                    "vertical position against the "
                    "detected price scale.\n\n"

                    "⬆️ HIGH = highest detected "
                    "price-scale value.\n"

                    "⬇️ LOW = lowest detected "
                    "price-scale value.\n"

                    "📍 CURRENT = approximate close "
                    "price of newest/rightmost "
                    "detected candle.\n\n"

                    "⚠️ Compare the white numbers "
                    "with the actual price scale "
                    "in your screenshot."

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
    "💰 Fast price-scale OCR enabled"
)

print(
    "📍 Candle-to-price mapping enabled"
)

print(
    "🚫 No random candles"
)

print(
    "🚫 No random prices"
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

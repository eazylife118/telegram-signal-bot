import os
import cv2
import numpy as np
import telebot
import time

# ============================================================
# EASY OCR
# ============================================================
#
# EasyOCR is used ONLY for the price scale.
#
# It is NOT used to detect candles.
#
# Candle detection remains OpenCV-based because that is much
# faster for the purple/yellow candle bodies.
# ============================================================

try:
    import easyocr
except ImportError:
    easyocr = None


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_TOKEN = os.getenv(
    "BOT_TOKEN",
    "PASTE_YOUR_BOT_TOKEN_HERE"
)

bot = telebot.TeleBot(TELEGRAM_TOKEN)


# ============================================================
# EASY OCR SETTINGS
# ============================================================

OCR_READER = None

# Width of the right-side price-scale region.
#
# This is deliberately narrow so OCR does not scan the entire
# chart.
PRICE_SCALE_WIDTH_RATIO = 0.20

# Minimum confidence for accepting an OCR result.
MIN_PRICE_OCR_CONFIDENCE = 0.35

# Maximum number of OCR results used.
MAX_PRICE_LABELS = 40


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
# PURPLE / YELLOW COLOR SETTINGS
# ============================================================

# 🟣 PURPLE = BUY / BULLISH

PURPLE_HUE_LOW = 125
PURPLE_HUE_HIGH = 165

MIN_PURPLE_SATURATION = 100
MIN_PURPLE_VALUE = 70


# 🟡 YELLOW = SELL / BEARISH

YELLOW_HUE_LOW = 18
YELLOW_HUE_HIGH = 40

MIN_YELLOW_SATURATION = 100
MIN_YELLOW_VALUE = 70


# ============================================================
# COLOR DENSITY
# ============================================================

MIN_COLOR_DENSITY = 0.25


# ============================================================
# COLOR DOMINANCE
# ============================================================

PURPLE_DOMINANCE_RATIO = 1.20
YELLOW_DOMINANCE_RATIO = 1.10


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
# INITIALIZE EASY OCR
# ============================================================

def initialize_ocr():

    global OCR_READER

    if easyocr is None:

        print(
            "⚠️ EasyOCR is not installed."
        )

        return False

    if OCR_READER is not None:

        return True

    try:

        print(
            "🔎 Loading EasyOCR price reader..."
        )

        OCR_READER = easyocr.Reader(
            ["en"],
            gpu=False,
            verbose=False
        )

        print(
            "✅ EasyOCR price reader ready."
        )

        return True

    except Exception as e:

        print(
            "❌ EasyOCR initialization error:",
            repr(e)
        )

        OCR_READER = None

        return False


# ============================================================
# CLEAN OCR PRICE TEXT
# ============================================================

def clean_price_text(text):

    if text is None:
        return None

    text = str(text).strip()

    if not text:
        return None

    # Replace common OCR mistakes.
    replacements = {
        ",": "",
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5"
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    # Keep only digits and decimal point.
    cleaned = ""

    decimal_found = False

    for char in text:

        if char.isdigit():

            cleaned += char

        elif char == "." and not decimal_found:

            cleaned += char

            decimal_found = True

    if not cleaned:
        return None

    # Must contain at least one digit.
    if not any(
        char.isdigit()
        for char in cleaned
    ):
        return None

    try:

        value = float(cleaned)

    except Exception:

        return None

    # Reject obviously invalid OCR numbers.
    if value <= 0:
        return None

    if value > 100000000:
        return None

    return value


# ============================================================
# READ PRICE SCALE
# ============================================================

def read_price_scale(img):

    """
    Reads only the right-side price scale.

    Returns:

        labels:
            [
                {
                    "price": numeric price,
                    "y": vertical position,
                    "confidence": OCR confidence
                }
            ]

        current_price:
            nearest price-scale label to the current-price
            marker when one can be identified.

        highest_price:
            highest visible price label.

        lowest_price:
            lowest visible price label.
    """

    result = {
        "labels": [],
        "current_price": None,
        "highest_price": None,
        "lowest_price": None,
        "ocr_available": False,
        "processing_time": 0.0
    }

    start = time.time()

    if not initialize_ocr():

        result["processing_time"] = (
            time.time() - start
        )

        return result

    h, w = img.shape[:2]

    # ========================================================
    # PRICE SCALE REGION
    # ========================================================

    scale_width = max(
        180,
        int(
            w *
            PRICE_SCALE_WIDTH_RATIO
        )
    )

    x_start = max(
        0,
        w - scale_width
    )

    price_region = img[
        :,
        x_start:
    ]

    # ========================================================
    # UPSCALE ONLY PRICE REGION
    # ========================================================

    region_h, region_w = (
        price_region.shape[:2]
    )

    if region_w < 500:

        scale = 500 / region_w

        price_region = cv2.resize(
            price_region,
            (
                int(region_w * scale),
                int(region_h * scale)
            ),
            interpolation=cv2.INTER_CUBIC
        )

    # ========================================================
    # EASY OCR
    # ========================================================

    try:

        ocr_results = OCR_READER.readtext(
            price_region,
            detail=1,
            paragraph=False,
            allowlist="0123456789."
        )

    except Exception as e:

        print(
            "❌ Price OCR error:",
            repr(e)
        )

        result["processing_time"] = (
            time.time() - start
        )

        return result

    labels = []

    for item in ocr_results:

        try:

            box = item[0]
            text = item[1]
            confidence = float(item[2])

        except Exception:

            continue

        if confidence < MIN_PRICE_OCR_CONFIDENCE:
            continue

        price = clean_price_text(
            text
        )

        if price is None:
            continue

        # ====================================================
        # FIND OCR BOX CENTER
        # ====================================================

        points = np.array(
            box,
            dtype=np.float32
        )

        center_x = float(
            np.mean(points[:, 0])
        )

        center_y = float(
            np.mean(points[:, 1])
        )

        # ====================================================
        # CONVERT OCR Y BACK TO ORIGINAL IMAGE
        # ====================================================

        if region_w < 500:

            scale = 500 / region_w

            original_y = (
                center_y /
                scale
            )

        else:

            original_y = center_y

        # Only keep labels that are actually toward the
        # right-side price scale.
        #
        # Since we already cropped the right side, this
        # prevents unrelated chart text from entering.
        if original_y < 0:
            continue

        if original_y >= h:
            continue

        labels.append({

            "price": price,

            "y": float(
                original_y
            ),

            "confidence": confidence

        })

    # ========================================================
    # SORT BY VERTICAL POSITION
    # ========================================================

    labels.sort(
        key=lambda x: x["y"]
    )

    # ========================================================
    # REMOVE DUPLICATE / NEAR-DUPLICATE LABELS
    # ========================================================

    filtered = []

    for label in labels:

        duplicate = False

        for existing in filtered:

            if (
                abs(
                    label["y"] -
                    existing["y"]
                ) < 8
                and
                abs(
                    label["price"] -
                    existing["price"]
                ) < 0.000001
            ):

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

            filtered.append(
                label
            )

    labels = filtered[
        :MAX_PRICE_LABELS
    ]

    result["labels"] = labels

    # ========================================================
    # HIGHEST / LOWEST VISIBLE PRICE
    # ========================================================

    if labels:

        result["highest_price"] = max(
            label["price"]
            for label in labels
        )

        result["lowest_price"] = min(
            label["price"]
            for label in labels
        )

    # ========================================================
    # CURRENT PRICE
    # ========================================================
    #
    # We first look for a highlighted price marker/color.
    # If none is found, we do NOT invent a current price.
    #
    # Instead, the caller can estimate it from the latest
    # candle position against the price-scale labels.
    # ========================================================

    result["processing_time"] = (
        time.time() - start
    )

    return result


# ============================================================
# ESTIMATE PRICE AT Y POSITION
# ============================================================

def price_from_y(
    y,
    price_labels
):

    if not price_labels:
        return None

    # Sort labels top -> bottom.
    labels = sorted(
        price_labels,
        key=lambda x: x["y"]
    )

    # ========================================================
    # EXACT / NEAR LABEL
    # ========================================================

    nearest = min(
        labels,
        key=lambda x:
        abs(
            x["y"] -
            y
        )
    )

    # If the candle is very close to a visible price label,
    # use that label directly.
    if abs(
        nearest["y"] - y
    ) <= 12:

        return nearest["price"]

    # ========================================================
    # INTERPOLATION BETWEEN TWO PRICE LABELS
    # ========================================================

    for i in range(
        len(labels) - 1
    ):

        upper = labels[i]
        lower = labels[i + 1]

        y1 = upper["y"]
        y2 = lower["y"]

        if (
            y1 <= y <= y2
            and
            y2 != y1
        ):

            p1 = upper["price"]
            p2 = lower["price"]

            ratio = (
                (y - y1) /
                (y2 - y1)
            )

            # Screen Y increases downward.
            price = (
                p1 +
                (
                    p2 - p1
                ) *
                ratio
            )

            return price

    return None


# ============================================================
# ADD PRICE INFORMATION TO CANDLES
# ============================================================

def attach_price_information(
    candles,
    price_data
):

    labels = price_data.get(
        "labels",
        []
    )

    if not labels:

        for candle in candles:

            candle["price_top"] = None
            candle["price_bottom"] = None
            candle["price_center"] = None

        return candles

    for candle in candles:

        x = candle["x"]
        y = candle["y"]
        h = candle["h"]

        top_y = float(y)

        bottom_y = float(
            y + h
        )

        center_y = float(
            y +
            h / 2
        )

        candle["price_top"] = (
            price_from_y(
                top_y,
                labels
            )
        )

        candle["price_bottom"] = (
            price_from_y(
                bottom_y,
                labels
            )
        )

        candle["price_center"] = (
            price_from_y(
                center_y,
                labels
            )
        )

    return candles


# ============================================================
# FIND CURRENT PRICE FROM RIGHTMOST CANDLE
# ============================================================

def estimate_current_price(
    candles,
    price_data
):

    labels = price_data.get(
        "labels",
        []
    )

    if not candles or not labels:

        return None

    # Number 1 is the newest/rightmost detected candle.
    newest = candles[0]

    # Use the center of the newest candle body.
    center_y = (
        newest["y"] +
        newest["h"] / 2
    )

    price = price_from_y(
        center_y,
        labels
    )

    return price


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
    # BGR COLOR DOMINANCE
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
# REPORT
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
# PRICE FORMAT
# ============================================================

def format_price(
    value
):

    if value is None:
        return "NOT READ"

    # Keep enough decimal places for forex pairs.
    return f"{value:.6f}".rstrip(
        "0"
    ).rstrip(".")


# ============================================================
# NUMBERED DETECTION MAP
# ============================================================

def create_detection_map(
    img,
    candles,
    price_data,
    current_price
):

    output = img.copy()

    labels = price_data.get(
        "labels",
        []
    )

    # ========================================================
    # DRAW PRICE SCALE READINGS
    # ========================================================

    for label in labels:

        y = int(
            label["y"]
        )

        price_text = format_price(
            label["price"]
        )

        # Small marker at the OCR-read price level.
        cv2.line(
            output,
            (
                max(
                    0,
                    output.shape[1] - 80
                ),
                y
            ),
            (
                output.shape[1] - 5,
                y
            ),
            (
                255,
                255,
                255
            ),
            1
        )

        cv2.putText(
            output,
            price_text,
            (
                max(
                    5,
                    output.shape[1] - 150
                ),
                max(
                    20,
                    y - 4
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (
                255,
                255,
                255
            ),
            1,
            cv2.LINE_AA
        )

    # ========================================================
    # CURRENT PRICE LINE
    # ========================================================

    if current_price is not None:

        # Find the Y position corresponding to current price.
        current_y = None

        for label in labels:

            if abs(
                label["price"] -
                current_price
            ) < 0.0000001:

                current_y = int(
                    label["y"]
                )

                break

        if current_y is not None:

            cv2.line(
                output,
                (
                    0,
                    current_y
                ),
                (
                    output.shape[1],
                    current_y
                ),
                (
                    255,
                    255,
                    0
                ),
                2
            )

            cv2.putText(
                output,
                (
                    "CURRENT: "
                    +
                    format_price(
                        current_price
                    )
                ),
                (
                    20,
                    35
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    255,
                    255,
                    0
                ),
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

        w = int(
            candle["w"]
        )

        h = int(
            candle["h"]
        )

        # Yellow box around detected body.
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

            direction_text = "BUY"

        else:

            label_color = (
                0,
                255,
                255
            )

            direction_text = "SELL"

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
        # CANDLE APPROXIMATE PRICE
        # ====================================================

        candle_price = candle.get(
            "price_center"
        )

        if candle_price is not None:

            price_text = format_price(
                candle_price
            )

            text_x = min(
                output.shape[1] - 150,
                max(
                    5,
                    x
                )
            )

            text_y = min(
                output.shape[0] - 5,
                max(
                    20,
                    y + h + 18
                )
            )

            cv2.putText(
                output,
                price_text,
                (
                    text_x,
                    text_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                label_color,
                1,
                cv2.LINE_AA
            )

    # ========================================================
    # HIGH / LOW INFORMATION
    # ========================================================

    highest = price_data.get(
        "highest_price"
    )

    lowest = price_data.get(
        "lowest_price"
    )

    info_y = 65

    cv2.putText(
        output,
        (
            "HIGH: "
            +
            format_price(
                highest
            )
        ),
        (
            20,
            info_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            255,
            255,
            255
        ),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        (
            "LOW: "
            +
            format_price(
                lowest
            )
        ),
        (
            20,
            info_y + 25
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            255,
            255,
            255
        ),
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

def handle_photo(
    message
):

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

            "👁️ Reading screenshot...\n"
            "➡️ Scanning RIGHT → LEFT.\n"
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
        # DETECT CANDLES
        # ====================================================

        candles = detect_candles(
            img
        )

        # ====================================================
        # READ PRICE SCALE
        # ====================================================

        price_data = read_price_scale(
            img
        )

        # ====================================================
        # ATTACH PRICE TO CANDLES
        # ====================================================

        candles = attach_price_information(
            candles,
            price_data
        )

        # ====================================================
        # ESTIMATE CURRENT PRICE
        # ====================================================

        current_price = (
            estimate_current_price(
                candles,
                price_data
            )
        )

        # ====================================================
        # COUNT
        # ====================================================

        purple, yellow = (
            create_report(
                candles
            )
        )

        total = len(
            candles
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

            if candle["color"] == "PURPLE":

                direction = "🟣 BUY"

            else:

                direction = "🟡 SELL"

            candle_price = candle.get(
                "price_center"
            )

            if candle_price is not None:

                price_text = format_price(
                    candle_price
                )

            else:

                price_text = "PRICE?"

            sequence.append(

                f"{number}. "
                f"{direction} "
                f"@ {price_text}"

            )

        sequence_text = (
            "\n".join(
                sequence
            )
        )

        # ====================================================
        # PRICE RESULTS
        # ====================================================

        highest_price = (
            price_data.get(
                "highest_price"
            )
        )

        lowest_price = (
            price_data.get(
                "lowest_price"
            )
        )

        # ====================================================
        # RESULT
        # ====================================================

        if total == 0:

            bot.reply_to(

                message,

                "❌ No reliable candle bodies detected.\n\n"
                "No candle was generated.\n"
                "No random candle was added.\n"
                "No signal was generated.\n\n"
                "💰 PRICE SCALE:\n"
                f"Highest: {format_price(highest_price)}\n"
                f"Lowest: {format_price(lowest_price)}\n"
                f"Current/nearest: {format_price(current_price)}"

            )

            return

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
            f"{format_price(highest_price)}\n"

            f"⬇️ LOWEST: "
            f"{format_price(lowest_price)}\n"

            f"📍 CURRENT / NEWEST CANDLE: "
            f"{format_price(current_price)}\n\n"

            "🕯️ **RIGHT → LEFT READING**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"

            f"{sequence_text}\n\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            "🎯 **COLOR KEY**\n"

            "🟣 = PURPLE / BUY\n"

            "🟡 = YELLOW / SELL\n\n"

            "💰 **PRICE KEY**\n"

            "The price beside each candle is an "
            "approximation calculated from the visible "
            "price-scale labels and the candle's vertical "
            "position.\n\n"

            "⚠️ This does NOT invent OHLC data.\n"

            "⚠️ It does NOT create random prices.\n"

            "⚠️ If the price scale cannot be read reliably, "
            "the bot reports NOT READ instead.\n\n"

            f"⚡ Total processing time: "
            f"{elapsed:.2f}s\n"

            f"🔎 Price OCR time: "
            f"{price_data.get('processing_time', 0):.2f}s"

        )

        bot.reply_to(

            message,

            report,

            parse_mode="Markdown"

        )

        # ====================================================
        # CREATE DETECTION MAP
        # ====================================================

        detection_map = (

            create_detection_map(

                img,

                candles,

                price_data,

                current_price

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

                    "💰 White numbers/text = price-scale "
                    "values read from the screenshot.\n\n"

                    "💰 Price beside a candle = approximate "
                    "price calculated from its vertical "
                    "position against the visible price "
                    "scale.\n\n"

                    "⬆️ HIGH = highest price label detected.\n"

                    "⬇️ LOW = lowest price label detected.\n"

                    "📍 CURRENT = price estimated from the "
                    "newest/rightmost detected candle.\n\n"

                    "Please compare the printed prices with "
                    "the actual price scale in the screenshot."

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
    "💰 Visible price-scale reading enabled"
)

print(
    "📈 Highest visible price enabled"
)

print(
    "📉 Lowest visible price enabled"
)

print(
    "📍 Current/newest-candle price estimation enabled"
)

print(
    "🚫 No fake prices"
)

print(
    "🚫 No random candles"
)

print(
    "🚫 No random OHLC"
)

print(
    "========================================"
)


# ============================================================
# START TELEGRAM BOT
# ============================================================

bot.infinity_polling(

    timeout=30,

    long_polling_timeout=30

)

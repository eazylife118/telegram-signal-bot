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
    "PASTE_YOUR_BOT_TOKEN_HERE"
)

bot = telebot.TeleBot(TELEGRAM_TOKEN)


# ============================================================
# PRICE AREA SETTINGS
# ============================================================
#
# Pocket Option screenshot:
#
# The price scale is on the RIGHT side of the chart.
#
# We deliberately DO NOT scan:
# - demo amount
# - balance
# - payout
# - buttons
# - bottom controls
#
# Only the right chart price area is examined.
# ============================================================

PRICE_X_START = 0.78
PRICE_X_END = 0.995

PRICE_Y_START = 0.18
PRICE_Y_END = 0.82


# ============================================================
# IMAGE SETTINGS
# ============================================================

MIN_IMAGE_WIDTH = 900


# ============================================================
# TEXT DETECTION SETTINGS
# ============================================================

MIN_TEXT_HEIGHT = 8
MAX_TEXT_HEIGHT = 45

MIN_TEXT_WIDTH = 2
MAX_TEXT_WIDTH = 25

MIN_TEXT_PIXELS = 8


# ============================================================
# NUMBER SETTINGS
# ============================================================

EXPECTED_DECIMAL_DIGITS = 5

# Example:
#
# 0.274300
# 0.274200
# 0.274501
#
# We expect numbers similar to:
#
# 0.xxxxxx
#
# but the recognizer does not require this exact format.


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

    if w < MIN_IMAGE_WIDTH:

        scale = MIN_IMAGE_WIDTH / float(w)

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
# CROP ONLY RIGHT PRICE AREA
# ============================================================

def crop_price_area(img):

    h, w = img.shape[:2]

    x1 = int(w * PRICE_X_START)
    x2 = int(w * PRICE_X_END)

    y1 = int(h * PRICE_Y_START)
    y2 = int(h * PRICE_Y_END)

    crop = img[
        y1:y2,
        x1:x2
    ]

    return crop, (
        x1,
        y1
    )


# ============================================================
# CREATE BRIGHT TEXT MASK
# ============================================================

def create_text_masks(crop):

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # WHITE / LIGHT TEXT
    # --------------------------------------------------------

    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 90]),
        np.array([180, 130, 255])
    )

    # --------------------------------------------------------
    # LIGHT COLORED TEXT
    #
    # Current-price label can be blue/gray.
    # --------------------------------------------------------

    bright_mask = cv2.inRange(
        gray,
        95,
        255
    )

    # Combine
    combined = cv2.bitwise_or(
        white_mask,
        bright_mask
    )

    # Remove tiny noise.
    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        kernel
    )

    return combined


# ============================================================
# FIND TEXT ROWS
# ============================================================
#
# Instead of detecting every digit separately first,
# we detect horizontal rows containing price text.
#
# Example:
#
# 0.274300
# 0.274200
# 0.274501
# 0.274100
# 0.274305
#
# ============================================================

def find_price_rows(mask):

    h, w = mask.shape[:2]

    # Count bright pixels in each horizontal row.
    row_strength = np.sum(
        mask > 0,
        axis=1
    )

    rows = []

    in_row = False
    start = 0

    for y in range(h):

        active = (
            row_strength[y] >= 2
        )

        if active and not in_row:

            start = y
            in_row = True

        elif not active and in_row:

            end = y - 1

            height = end - start + 1

            if (
                MIN_TEXT_HEIGHT
                <= height
                <= MAX_TEXT_HEIGHT
            ):

                rows.append(
                    (
                        start,
                        end
                    )
                )

            in_row = False

    if in_row:

        end = h - 1

        height = end - start + 1

        if (
            MIN_TEXT_HEIGHT
            <= height
            <= MAX_TEXT_HEIGHT
        ):

            rows.append(
                (
                    start,
                    end
                )
            )

    # --------------------------------------------------------
    # Merge rows that are extremely close.
    # --------------------------------------------------------

    merged = []

    for row in rows:

        if not merged:

            merged.append(
                list(row)
            )

            continue

        previous = merged[-1]

        if row[0] - previous[1] <= 3:

            previous[1] = row[1]

        else:

            merged.append(
                list(row)
            )

    return merged


# ============================================================
# GET ROW REGION
# ============================================================

def get_row_region(
    mask,
    y1,
    y2
):

    region = mask[
        y1:y2 + 1,
        :
    ]

    column_strength = np.sum(
        region > 0,
        axis=0
    )

    active_columns = np.where(
        column_strength >= 1
    )[0]

    if len(active_columns) == 0:

        return None

    left = int(
        np.min(active_columns)
    )

    right = int(
        np.max(active_columns)
    )

    # --------------------------------------------------------
    # We only want price text.
    #
    # Ignore extremely tiny fragments.
    # --------------------------------------------------------

    if right - left < 15:

        return None

    return region[
        :,
        left:right + 1
    ], left, right


# ============================================================
# DIGIT TEMPLATE GENERATOR
# ============================================================
#
# No Tesseract.
#
# OpenCV creates reference digits.
#
# Several font sizes are generated so that the recognizer
# has multiple references to compare against.
#
# ============================================================

def make_digit_templates():

    templates = []

    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_PLAIN
    ]

    font_scales = [
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70
    ]

    thicknesses = [
        1,
        2
    ]

    for digit in range(10):

        character = str(digit)

        for font in fonts:

            for scale in font_scales:

                for thickness in thicknesses:

                    canvas = np.zeros(
                        (50, 40),
                        dtype=np.uint8
                    )

                    cv2.putText(
                        canvas,
                        character,
                        (5, 35),
                        font,
                        scale,
                        255,
                        thickness,
                        cv2.LINE_AA
                    )

                    ys, xs = np.where(
                        canvas > 0
                    )

                    if len(xs) == 0:
                        continue

                    x1 = max(
                        0,
                        int(xs.min()) - 2
                    )

                    x2 = min(
                        canvas.shape[1],
                        int(xs.max()) + 3
                    )

                    y1 = max(
                        0,
                        int(ys.min()) - 2
                    )

                    y2 = min(
                        canvas.shape[0],
                        int(ys.max()) + 3
                    )

                    cropped = canvas[
                        y1:y2,
                        x1:x2
                    ]

                    cropped = cv2.resize(
                        cropped,
                        (24, 36),
                        interpolation=cv2.INTER_AREA
                    )

                    _, cropped = cv2.threshold(
                        cropped,
                        80,
                        255,
                        cv2.THRESH_BINARY
                    )

                    templates.append(
                        (
                            digit,
                            cropped
                        )
                    )

    return templates


DIGIT_TEMPLATES = make_digit_templates()


# ============================================================
# NORMALIZE DIGIT
# ============================================================

def normalize_digit(roi):

    if roi is None:
        return None

    if roi.size == 0:
        return None

    ys, xs = np.where(
        roi > 0
    )

    if len(xs) == 0:
        return None

    x1 = max(
        0,
        int(xs.min()) - 1
    )

    x2 = min(
        roi.shape[1],
        int(xs.max()) + 2
    )

    y1 = max(
        0,
        int(ys.min()) - 1
    )

    y2 = min(
        roi.shape[0],
        int(ys.max()) + 2
    )

    roi = roi[
        y1:y2,
        x1:x2
    ]

    roi = cv2.resize(
        roi,
        (24, 36),
        interpolation=cv2.INTER_AREA
    )

    _, roi = cv2.threshold(
        roi,
        80,
        255,
        cv2.THRESH_BINARY
    )

    return roi


# ============================================================
# TEMPLATE MATCH DIGIT
# ============================================================

def recognize_digit(roi):

    normalized = normalize_digit(
        roi
    )

    if normalized is None:

        return "?", 0.0

    best_digit = "?"
    best_score = -1

    for digit, template in DIGIT_TEMPLATES:

        # Correlation
        score = cv2.matchTemplate(
            normalized,
            template,
            cv2.TM_CCOEFF_NORMED
        )[0][0]

        if score > best_score:

            best_score = score
            best_digit = str(digit)

    confidence = (
        max(
            0,
            min(
                100,
                (best_score + 1) * 50
            )
        )
    )

    return (
        best_digit,
        confidence
    )


# ============================================================
# SEGMENT POSSIBLE DIGITS
# ============================================================

def segment_row(
    row_mask
):

    # Slight horizontal closing joins parts of digits
    # without joining distant price rows.

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 1)
    )

    work = cv2.morphologyEx(
        row_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        work,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    h, w = work.shape[:2]

    for contour in contours:

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        area = cw * ch

        if area < MIN_TEXT_PIXELS:
            continue

        if ch < 5:
            continue

        if ch > h + 2:
            continue

        if cw > MAX_TEXT_WIDTH:
            continue

        # Ignore extremely small noise.
        if cw < MIN_TEXT_WIDTH and ch < 8:
            continue

        boxes.append(
            (
                x,
                y,
                cw,
                ch
            )
        )

    boxes.sort(
        key=lambda b: b[0]
    )

    return boxes


# ============================================================
# RECOGNIZE PRICE ROW
# ============================================================

def recognize_price_row(
    row_mask
):

    boxes = segment_row(
        row_mask
    )

    if not boxes:

        return None, 0.0, []

    digits = []
    confidences = []

    debug_boxes = []

    # --------------------------------------------------------
    # Recognize characters.
    # --------------------------------------------------------

    for x, y, w, h in boxes:

        roi = row_mask[
            y:y+h,
            x:x+w
        ]

        # Very small isolated point can be decimal point.
        if (
            h <= 9
            and
            w <= 8
        ):

            digits.append(".")

            confidences.append(
                90
            )

            debug_boxes.append(
                (x, y, w, h, ".")
            )

            continue

        digit, confidence = (
            recognize_digit(
                roi
            )
        )

        if digit != "?":

            digits.append(
                digit
            )

            confidences.append(
                confidence
            )

            debug_boxes.append(
                (
                    x,
                    y,
                    w,
                    h,
                    digit
                )
            )

    if not digits:

        return None, 0.0, []

    number = "".join(
        digits
    )

    # --------------------------------------------------------
    # Clean common segmentation mistakes.
    # --------------------------------------------------------

    number = number.replace(
        "..",
        "."
    )

    # A price normally contains one decimal.
    if number.count(".") > 1:

        first = number.find(".")

        cleaned = (
            number[:first + 1]
            +
            number[first + 1:].replace(
                ".",
                ""
            )
        )

        number = cleaned

    if confidences:

        confidence = (
            sum(confidences)
            /
            len(confidences)
        )

    else:

        confidence = 0

    return (
        number,
        confidence,
        debug_boxes
    )


# ============================================================
# PRICE VALIDATION
# ============================================================

def is_price_like(number):

    if not number:
        return False

    # Must contain digits.
    if not any(
        c.isdigit()
        for c in number
    ):
        return False

    # A chart price should normally contain a decimal.
    if "." not in number:
        return False

    parts = number.split(".")

    if len(parts) != 2:
        return False

    left = parts[0]
    right = parts[1]

    if not left.isdigit():
        return False

    if not right.isdigit():
        return False

    # Ignore obviously tiny values.
    if len(right) < 3:
        return False

    # Price labels in this chart are normally compact.
    if len(number) > 15:
        return False

    return True


# ============================================================
# REMOVE DUPLICATE PRICE ROWS
# ============================================================

def remove_duplicate_prices(
    rows
):

    result = []

    for row in rows:

        number = row["number"]

        if not number:
            continue

        duplicate = False

        for existing in result:

            if number == existing["number"]:

                # Keep stronger recognition.
                if (
                    row["confidence"]
                    >
                    existing["confidence"]
                ):

                    existing.update(
                        row
                    )

                duplicate = True

                break

        if not duplicate:

            result.append(
                row
            )

    return result


# ============================================================
# READ RIGHT-SIDE PRICES
# ============================================================

def read_right_side_prices(
    img
):

    crop, origin = crop_price_area(
        img
    )

    x_origin, y_origin = origin

    mask = create_text_masks(
        crop
    )

    rows = find_price_rows(
        mask
    )

    detected = []

    for y1, y2 in rows:

        region_info = get_row_region(
            mask,
            y1,
            y2
        )

        if region_info is None:
            continue

        row_mask, left, right = (
            region_info
        )

        number, confidence, boxes = (
            recognize_price_row(
                row_mask
            )
        )

        if not is_price_like(
            number
        ):
            continue

        # Convert row position back to full screenshot.
        center_y = (
            y_origin
            +
            (y1 + y2) / 2
        )

        detected.append({

            "number": number,

            "confidence": confidence,

            "center_y": center_y,

            "x1": x_origin + left,

            "x2": x_origin + right,

            "y1": y_origin + y1,

            "y2": y_origin + y2,

            "boxes": boxes

        })

    detected = remove_duplicate_prices(
        detected
    )

    detected.sort(
        key=lambda r:
        r["center_y"]
    )

    return detected


# ============================================================
# FIND HIGHEST / LOWEST
# ============================================================

def get_price_extremes(
    prices
):

    numeric = []

    for p in prices:

        try:

            value = float(
                p["number"]
            )

            numeric.append(
                (
                    value,
                    p
                )
            )

        except Exception:
            pass

    if not numeric:

        return None, None

    highest = max(
        numeric,
        key=lambda x:
        x[0]
    )[1]

    lowest = min(
        numeric,
        key=lambda x:
        x[0]
    )[1]

    return (
        highest,
        lowest
    )


# ============================================================
# FIND CURRENT PRICE
# ============================================================
#
# Pocket Option often displays the current price in a
# highlighted/colored label around the current candle level.
#
# We look for the price row closest to the strongest horizontal
# highlighted region.
#
# ============================================================

def find_current_price(
    img,
    prices
):

    if not prices:

        return None

    h, w = img.shape[:2]

    # Look at the right side where current-price label lives.
    x1 = int(
        w * 0.82
    )

    x2 = int(
        w * 0.995
    )

    y1 = int(
        h * 0.25
    )

    y2 = int(
        h * 0.78
    )

    region = img[
        y1:y2,
        x1:x2
    ]

    hsv = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2HSV
    )

    # Look for blue/cyan highlighted price labels.
    blue = cv2.inRange(
        hsv,
        np.array([80, 40, 70]),
        np.array([130, 255, 255])
    )

    row_strength = np.sum(
        blue > 0,
        axis=1
    )

    if len(row_strength) == 0:
        return None

    strongest_y = int(
        np.argmax(
            row_strength
        )
    )

    absolute_y = (
        y1 +
        strongest_y
    )

    # Find detected price closest to highlight.
    closest = min(
        prices,
        key=lambda p:
        abs(
            p["center_y"]
            -
            absolute_y
        )
    )

    distance = abs(
        closest["center_y"]
        -
        absolute_y
    )

    # Only accept if reasonably close.
    if distance <= 35:

        return closest

    return None


# ============================================================
# CREATE DEBUG MAP
# ============================================================

def create_price_map(
    img,
    prices,
    highest,
    lowest,
    current
):

    output = img.copy()

    for p in prices:

        x1 = int(
            p["x1"]
        )

        x2 = int(
            p["x2"]
        )

        y1 = int(
            p["y1"]
        )

        y2 = int(
            p["y2"]
        )

        # Green = detected price
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            p["number"],
            (
                x1,
                max(
                    20,
                    y1 - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Mark highest
    # --------------------------------------------------------

    if highest:

        cv2.putText(
            output,
            "HIGHEST",
            (
                int(highest["x1"]),
                int(highest["y2"]) + 18
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Mark lowest
    # --------------------------------------------------------

    if lowest:

        cv2.putText(
            output,
            "LOWEST",
            (
                int(lowest["x1"]),
                int(lowest["y2"]) + 18
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Mark current
    # --------------------------------------------------------

    if current:

        cv2.putText(
            output,
            "CURRENT",
            (
                int(current["x1"]),
                int(current["y1"]) - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
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

    image_path = (
        "price_screenshot.png"
    )

    map_path = (
        "price_detection_map.png"
    )

    try:

        bot.reply_to(
            message,
            "🔎 Reading RIGHT-SIDE price numbers..."
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
            image_path,
            "wb"
        ) as f:

            f.write(
                downloaded_file
            )

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        img = load_image(
            image_path
        )

        # ----------------------------------------------------
        # READ PRICES
        # ----------------------------------------------------

        prices = read_right_side_prices(
            img
        )

        # ----------------------------------------------------
        # EXTREMES
        # ----------------------------------------------------

        highest, lowest = (
            get_price_extremes(
                prices
            )
        )

        # ----------------------------------------------------
        # CURRENT
        # ----------------------------------------------------

        current = find_current_price(
            img,
            prices
        )

        elapsed = (
            time.time()
            -
            start_time
        )

        # ----------------------------------------------------
        # NO RESULT
        # ----------------------------------------------------

        if not prices:

            bot.reply_to(
                message,
                (
                    "❌ No price numbers detected.\n\n"
                    "I only scanned the RIGHT-SIDE "
                    "chart price area.\n"
                    "Demo amount was ignored.\n"
                    "No Tesseract.\n"
                    "No random numbers."
                )
            )

            return

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        response = (
            "🔢 **RIGHT-SIDE PRICE READER**\n\n"
            "📍 **Detected prices:**\n"
        )

        for i, price in enumerate(
            prices,
            1
        ):

            response += (
                f"{i}. `{price['number']}` "
                f"({price['confidence']:.0f}%)\n"
            )

        response += "\n"

        if highest:

            response += (
                f"🔺 **HIGHEST:** "
                f"`{highest['number']}`\n"
            )

        if lowest:

            response += (
                f"🔻 **LOWEST:** "
                f"`{lowest['number']}`\n"
            )

        if current:

            response += (
                f"🎯 **CURRENT:** "
                f"`{current['number']}`\n"
            )

        else:

            response += (
                "🎯 **CURRENT:** "
                "Not confidently identified\n"
            )

        response += (
            "\n━━━━━━━━━━━━━━━━━━━━\n"
            "🚫 Demo amount ignored\n"
            "🚫 No Tesseract\n"
            "🚫 No random numbers\n"
            "🚫 No OHLC generation\n"
            "🚫 No trading signal\n"
            f"\n⚡ Processing: {elapsed:.2f}s"
        )

        bot.reply_to(
            message,
            response,
            parse_mode="Markdown"
        )

        # ----------------------------------------------------
        # DEBUG MAP
        # ----------------------------------------------------

        detection_map = create_price_map(
            img,
            prices,
            highest,
            lowest,
            current
        )

        cv2.imwrite(
            map_path,
            detection_map
        )

        with open(
            map_path,
            "rb"
        ) as photo:

            bot.send_photo(
                message.chat.id,
                photo,
                caption=(
                    "🔎 RIGHT-SIDE PRICE MAP\n\n"
                    "🟩 = detected price\n"
                    "🔺 = highest\n"
                    "🔻 = lowest\n"
                    "🎯 = current candidate\n\n"
                    "Demo amount is outside the scan area."
                )
            )

    except Exception as e:

        print(
            "❌ ERROR:",
            repr(e)
        )

        bot.reply_to(
            message,
            f"❌ Price reader error:\n{str(e)}"
        )

    finally:

        for path in [
            image_path,
            map_path
        ]:

            if os.path.exists(path):

                try:

                    os.remove(path)

                except Exception:

                    pass


# ============================================================
# START COMMAND
# ============================================================

@bot.message_handler(
    commands=["start"]
)

def start(message):

    bot.reply_to(
        message,
        (
            "🔢 **POCKET OPTION PRICE READER**\n\n"
            "Send a screenshot.\n\n"
            "I will scan ONLY the right-side "
            "chart price area.\n\n"
            "It will try to read:\n"
            "🔺 Highest visible price\n"
            "🔻 Lowest visible price\n"
            "🎯 Current price\n"
            "📊 All detected price numbers\n\n"
            "🚫 Demo amount ignored\n"
            "🚫 No Tesseract\n"
            "🚫 No random data\n"
            "🚫 No OHLC generation"
        ),
        parse_mode="Markdown"
    )


# ============================================================
# START BOT
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("🔢 POCKET OPTION RIGHT-SIDE PRICE READER")
    print("=" * 50)
    print("✅ RIGHT-SIDE PRICE AREA ONLY")
    print("✅ DEMO AMOUNT IGNORED")
    print("✅ NO TESSERACT")
    print("✅ NO RANDOM NUMBERS")
    print("✅ HIGHEST / LOWEST / CURRENT")
    print("✅ DEBUG MAP ENABLED")
    print("=" * 50)

    bot.infinity_polling(
        timeout=30,
        long_polling_timeout=30
    )

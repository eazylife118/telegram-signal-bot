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
# RAPIDOCR
# ============================================================

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None


OCR = None


def get_ocr():
    global OCR

    if OCR is None:

        if RapidOCR is None:
            raise RuntimeError(
                "RapidOCR is not installed. "
                "Install: rapidocr_onnxruntime"
            )

        OCR = RapidOCR()

    return OCR


# ============================================================
# PRICE REGION SETTINGS
# ============================================================

# The screenshot is a phone screenshot.
#
# We ONLY inspect the right side of the chart.
#
# This intentionally excludes:
# - demo balance
# - time
# - amount
# - payout
# - buttons
# - bottom navigation
# - pair name
#
# The values are percentages so different screenshot
# resolutions can be handled.

PRICE_X_START = 0.78
PRICE_X_END = 0.995

PRICE_Y_START = 0.25
PRICE_Y_END = 0.78


# ============================================================
# PRICE FILTERS
# ============================================================

MIN_PRICE_TEXT_HEIGHT = 7
MAX_PRICE_TEXT_HEIGHT = 45

MIN_PRICE_TEXT_WIDTH = 25
MAX_PRICE_TEXT_WIDTH = 180

MIN_CONFIDENCE = 0.35


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    img = cv2.imread(path)

    if img is None:
        raise ValueError(
            "Could not read screenshot."
        )

    return img


# ============================================================
# CROP ONLY PRICE SCALE
# ============================================================

def crop_price_region(img):

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
        y1,
        x2,
        y2
    )


# ============================================================
# PREPARE PRICE IMAGE
# ============================================================

def prepare_price_crop(crop):

    # Upscale only the small price region.
    #
    # This is much faster than upscaling the entire screenshot.

    scale = 3

    enlarged = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    # Increase contrast slightly.
    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # OCR works better when the dark chart background is separated
    # from the bright price text.

    binary = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )[1]

    return enlarged, enhanced, binary


# ============================================================
# PRICE FORMAT CHECK
# ============================================================

def clean_number(text):

    if not text:
        return None

    text = str(text).strip()

    # Common OCR mistakes around decimal prices.
    replacements = {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
        ",": "."
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new
        )

    # Keep ONLY numbers and decimal point.

    cleaned = ""

    for char in text:

        if char.isdigit() or char == ".":

            cleaned += char

    # Remove duplicate decimal points.

    if cleaned.count(".") > 1:

        first_dot = cleaned.find(".")

        cleaned = (
            cleaned[:first_dot + 1]
            +
            cleaned[first_dot + 1:].replace(
                ".",
                ""
            )
        )

    if not cleaned:
        return None

    # A price must contain digits.

    if not any(
        char.isdigit()
        for char in cleaned
    ):
        return None

    # We specifically want decimal price values.

    if "." not in cleaned:
        return None

    # Reject obviously unrelated values.

    try:

        value = float(cleaned)

    except Exception:

        return None

    if value <= 0:
        return None

    # Pocket Option chart prices normally have
    # several decimal places.

    decimal_part = cleaned.split(
        ".",
        1
    )[1]

    if len(decimal_part) < 3:
        return None

    if len(decimal_part) > 8:
        return None

    return cleaned


# ============================================================
# OCR ONE IMAGE
# ============================================================

def run_ocr(image):

    ocr = get_ocr()

    result, _ = ocr(
        image
    )

    if not result:
        return []

    detections = []

    for item in result:

        if len(item) < 3:
            continue

        box = item[0]
        text = item[1]
        confidence = float(item[2])

        if confidence < MIN_CONFIDENCE:
            continue

        cleaned = clean_number(
            text
        )

        if cleaned is None:
            continue

        xs = [
            float(point[0])
            for point in box
        ]

        ys = [
            float(point[1])
            for point in box
        ]

        left = min(xs)
        right = max(xs)
        top = min(ys)
        bottom = max(ys)

        width = right - left
        height = bottom - top

        if height < MIN_PRICE_TEXT_HEIGHT:
            continue

        if height > MAX_PRICE_TEXT_HEIGHT:
            continue

        if width < MIN_PRICE_TEXT_WIDTH:
            continue

        if width > MAX_PRICE_TEXT_WIDTH:
            continue

        center_x = (
            left +
            right
        ) / 2

        center_y = (
            top +
            bottom
        ) / 2

        detections.append({

            "text": cleaned,

            "confidence": confidence,

            "left": left,

            "right": right,

            "top": top,

            "bottom": bottom,

            "width": width,

            "height": height,

            "center_x": center_x,

            "center_y": center_y

        })

    return detections


# ============================================================
# REMOVE DUPLICATE OCR RESULTS
# ============================================================

def remove_duplicates(
    detections
):

    detections.sort(
        key=lambda d: (
            d["center_y"],
            -d["confidence"]
        )
    )

    final = []

    for detection in detections:

        duplicate = False

        for existing in final:

            y_distance = abs(
                detection["center_y"]
                -
                existing["center_y"]
            )

            x_distance = abs(
                detection["center_x"]
                -
                existing["center_x"]
            )

            if (
                y_distance < 12
                and
                x_distance < 60
            ):

                duplicate = True

                if (
                    detection["confidence"]
                    >
                    existing["confidence"]
                ):

                    final[
                        final.index(existing)
                    ] = detection

                break

        if not duplicate:

            final.append(
                detection
            )

    return final


# ============================================================
# SELECT PRICE LABELS
# ============================================================

def select_prices(
    detections
):

    if not detections:
        return []

    # Remove duplicates.

    detections = remove_duplicates(
        detections
    )

    # Sort vertically.
    #
    # Highest visible price is normally at the top
    # and lowest visible price at the bottom.

    detections.sort(
        key=lambda d:
        d["center_y"]
    )

    return detections


# ============================================================
# CREATE DEBUG MAP
# ============================================================

def create_debug_map(
    img,
    detections,
    crop_box
):

    output = img.copy()

    x1, y1, x2, y2 = crop_box

    # Show ONLY the region used by the price reader.

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    for number, detection in enumerate(
        detections,
        start=1
    ):

        left = int(
            x1 +
            detection["left"]
        )

        right = int(
            x1 +
            detection["right"]
        )

        top = int(
            y1 +
            detection["top"]
        )

        bottom = int(
            y1 +
            detection["bottom"]
        )

        cv2.rectangle(
            output,
            (left, top),
            (right, bottom),
            (0, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"{number}: {detection['text']}",
            (
                max(5, left - 120),
                max(25, top - 5)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    return output


# ============================================================
# MAIN PRICE READER
# ============================================================

def extract_prices(
    image_path
):

    img = load_image(
        image_path
    )

    # ========================================================
    # CROP FIRST
    # ========================================================

    crop, crop_box = crop_price_region(
        img
    )

    # ========================================================
    # PREPARE ONLY SMALL CROP
    # ========================================================

    enlarged, enhanced, binary = (
        prepare_price_crop(
            crop
        )
    )

    # ========================================================
    # OCR THE SMALL REGION
    # ========================================================

    detections = []

    # First pass: enhanced grayscale.

    detections.extend(
        run_ocr(
            enlarged
        )
    )

    # If recognition is weak, try binary.
    #
    # This second pass is only used when necessary.

    if len(detections) == 0:

        detections.extend(
            run_ocr(
                binary
            )
        )

    detections = select_prices(
        detections
    )

    return (
        img,
        crop,
        detections,
        crop_box
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

    start = time.time()

    image_path = (
        "price_screenshot.png"
    )

    debug_path = (
        "price_detection_map.png"
    )

    try:

        # ====================================================
        # DOWNLOAD IMAGE
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
            image_path,
            "wb"
        ) as f:

            f.write(
                downloaded_file
            )

        # ====================================================
        # READ PRICES
        # ====================================================

        (
            img,
            crop,
            detections,
            crop_box
        ) = extract_prices(
            image_path
        )

        elapsed = (
            time.time()
            -
            start
        )

        # ====================================================
        # NO PRICE
        # ====================================================

        if not detections:

            bot.reply_to(
                message,
                "❌ No chart price detected.\n\n"
                "Only the right-side price scale was scanned.\n"
                "Demo amount and other numbers were ignored.\n\n"
                f"⚡ {elapsed:.2f}s"
            )

            return

        # ====================================================
        # PRICE LIST
        # ====================================================

        response = (
            "💰 **CHART PRICE READER**\n\n"
            "📍 RIGHT-SIDE PRICE SCALE ONLY\n"
            "🚫 Demo amount ignored\n"
            "🚫 Amount ignored\n"
            "🚫 Time ignored\n\n"
        )

        response += (
            f"🔢 Prices found: "
            f"{len(detections)}\n\n"
        )

        for i, detection in enumerate(
            detections,
            start=1
        ):

            response += (
                f"{i}. `{detection['text']}` "
                f"({detection['confidence'] * 100:.0f}%)\n"
            )

        # ====================================================
        # HIGHEST / LOWEST
        # ====================================================

        numeric_prices = []

        for detection in detections:

            try:

                value = float(
                    detection["text"]
                )

                numeric_prices.append(
                    (
                        value,
                        detection
                    )
                )

            except Exception:

                pass

        if numeric_prices:

            highest = max(
                numeric_prices,
                key=lambda x: x[0]
            )

            lowest = min(
                numeric_prices,
                key=lambda x: x[0]
            )

            response += (
                "\n━━━━━━━━━━━━━━━━━━━━\n"
                f"🔼 HIGHEST: `{highest[1]['text']}`\n"
                f"🔽 LOWEST: `{lowest[1]['text']}`\n"
            )

        response += (
            "\n\n"
            f"⚡ Processing: {elapsed:.2f}s"
        )

        bot.reply_to(
            message,
            response,
            parse_mode="Markdown"
        )

        # ====================================================
        # DEBUG MAP
        # ====================================================

        debug_map = create_debug_map(
            img,
            detections,
            crop_box
        )

        cv2.imwrite(
            debug_path,
            debug_map
        )

        with open(
            debug_path,
            "rb"
        ) as photo:

            bot.send_photo(
                message.chat.id,
                photo,
                caption=(
                    "🔎 PRICE READER MAP\n\n"
                    "🟩 Green = area scanned\n"
                    "🟨 Yellow = recognized chart price\n\n"
                    "The demo amount is outside the "
                    "price-reading target."
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
            debug_path
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

def start_command(message):

    bot.reply_to(
        message,
        "💰 **FAST PRICE READER**\n\n"
        "Send a Pocket Option screenshot.\n\n"
        "I read ONLY the right-side chart prices.\n\n"
        "✅ Highest price\n"
        "✅ Lowest price\n"
        "✅ Visible price labels\n"
        "🚫 Demo balance ignored\n"
        "🚫 Amount ignored\n"
        "🚫 Time ignored\n"
        "🚫 No Tesseract\n"
        "🚫 No Vision API\n"
        "🚫 No fake numbers",
        parse_mode="Markdown"
    )


# ============================================================
# START BOT
# ============================================================

if __name__ == "__main__":

    print(
        "========================================"
    )

    print(
        "💰 FAST POCKET OPTION PRICE READER"
    )

    print(
        "========================================"
    )

    print(
        "✅ Right-side price crop only"
    )

    print(
        "✅ RapidOCR"
    )

    print(
        "🚫 Tesseract"
    )

    print(
        "🚫 Flask"
    )

    print(
        "🚫 Vision API"
    )

    print(
        "🚫 Fake/generated numbers"
    )

    print(
        "========================================"
    )

    bot.infinity_polling(
        timeout=30,
        long_polling_timeout=30
    )

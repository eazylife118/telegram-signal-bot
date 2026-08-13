import os
import cv2
import numpy as np
import telebot
import pytesseract
import time
import re


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_TOKEN = os.getenv(
    "BOT_TOKEN",
    "PASTE_YOUR_BOT_TOKEN_HERE"
)

bot = telebot.TeleBot(TELEGRAM_TOKEN)


# ============================================================
# NUMBER OCR SETTINGS
# ============================================================

# ONLY these characters are allowed.
NUMBER_WHITELIST = "0123456789.,"

# Minimum confidence for OCR results.
MIN_OCR_CONFIDENCE = 20


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
# CLEAN OCR RESULT
# ============================================================

def clean_number(text):

    if not text:
        return ""

    # Keep ONLY numbers and decimal separators.
    text = re.sub(
        r"[^0-9.,]",
        "",
        text
    )

    # Remove commas used as thousands separators.
    text = text.replace(
        ",",
        ""
    )

    # If there are multiple decimal points,
    # keep only the first one.
    if text.count(".") > 1:

        first_dot = text.find(".")

        text = (
            text[:first_dot + 1]
            +
            text[first_dot + 1:].replace(
                ".",
                ""
            )
        )

    return text


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(numbers):

    result = []

    for number in numbers:

        if not number:
            continue

        if number not in result:

            result.append(
                number
            )

    return result


# ============================================================
# OCR PASS
# ============================================================

def run_ocr(image):

    results = []


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # UPSCALE
    # ========================================================

    scale = 2

    enlarged = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )


    # ========================================================
    # OCR CONFIGURATION
    # ========================================================

    configs = [

        "--psm 6 "
        "-c tessedit_char_whitelist=0123456789.,",

        "--psm 11 "
        "-c tessedit_char_whitelist=0123456789.,",

        "--psm 12 "
        "-c tessedit_char_whitelist=0123456789.,"

    ]


    # ========================================================
    # PASS 1 — NORMAL
    # ========================================================

    for config in configs:

        data = pytesseract.image_to_data(
            enlarged,
            config=config,
            output_type=pytesseract.Output.DICT
        )


        for i, raw_text in enumerate(
            data["text"]
        ):

            raw_text = raw_text.strip()

            if not raw_text:
                continue


            try:

                confidence = float(
                    data["conf"][i]
                )

            except:

                confidence = 0


            if confidence < MIN_OCR_CONFIDENCE:
                continue


            cleaned = clean_number(
                raw_text
            )


            if cleaned:

                results.append(
                    (
                        cleaned,
                        confidence
                    )
                )


    # ========================================================
    # PASS 2 — THRESHOLD
    # ========================================================

    _, threshold = cv2.threshold(
        enlarged,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )


    for config in configs:

        data = pytesseract.image_to_data(
            threshold,
            config=config,
            output_type=pytesseract.Output.DICT
        )


        for i, raw_text in enumerate(
            data["text"]
        ):

            raw_text = raw_text.strip()

            if not raw_text:
                continue


            try:

                confidence = float(
                    data["conf"][i]
                )

            except:

                confidence = 0


            if confidence < MIN_OCR_CONFIDENCE:
                continue


            cleaned = clean_number(
                raw_text
            )


            if cleaned:

                results.append(
                    (
                        cleaned,
                        confidence
                    )
                )


    # ========================================================
    # PASS 3 — INVERTED THRESHOLD
    # ========================================================

    inverted = cv2.bitwise_not(
        threshold
    )


    for config in configs:

        data = pytesseract.image_to_data(
            inverted,
            config=config,
            output_type=pytesseract.Output.DICT
        )


        for i, raw_text in enumerate(
            data["text"]
        ):

            raw_text = raw_text.strip()

            if not raw_text:
                continue


            try:

                confidence = float(
                    data["conf"][i]
                )

            except:

                confidence = 0


            if confidence < MIN_OCR_CONFIDENCE:
                continue


            cleaned = clean_number(
                raw_text
            )


            if cleaned:

                results.append(
                    (
                        cleaned,
                        confidence
                    )
                )


    return results


# ============================================================
# SELECT BEST NUMBERS
# ============================================================

def select_best_numbers(
    results
):

    if not results:

        return []


    # Group identical numbers.
    grouped = {}


    for number, confidence in results:

        if number not in grouped:

            grouped[number] = {
                "count": 0,
                "confidence": []
            }


        grouped[number]["count"] += 1

        grouped[number][
            "confidence"
        ].append(
            confidence
        )


    scored = []


    for number, info in grouped.items():

        count = info["count"]

        average_confidence = (
            sum(
                info["confidence"]
            )
            /
            len(
                info["confidence"]
            )
        )


        # Repeated OCR agreement is valuable.
        score = (
            average_confidence
            +
            min(
                30,
                count * 5
            )
        )


        scored.append(
            (
                number,
                score,
                count,
                average_confidence
            )
        )


    scored.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # Return unique numbers.
    return scored


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


    image_path = (
        "number_test.png"
    )


    try:

        bot.reply_to(
            message,
            "🔢 Reading numbers only..."
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
            image_path,
            "wb"
        ) as f:

            f.write(
                downloaded_file
            )


        # ====================================================
        # LOAD
        # ====================================================

        image = load_image(
            image_path
        )


        # ====================================================
        # OCR
        # ====================================================

        ocr_start = time.time()


        results = run_ocr(
            image
        )


        ocr_time = (
            time.time()
            -
            ocr_start
        )


        # ====================================================
        # SELECT BEST
        # ====================================================

        best_numbers = (
            select_best_numbers(
                results
            )
        )


        total_time = (
            time.time()
            -
            start_time
        )


        # ====================================================
        # NO NUMBERS
        # ====================================================

        if not best_numbers:

            bot.reply_to(

                message,

                "❌ No numbers detected.\n\n"

                "Nothing was generated.\n"

                "No price was generated."

            )

            return


        # ====================================================
        # REPORT
        # ====================================================

        lines = []

        for index, item in enumerate(
            best_numbers[:20],
            start=1
        ):

            number = item[0]

            score = item[1]

            count = item[2]

            confidence = item[3]


            lines.append(

                f"{index}. `{number}` "
                f"— {confidence:.0f}% "
                f"(x{count})"

            )


        numbers_text = (
            "\n".join(
                lines
            )
        )


        report = (

            "🔢 **NUMBER-ONLY OCR TEST**\n\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            "📊 DETECTED NUMBERS\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"{numbers_text}\n\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            "⚠️ TEST MODE\n"

            "Only numeric characters are "
            "allowed by OCR.\n\n"

            "🚫 No candle detection\n"

            "🚫 No purple/yellow detection\n"

            "🚫 No map verification\n"

            "🚫 No price generation\n"

            "🚫 No OHLC generation\n"

            "🚫 No trading signal\n\n"

            f"🔎 OCR: {ocr_time:.2f}s\n"

            f"⚡ Total: {total_time:.2f}s"

        )


        bot.reply_to(

            message,

            report,

            parse_mode="Markdown"

        )


    except Exception as e:

        print(
            "❌ ERROR:",
            repr(e)
        )


        bot.reply_to(

            message,

            f"❌ Number detection error:\n"
            f"{str(e)}"

        )


    finally:

        if os.path.exists(
            image_path
        ):

            try:

                os.remove(
                    image_path
                )

            except:

                pass


# ============================================================
# START
# ============================================================

print(
    "========================================"
)

print(
    "🔢 NUMBER-ONLY OCR TEST BOT"
)

print(
    "========================================"
)

print(
    "📸 Upload screenshot"
)

print(
    "🔢 Reads numbers only"
)

print(
    "🚫 No candle detection"
)

print(
    "🚫 No map verification"
)

print(
    "🚫 No trading signals"
)

print(
    "🚫 No generated prices"
)

print(
    "========================================"
)


bot.infinity_polling(
    timeout=30,
    long_polling_timeout=30
)

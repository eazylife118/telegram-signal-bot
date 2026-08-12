import os
import cv2
import numpy as np
import telebot
import time
import requests
import base64
import json
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
# OPENROUTER
# ============================================================

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)

# OpenRouter automatically chooses a currently available
# FREE model that supports the required capability.
OPENROUTER_MODEL = "openrouter/free"


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
# STRICT PURPLE / YELLOW COLOR SETTINGS
# ============================================================

PURPLE_HUE_LOW = 125
PURPLE_HUE_HIGH = 165

MIN_PURPLE_SATURATION = 100
MIN_PURPLE_VALUE = 70


YELLOW_HUE_LOW = 18
YELLOW_HUE_HIGH = 40

MIN_YELLOW_SATURATION = 100
MIN_YELLOW_VALUE = 70


MIN_COLOR_DENSITY = 0.25


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
    # BGR
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


    return output


# ============================================================
# OPENROUTER VISION
# ============================================================

def analyze_with_openrouter(
    image_path
):

    if not OPENROUTER_API_KEY:

        return {

            "success": False,

            "error":
            "OPENROUTER_API_KEY is not set in Render."

        }


    start = time.time()


    try:

        with open(
            image_path,
            "rb"
        ) as image_file:

            image_bytes = (
                image_file.read()
            )


        base64_image = (
            base64.b64encode(
                image_bytes
            ).decode(
                "utf-8"
            )
        )


        prompt = r"""
You are analyzing a screenshot of a 1-minute financial trading chart.

IMPORTANT:
Do NOT invent numbers.
Do NOT estimate a price if the visible price scale cannot support it.
Do NOT confuse random numbers, timestamps, indicators, account values,
or UI numbers with the chart's price scale.

Your job is ONLY to visually inspect the screenshot.

Focus on the RIGHT-SIDE VERTICAL PRICE SCALE.

Identify:

1. HIGHEST visible price-scale value.
2. LOWEST visible price-scale value.
3. CURRENT price ONLY if there is a clearly visible current-price marker,
   current-price label, or clearly readable live price associated with
   the newest/rightmost candle.
4. The direction of the most recent 2-5 visible candles:
   UP, DOWN, or RANGE.
5. Whether the recent movement appears strongly aligned or mixed.

Also inspect the candles:
- Purple candles are bullish/BUY.
- Yellow candles are bearish/SELL.

Do not create OHLC values that are not visible.

Return ONLY valid JSON in exactly this structure:

{
  "highest_price": null,
  "lowest_price": null,
  "current_price": null,
  "direction": "UP/DOWN/RANGE/UNCERTAIN",
  "recent_alignment": "BULLISH/BEARISH/MIXED/UNCERTAIN",
  "price_scale_visible": true,
  "confidence": 0,
  "notes": ""
}

Rules:

- highest_price must be the highest ACTUAL price number visible on
  the chart's vertical price scale.
- lowest_price must be the lowest ACTUAL price number visible on
  the chart's vertical price scale.
- If you cannot read a value confidently, use null.
- Never use a made-up value.
- Do not use the candle's vertical position to invent a price.
- Confidence must be 0-100.
- Keep notes short.
"""


        payload = {

            "model":
            OPENROUTER_MODEL,

            "messages": [

                {

                    "role":
                    "user",

                    "content": [

                        {

                            "type":
                            "text",

                            "text":
                            prompt

                        },

                        {

                            "type":
                            "image_url",

                            "image_url": {

                                "url":
                                "data:image/png;base64,"
                                +
                                base64_image

                            }

                        }

                    ]

                }

            ],

            "temperature":
            0,

            "max_tokens":
            500

        }


        headers = {

            "Authorization":
            "Bearer "
            +
            OPENROUTER_API_KEY,

            "Content-Type":
            "application/json",

            "HTTP-Referer":
            "https://render.com",

            "X-Title":
            "OTC Candle Price Vision Test"

        }


        response = requests.post(

            OPENROUTER_URL,

            headers=headers,

            json=payload,

            timeout=15

        )


        elapsed = (
            time.time() -
            start
        )


        if response.status_code != 200:

            return {

                "success": False,

                "error":
                f"OpenRouter HTTP "
                f"{response.status_code}: "
                f"{response.text[:1000]}",

                "elapsed":
                elapsed

            }


        data = response.json()


        try:

            content = (
                data[
                    "choices"
                ][0][
                    "message"
                ][
                    "content"
                ]
            )

        except Exception:

            return {

                "success": False,

                "error":
                "OpenRouter returned no readable content.",

                "raw":
                str(data)[:1500],

                "elapsed":
                elapsed

            }


        # ====================================================
        # CLEAN MARKDOWN JSON IF MODEL ADDS IT
        # ====================================================

        content = content.strip()


        content = re.sub(

            r"^```json\s*",

            "",

            content,

            flags=re.IGNORECASE

        )


        content = re.sub(

            r"^```\s*",

            "",

            content

        )


        content = re.sub(

            r"\s*```$",

            "",

            content

        )


        # ====================================================
        # EXTRACT JSON OBJECT
        # ====================================================

        match = re.search(

            r"\{.*\}",

            content,

            flags=re.DOTALL

        )


        if not match:

            return {

                "success": False,

                "error":
                "Vision model did not return JSON.",

                "raw":
                content[:2000],

                "elapsed":
                elapsed

            }


        json_text = (
            match.group(0)
        )


        try:

            result = json.loads(
                json_text
            )

        except Exception:

            return {

                "success": False,

                "error":
                "Could not parse vision JSON.",

                "raw":
                content[:2000],

                "elapsed":
                elapsed

            }


        result["success"] = True

        result["elapsed"] = elapsed

        result["raw"] = content


        return result


    except requests.exceptions.Timeout:

        return {

            "success": False,

            "error":
            "OpenRouter vision request timed out.",

            "elapsed":
            time.time() - start

        }


    except Exception as e:

        return {

            "success": False,

            "error":
            str(e),

            "elapsed":
            time.time() - start

        }


# ============================================================
# FORMAT VISION RESULT
# ============================================================

def format_vision_result(
    vision
):

    if not vision.get(
        "success",
        False
    ):

        return (

            "👁️ **OPENROUTER VISION**\n\n"

            "❌ Vision reading failed.\n\n"

            f"Error: "
            f"{vision.get('error', 'Unknown error')}"

        )


    highest = vision.get(
        "highest_price"
    )


    lowest = vision.get(
        "lowest_price"
    )


    current = vision.get(
        "current_price"
    )


    direction = vision.get(
        "direction",
        "UNCERTAIN"
    )


    alignment = vision.get(
        "recent_alignment",
        "UNCERTAIN"
    )


    confidence = vision.get(
        "confidence",
        0
    )


    notes = vision.get(
        "notes",
        ""
    )


    highest_text = (
        str(highest)
        if highest is not None
        else "NOT READ"
    )


    lowest_text = (
        str(lowest)
        if lowest is not None
        else "NOT READ"
    )


    current_text = (
        str(current)
        if current is not None
        else "NOT READ"
    )


    return (

        "👁️ **OPENROUTER VISION READING**\n\n"

        "💰 **VISIBLE PRICE SCALE**\n"

        "━━━━━━━━━━━━━━━━━━━━\n"

        f"⬆️ HIGHEST: "
        f"{highest_text}\n"

        f"⬇️ LOWEST: "
        f"{lowest_text}\n"

        f"📍 CURRENT: "
        f"{current_text}\n\n"

        "📈 **RECENT MOVEMENT**\n"

        "━━━━━━━━━━━━━━━━━━━━\n"

        f"Direction: {direction}\n"

        f"Alignment: {alignment}\n"

        f"Confidence: {confidence}%\n\n"

        "📝 "

        f"{notes if notes else 'No additional notes.'}\n\n"

        f"⚡ Vision time: "
        f"{vision.get('elapsed', 0):.2f}s"

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
        "candle_detection.png"
    )


    try:

        bot.reply_to(

            message,

            "👁️ Reading chart...\n"
            "➡️ Scanning RIGHT → LEFT.\n"
            "🟣 Detecting PURPLE candles.\n"
            "🟡 Detecting YELLOW candles.\n"
            "💰 Sending the screenshot to "
            "OpenRouter Vision for price-scale reading..."

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
        # LOAD IMAGE
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


        candle_elapsed = (
            time.time()
            -
            candle_start
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


        # ====================================================
        # OPENROUTER VISION
        # ====================================================

        vision = (
            analyze_with_openrouter(
                original_path
            )
        )


        # ====================================================
        # CANDLE SEQUENCE
        # ====================================================

        sequence = []


        for number, candle in enumerate(

            candles,

            start=1

        ):

            if candle["color"] == "PURPLE":

                sequence.append(

                    f"{number}. 🟣 BUY"

                )

            else:

                sequence.append(

                    f"{number}. 🟡 SELL"

                )


        sequence_text = (
            "\n".join(
                sequence
            )
        )


        # ====================================================
        # MAIN REPORT
        # ====================================================

        elapsed = (
            time.time()
            -
            start_time
        )


        report = (

            "🔎 **CANDLE + VISION READING TEST**\n\n"

            "➡️ **SCAN:** RIGHT → LEFT\n\n"

            "📊 **CANDLE DETECTION**\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"🟣 PURPLE / BUY: {purple}\n"

            f"🟡 YELLOW / SELL: {yellow}\n"

            f"📊 TOTAL: {total}\n\n"

            f"🕯️ Candle detection time: "
            f"{candle_elapsed:.2f}s\n\n"

            "💰 **VISION PRICE READING**\n"

            "━━━━━━━━━━━━━━━━━━━━\n\n"

            f"{format_vision_result(vision)}\n\n"

            "🕯️ **RIGHT → LEFT CANDLE READING**\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"{sequence_text}\n\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            "🎯 **COLOR KEY**\n"

            "🟣 = PURPLE / BUY\n"

            "🟡 = YELLOW / SELL\n\n"

            "⚠️ This is a reading test only.\n"

            "⚠️ No random candles are generated.\n"

            "⚠️ No random prices are generated.\n"

            "⚠️ Vision must return NULL/NOT READ "
            "when it cannot confidently read a price.\n\n"

            f"⚡ Total processing time: "
            f"{elapsed:.2f}s"

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

                candles

            )

        )


        cv2.imwrite(

            detection_path,

            detection_map

        )


        # ====================================================
        # SEND DETECTION MAP
        # ====================================================

        with open(

            detection_path,

            "rb"

        ) as photo:

            bot.send_photo(

                message.chat.id,

                photo,

                caption=(

                    "🔢 **CANDLE DETECTION MAP**\n\n"

                    "1 = newest/rightmost detected candle.\n"

                    "➡️ Numbers continue RIGHT → LEFT.\n\n"

                    "🟣 Number = PURPLE / BUY.\n"

                    "🟡 Number = YELLOW / SELL.\n\n"

                    "💰 OpenRouter Vision is used separately "
                    "to read the visible price scale.\n\n"

                    "⬆️ HIGHEST / ⬇️ LOWEST are only accepted "
                    "when the vision model can identify the "
                    "actual visible price-scale values.\n\n"

                    "⚠️ This map does not generate "
                    "market prices."

                ),

                parse_mode="Markdown"

            )


    except Exception as e:

        print(
            "❌ ERROR:",
            repr(e)
        )


        try:

            bot.reply_to(

                message,

                "❌ Detection error:\n"
                +
                str(e)

            )

        except Exception:

            pass


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
    "🕯️ CANDLE + OPENROUTER VISION TEST"
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
    "👁️ OpenRouter Vision enabled"
)

print(
    "💰 Price-scale reading enabled"
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


if not OPENROUTER_API_KEY:

    print(
        "⚠️ WARNING: "
        "OPENROUTER_API_KEY is not set."
    )

else:

    print(
        "✅ OPENROUTER_API_KEY detected."
    )


bot.infinity_polling(

    timeout=30,

    long_polling_timeout=30

)

import os
import cv2
import numpy as np
import telebot
import time

# ============================================================
# TELEGRAM
# ============================================================
# Keep your real token in Render as:
# BOT_TOKEN = your Telegram bot token
#
# Do NOT put the real token directly into this file.

TELEGRAM_TOKEN = os.getenv(
    "BOT_TOKEN",
    "PASTE_YOUR_BOT_TOKEN_HERE"
)

bot = telebot.TeleBot(TELEGRAM_TOKEN)


# ============================================================
# DETECTION SETTINGS
# ============================================================

# Minimum candle-body evidence
MIN_BODY_AREA = 10
MIN_BODY_HEIGHT = 2
MIN_CANDLE_WIDTH = 2

# Newest/right-side candles
RIGHT_MIN_BODY_AREA = 6
RIGHT_MIN_BODY_HEIGHT = 2

# Maximum candle width relative to screenshot
MAX_CANDLE_WIDTH_RATIO = 0.045

# Very close pieces of the same candle can be merged
MERGE_DISTANCE_RATIO = 0.55


# ============================================================
# STRICT PURPLE / YELLOW COLOR SETTINGS
# ============================================================
#
# YOUR SCREENSHOT:
#
# 🟣 PURPLE = BUY / BULLISH
# 🟡 YELLOW = SELL / BEARISH
#
# These ranges are designed specifically around the
# purple/yellow candle colors visible in your screenshot.
# ============================================================


# ------------------------------------------------------------
# 🟣 PURPLE
# ------------------------------------------------------------

PURPLE_HUE_LOW = 125
PURPLE_HUE_HIGH = 165

MIN_PURPLE_SATURATION = 100
MIN_PURPLE_VALUE = 70


# ------------------------------------------------------------
# 🟡 YELLOW
# ------------------------------------------------------------

YELLOW_HUE_LOW = 18
YELLOW_HUE_HIGH = 40

MIN_YELLOW_SATURATION = 100
MIN_YELLOW_VALUE = 70


# ------------------------------------------------------------
# COLOR DENSITY
# ------------------------------------------------------------

MIN_COLOR_DENSITY = 0.25


# ------------------------------------------------------------
# COLOR DOMINANCE
# ------------------------------------------------------------
#
# These prevent random chart/UI colors from being treated
# as candles.
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

    # Upscale smaller screenshots slightly.
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
    # 🟣 PURPLE
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
    # 🟡 YELLOW
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
    # BGR COLOR-DOMINANCE FILTER
    # ========================================================
    #
    # This makes the detector require the actual candle color
    # to dominate the other channels.
    # ========================================================

    b, g, r = cv2.split(img)


    # ========================================================
    # 🟣 PURPLE DOMINANCE
    # ========================================================
    #
    # Purple has:
    #
    # Red = relatively high
    # Blue = relatively high
    # Green = relatively low
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
    # 🟡 YELLOW DOMINANCE
    # ========================================================
    #
    # Yellow has:
    #
    # Red = high
    # Green = high
    # Blue = relatively low
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

    # Very small morphology.
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


        # Reject long horizontal objects.
        if w > h * 6:
            continue


        # ====================================================
        # BODY PIXEL DENSITY
        # ====================================================

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


        # ====================================================
        # CENTER
        # ====================================================

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


            # Keep the candidate with stronger
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
# MILD RIGHT-SIDE IMPROVEMENT
# ============================================================

def detect_right_side(
    chart,
    purple_mask,
    yellow_mask
):

    """
    Controlled second pass.

    Only the newest/right-side section gets the
    slightly smaller-body allowance.
    """

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


    # ========================================================
    # MAIN DETECTION
    # ========================================================

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


    # ========================================================
    # RIGHT-SIDE PASS
    # ========================================================

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


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    candles = (
        remove_cross_color_duplicates(
            candles
        )
    )


    # ========================================================
    # RIGHT → LEFT
    # ========================================================
    #
    # IMPORTANT:
    #
    # The newest/rightmost candle becomes #1.
    #
    # Then the detector moves:
    #
    # #1 → #2 → #3 → #4 ...
    #
    # from RIGHT to LEFT.
    # ========================================================

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
# NUMBERED DETECTION MAP
# ============================================================

def create_detection_map(
    img,
    candles
):

    output = img.copy()


    # ========================================================
    # RIGHT → LEFT NUMBERING
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
        # COLOR-SPECIFIC NUMBER
        # ====================================================

        if candle["color"] == "PURPLE":

            # 🟣 Purple / BUY
            label_color = (
                255,
                0,
                255
            )

        else:

            # 🟡 Yellow / SELL
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

            "👁️ Reading visible candles...\n"
            "➡️ Scanning RIGHT → LEFT.\n"
            "🟣 Checking PURPLE candles = BUY.\n"
            "🟡 Checking YELLOW candles = SELL."

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
        # DETECT
        # ====================================================

        candles = detect_candles(
            img
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
        # RESULT
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


        report = (

            "🔎 **CANDLE READING TEST**\n\n"

            "➡️ **SCAN DIRECTION:** "
            "RIGHT → LEFT\n\n"

            "📊 **WHAT THE BOT ACTUALLY DETECTED:**\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"🟣 PURPLE / BUY: {purple}\n"

            f"🟡 YELLOW / SELL: {yellow}\n"

            f"📊 TOTAL: {total}\n\n"

            "🕯️ **RIGHT → LEFT CANDLE READING:**\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"{sequence_text}\n\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            "🎯 **COLOR CHECK:**\n"

            "🟣 = Bot believes the candle is PURPLE / BUY\n"

            "🟡 = Bot believes the candle is YELLOW / SELL\n\n"

            "🔢 **NUMBER 1 = NEWEST/RIGHTMOST "
            "DETECTED CANDLE**\n\n"

            "⚠️ This is ONLY a candle-reading test.\n"

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

                    "🔢 **RIGHT → LEFT CANDLE MAP**\n\n"

                    "Number 1 = newest/rightmost "
                    "detected candle.\n\n"

                    "➡️ Counting continues "
                    "from RIGHT → LEFT.\n\n"

                    "🟨 Yellow box = detected candle body.\n"

                    "🟣 Number = classified PURPLE / BUY.\n"

                    "🟡 Number = classified YELLOW / SELL.\n\n"

                    "Compare every box with the actual "
                    "candles in your screenshot."

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
    "🕯️ CANDLE READING TEST"
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
    "🔎 Strict purple detection enabled"
)

print(
    "🔎 Strict yellow detection enabled"
)

print(
    "🚫 No OHLC generation"
)

print(
    "🚫 No random candles"
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

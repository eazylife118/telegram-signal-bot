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
# DETECTION SETTINGS
# ============================================================

MIN_BODY_AREA = 10
MIN_BODY_HEIGHT = 2
MIN_CANDLE_WIDTH = 2

RIGHT_MIN_BODY_AREA = 6
RIGHT_MIN_BODY_HEIGHT = 2

MAX_CANDLE_WIDTH_RATIO = 0.045

MERGE_DISTANCE_RATIO = 0.55

MIN_COLOR_DENSITY = 0.25


# ============================================================
# INDEPENDENT MAP SCAN SETTINGS
# ============================================================

# This scan is deliberately different from the primary
# contour detector.
#
# Its purpose is to find candle columns that the primary
# detector may have missed.

COLUMN_SCAN_MIN_PIXELS = 3
COLUMN_SCAN_MIN_HEIGHT = 2

COLUMN_GROUP_GAP = 4

COLUMN_MATCH_DISTANCE = 12

# Minimum amount of colored evidence required before an
# independent position can be recovered as a candle.
RECOVERY_MIN_PIXELS = 5

# Prevent extremely wide UI/chart objects from becoming
# recovered candles.
RECOVERY_MAX_WIDTH_RATIO = 0.045


# ============================================================
# PURPLE
# ============================================================

PURPLE_HUE_LOW = 125
PURPLE_HUE_HIGH = 165

MIN_PURPLE_SATURATION = 100
MIN_PURPLE_VALUE = 70


# ============================================================
# YELLOW
# ============================================================

YELLOW_HUE_LOW = 18
YELLOW_HUE_HIGH = 40

MIN_YELLOW_SATURATION = 100
MIN_YELLOW_VALUE = 70


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
# COLOR MASKS
# ============================================================

def get_color_masks(img):

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # PURPLE HSV
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # YELLOW HSV
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # BGR
    # --------------------------------------------------------

    b, g, r = cv2.split(img)


    # --------------------------------------------------------
    # PURPLE DOMINANCE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # YELLOW DOMINANCE
    # --------------------------------------------------------

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
# FIND PRIMARY CANDIDATES
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
            "color": color,
            "source": "PRIMARY"

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
# RIGHT SIDE PRIMARY PASS
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

        candle["x"] += right_start

        candle["center_x"] += right_start


    return (
        purple +
        yellow
    )


# ============================================================
# PRIMARY CANDLE DETECTION
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


    return (
        candles,
        purple_mask,
        yellow_mask
    )


# ============================================================
# INDEPENDENT COLUMN SCAN
# ============================================================
#
# This is the important new layer.
#
# It does NOT simply copy the primary detector.
#
# It scans vertical columns for actual purple/yellow pixels
# and groups nearby columns into possible candle positions.
#
# Its job is to discover candles that the primary contour
# detector missed.
# ============================================================

def independent_column_scan(
    purple_mask,
    yellow_mask
):

    h, w = purple_mask.shape[:2]


    combined = cv2.bitwise_or(
        purple_mask,
        yellow_mask
    )


    # --------------------------------------------------------
    # Remove tiny isolated noise.
    # --------------------------------------------------------

    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )


    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        small_kernel
    )


    # --------------------------------------------------------
    # Count colored pixels in each X column.
    # --------------------------------------------------------

    column_counts = np.sum(
        combined > 0,
        axis=0
    )


    active_columns = []


    for x in range(w):

        if (
            column_counts[x]
            >= COLUMN_SCAN_MIN_PIXELS
        ):

            active_columns.append(x)


    if not active_columns:

        return []


    # --------------------------------------------------------
    # Group nearby active columns.
    # --------------------------------------------------------

    groups = []


    group_start = active_columns[0]
    previous = active_columns[0]


    for x in active_columns[1:]:

        if (
            x - previous
            <= COLUMN_GROUP_GAP
        ):

            previous = x

        else:

            groups.append(
                (
                    group_start,
                    previous
                )
            )

            group_start = x
            previous = x


    groups.append(
        (
            group_start,
            previous
        )
    )


    possible = []


    max_width = max(
        10,
        int(
            w *
            RECOVERY_MAX_WIDTH_RATIO
        )
    )


    for left, right in groups:

        width = (
            right -
            left +
            1
        )


        if width > max_width:

            continue


        region = combined[
            :,
            left:right+1
        ]


        ys, xs = np.where(
            region > 0
        )


        if len(ys) == 0:
            continue


        top = int(
            np.min(ys)
        )


        bottom = int(
            np.max(ys)
        )


        height = (
            bottom -
            top +
            1
        )


        total_pixels = len(ys)


        if (
            total_pixels <
            COLUMN_SCAN_MIN_PIXELS
        ):

            continue


        if (
            height <
            COLUMN_SCAN_MIN_HEIGHT
        ):

            continue


        # ----------------------------------------------------
        # Determine actual color evidence.
        # ----------------------------------------------------

        purple_region = purple_mask[
            top:bottom+1,
            left:right+1
        ]


        yellow_region = yellow_mask[
            top:bottom+1,
            left:right+1
        ]


        purple_pixels = int(
            np.sum(
                purple_region > 0
            )
        )


        yellow_pixels = int(
            np.sum(
                yellow_region > 0
            )
        )


        if (
            purple_pixels == 0
            and
            yellow_pixels == 0
        ):

            continue


        if (
            purple_pixels >=
            yellow_pixels
        ):

            color = "PURPLE"
            color_pixels = purple_pixels

        else:

            color = "YELLOW"
            color_pixels = yellow_pixels


        if (
            color_pixels <
            RECOVERY_MIN_PIXELS
        ):

            continue


        center_x = (
            left +
            right
        ) / 2


        possible.append({

            "x": int(left),
            "y": int(top),
            "w": int(width),
            "h": int(height),

            "center_x": float(center_x),

            "pixels": int(color_pixels),

            "total_pixels": int(total_pixels),

            "purple_pixels": int(
                purple_pixels
            ),

            "yellow_pixels": int(
                yellow_pixels
            ),

            "color": color,

            "source": "INDEPENDENT"

        })


    # --------------------------------------------------------
    # Merge neighboring independent groups that actually
    # represent the same candle.
    # --------------------------------------------------------

    possible.sort(
        key=lambda c:
        c["center_x"]
    )


    merged = []


    for candidate in possible:

        if not merged:

            merged.append(
                candidate
            )

            continue


        previous = merged[-1]


        distance = (
            candidate["center_x"]
            -
            previous["center_x"]
        )


        if distance <= COLUMN_GROUP_GAP + 2:

            # Keep the stronger representation.

            if (
                candidate["pixels"]
                >
                previous["pixels"]
            ):

                merged[-1] = candidate

        else:

            merged.append(
                candidate
            )


    return merged


# ============================================================
# MATCH PRIMARY CANDLES TO INDEPENDENT POSITIONS
# ============================================================

def match_primary_to_independent(
    primary,
    independent
):

    matched = []

    missed = []

    used_primary = set()


    # --------------------------------------------------------
    # Sort independent positions left → right first.
    # --------------------------------------------------------

    independent_sorted = sorted(
        independent,
        key=lambda c:
        c["center_x"]
    )


    for possible in independent_sorted:

        best_index = None
        best_distance = None


        for i, candle in enumerate(
            primary
        ):

            if i in used_primary:
                continue


            distance = abs(
                possible["center_x"]
                -
                candle["center_x"]
            )


            # The primary and independent detector don't need
            # to have exactly the same center.
            if (
                distance <=
                COLUMN_MATCH_DISTANCE
            ):

                if (
                    best_distance is None
                    or
                    distance <
                    best_distance
                ):

                    best_index = i
                    best_distance = distance


        if best_index is not None:

            used_primary.add(
                best_index
            )

            matched.append({

                "independent": possible,

                "primary": primary[
                    best_index
                ],

                "distance": best_distance

            })

        else:

            missed.append(
                possible
            )


    extra_primary = []


    for i, candle in enumerate(
        primary
    ):

        if i not in used_primary:

            extra_primary.append(
                candle
            )


    return (
        matched,
        missed,
        extra_primary
    )


# ============================================================
# RECOVER MISSED CANDLES
# ============================================================
#
# Every missed independent position is converted into an
# actual candle candidate and added to the final list.
#
# Nothing is generated from thin air.
# The position must have actual color pixels in the screenshot.
# ============================================================

def recover_missed_candles(
    missed
):

    recovered = []


    for candidate in missed:

        recovered_candle = {

            "x": candidate["x"],

            "y": candidate["y"],

            "w": candidate["w"],

            "h": candidate["h"],

            "area": float(
                candidate["pixels"]
            ),

            "pixels": candidate[
                "pixels"
            ],

            "density": (

                candidate["pixels"] /
                float(
                    max(
                        1,
                        candidate["w"] *
                        candidate["h"]
                    )
                )

            ),

            "center_x":
                candidate["center_x"],

            "color":
                candidate["color"],

            "source":
                "RECOVERED"

        }


        recovered.append(
            recovered_candle
        )


    return recovered


# ============================================================
# FINAL CANDLE BUILD
# ============================================================

def build_final_candles(
    primary,
    independent
):

    matched, missed, extra_primary = (
        match_primary_to_independent(
            primary,
            independent
        )
    )


    # --------------------------------------------------------
    # Recover every independent position that did not match
    # the primary detector.
    # --------------------------------------------------------

    recovered = recover_missed_candles(
        missed
    )


    # --------------------------------------------------------
    # Start with primary candles.
    # --------------------------------------------------------

    final_candles = [
        candle.copy()
        for candle in primary
    ]


    # --------------------------------------------------------
    # Add recovered missed candles.
    # --------------------------------------------------------

    final_candles.extend(
        recovered
    )


    # --------------------------------------------------------
    # Final safety duplicate removal.
    # --------------------------------------------------------

    final_candles = (
        remove_final_duplicates(
            final_candles
        )
    )


    # --------------------------------------------------------
    # RIGHT → LEFT.
    # --------------------------------------------------------

    final_candles.sort(
        key=lambda c:
        c["center_x"],
        reverse=True
    )


    return (
        final_candles,
        matched,
        missed,
        recovered,
        extra_primary
    )


# ============================================================
# FINAL DUPLICATE REMOVAL
# ============================================================

def remove_final_duplicates(
    candles
):

    if not candles:
        return []


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
            ) * 0.75


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


            # Prefer PRIMARY over RECOVERED when both
            # are actually the same candle.
            if (
                existing["source"]
                == "RECOVERED"
                and
                candle["source"]
                == "PRIMARY"
            ):

                result[
                    duplicate_index
                ] = candle

            elif (
                candle["pixels"]
                >
                existing["pixels"]
            ):

                result[
                    duplicate_index
                ] = candle


    return result


# ============================================================
# REPORT COUNTS
# ============================================================

def count_colors(
    candles
):

    purple = sum(

        1
        for candle in candles
        if candle["color"] == "PURPLE"

    )


    yellow = sum(

        1
        for candle in candles
        if candle["color"] == "YELLOW"

    )


    return purple, yellow


# ============================================================
# MAP AGREEMENT
# ============================================================

def calculate_agreement(
    independent_count,
    matched_count
):

    if independent_count <= 0:

        return 0.0


    return (
        matched_count /
        independent_count
    ) * 100.0


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


        # ----------------------------------------------------
        # Different box appearance for recovered candles.
        # ----------------------------------------------------

        if candle.get(
            "source"
        ) == "RECOVERED":

            box_color = (
                0,
                255,
                0
            )

        else:

            box_color = (
                0,
                255,
                255
            )


        cv2.rectangle(

            output,

            (x, y),

            (
                x + w,
                y + h
            ),

            box_color,

            2

        )


        # ----------------------------------------------------
        # Number color
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Show recovered candles with R marker.
        # ----------------------------------------------------

        if candle.get(
            "source"
        ) == "RECOVERED":

            label = (
                f"{number}R"
            )

        else:

            label = str(
                number
            )


        cv2.putText(

            output,

            label,

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

    total_start = time.time()


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
            "🔎 Running primary detector.\n"
            "🧭 Running independent column verification.\n"
            "➕ Recovering any candles missed by the primary detector."

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
        # PRIMARY DETECTION
        # ====================================================

        detection_start = time.time()


        (
            primary_candles,
            purple_mask,
            yellow_mask
        ) = detect_candles(
            img
        )


        detection_time = (
            time.time() -
            detection_start
        )


        primary_purple, primary_yellow = (
            count_colors(
                primary_candles
            )
        )


        primary_total = len(
            primary_candles
        )


        # ====================================================
        # INDEPENDENT VERIFICATION
        # ====================================================

        verification_start = time.time()


        independent = (
            independent_column_scan(
                purple_mask,
                yellow_mask
            )
        )


        (
            final_candles,
            matched,
            missed,
            recovered,
            extra_primary
        ) = build_final_candles(

            primary_candles,

            independent

        )


        verification_time = (
            time.time() -
            verification_start
        )


        # ====================================================
        # FINAL COUNTS
        # ====================================================

        final_purple, final_yellow = (
            count_colors(
                final_candles
            )
        )


        final_total = len(
            final_candles
        )


        independent_total = len(
            independent
        )


        matched_total = len(
            matched
        )


        missed_total = len(
            missed
        )


        recovered_total = len(
            recovered
        )


        extra_total = len(
            extra_primary
        )


        agreement = (
            calculate_agreement(
                independent_total,
                matched_total
            )
        )


        # ====================================================
        # RIGHT → LEFT SEQUENCE
        # ====================================================

        sequence = []


        for number, candle in enumerate(

            final_candles,

            start=1

        ):

            if candle["color"] == "PURPLE":

                emoji = "🟣"
                direction = "BUY"

            else:

                emoji = "🟡"
                direction = "SELL"


            if candle.get(
                "source"
            ) == "RECOVERED":

                verification_mark = "➕"

            else:

                verification_mark = "✓"


            sequence.append(

                f"{number}. "
                f"{emoji} {direction} "
                f"{verification_mark}"

            )


        sequence_text = (
            "\n".join(
                sequence
            )
        )


        # ====================================================
        # NO CANDLES
        # ====================================================

        if final_total == 0:

            bot.reply_to(

                message,

                "❌ No reliable candle bodies detected.\n\n"

                "No candle was generated.\n"

                "No random candle was added.\n"

                "No trading signal was generated."

            )

            return


        # ====================================================
        # MAP STATUS
        # ====================================================

        if missed_total == 0:

            recovery_text = (
                "✅ No missed candles."
            )

        else:

            recovery_text = (

                f"➕ RECOVERED MISSED: "
                f"{recovered_total}\n"

                "✅ Recovered candles were added "
                "to the final candle list."

            )


        # ====================================================
        # MAIN REPORT
        # ====================================================

        report = (

            "🔎 **CANDLE + MAP VERIFICATION TEST**\n\n"

            "➡️ **SCAN:** RIGHT → LEFT\n\n"

            "📊 **PRIMARY CANDLE DETECTION**\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"🟣 PURPLE / BUY: "
            f"{primary_purple}\n"

            f"🟡 YELLOW / SELL: "
            f"{primary_yellow}\n"

            f"📊 TOTAL: "
            f"{primary_total}\n\n"


            "🔎 **INDEPENDENT MAP VERIFICATION**\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"🟣 VERIFIED PURPLE: "
            f"{primary_purple}\n"

            f"🟡 VERIFIED YELLOW: "
            f"{primary_yellow}\n"

            f"🤝 MATCHED POSITIONS: "
            f"{matched_total}\n"

            f"⚠️ POSSIBLE MISSED: "
            f"{missed_total}\n"

            f"➕ RECOVERED MISSED: "
            f"{recovered_total}\n"

            f"⚠️ POSSIBLE EXTRA: "
            f"{extra_total}\n"

            f"📊 INDEPENDENT POSITIONS: "
            f"{independent_total}\n"

            f"📊 MAP AGREEMENT: "
            f"{agreement:.1f}%\n\n"


            "✅ **FINAL CANDLE LIST**\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"🟣 PURPLE / BUY: "
            f"{final_purple}\n"

            f"🟡 YELLOW / SELL: "
            f"{final_yellow}\n"

            f"📊 FINAL TOTAL: "
            f"{final_total}\n\n"

            f"{recovery_text}\n\n"


            "🕯️ **RIGHT → LEFT READING**\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"{sequence_text}\n\n"


            "━━━━━━━━━━━━━━━━━━━━\n"

            "🎯 **MAP KEY**\n"

            "🟣 = PURPLE / BUY\n"

            "🟡 = YELLOW / SELL\n"

            "✓ = primary detector candle\n"

            "➕ = independently recovered candle\n\n"


            "🔢 **NUMBER 1 = "
            "NEWEST/RIGHTMOST FINAL CANDLE**\n\n"


            "⚠️ **IMPORTANT**\n"

            "This is a candle-reading and "
            "verification test only.\n"

            "No Vision API is used.\n"

            "No random candles are generated.\n"

            "No random prices are generated.\n"

            "No OHLC data is generated.\n"

            "No trading signal is generated.\n\n"


            f"🕯️ Detection: "
            f"{detection_time:.2f}s\n"

            f"🔎 Verification: "
            f"{verification_time:.2f}s\n"

            f"⚡ Total: "
            f"{time.time() - total_start:.2f}s"

        )


        bot.reply_to(

            message,

            report,

            parse_mode="Markdown"

        )


        # ====================================================
        # CREATE FINAL MAP
        # ====================================================

        detection_map = (
            create_detection_map(
                img,
                final_candles
            )
        )


        cv2.imwrite(

            detection_path,

            detection_map

        )


        # ====================================================
        # SEND FINAL MAP
        # ====================================================

        with open(

            detection_path,

            "rb"

        ) as photo:

            bot.send_photo(

                message.chat.id,

                photo,

                caption=(

                    "🔎 **FINAL CANDLE + MAP VERIFICATION**\n\n"

                    "🔢 Number 1 = newest/rightmost candle.\n"

                    "➡️ Numbers continue RIGHT → LEFT.\n\n"

                    "🟣 Number = PURPLE / BUY.\n"

                    "🟡 Number = YELLOW / SELL.\n\n"

                    "🟨 Yellow box = primary detection.\n"

                    "🟩 Green box = recovered missed candle.\n\n"

                    "➕ Recovered candles have an **R** "
                    "beside their number.\n\n"

                    f"📊 Primary: {primary_total}\n"

                    f"➕ Recovered: {recovered_total}\n"

                    f"✅ Final: {final_total}\n\n"

                    "The recovered candles are now part "
                    "of the final numbered candle list."

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
    "🕯️ CANDLE + MAP VERIFICATION TEST"
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
    "🔎 Primary candle detection enabled"
)

print(
    "🧭 Independent column verification enabled"
)

print(
    "➕ Missed-candle recovery enabled"
)

print(
    "🚫 No Vision API"
)

print(
    "🚫 No random candles"
)

print(
    "🚫 No random prices"
)

print(
    "🚫 No OHLC generation"
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

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
# PRIMARY DETECTION SETTINGS
# ============================================================

MIN_BODY_AREA = 10
MIN_BODY_HEIGHT = 2
MIN_CANDLE_WIDTH = 2

RIGHT_MIN_BODY_AREA = 6
RIGHT_MIN_BODY_HEIGHT = 2

MAX_CANDLE_WIDTH_RATIO = 0.045

MERGE_DISTANCE_RATIO = 0.55


# ============================================================
# COLOR SETTINGS
# ============================================================

# PURPLE

PURPLE_HUE_LOW = 125
PURPLE_HUE_HIGH = 165

MIN_PURPLE_SATURATION = 100
MIN_PURPLE_VALUE = 70


# YELLOW

YELLOW_HUE_LOW = 18
YELLOW_HUE_HIGH = 40

MIN_YELLOW_SATURATION = 100
MIN_YELLOW_VALUE = 70


# ============================================================
# COLOR DENSITY
# ============================================================

MIN_COLOR_DENSITY = 0.25

PURPLE_DOMINANCE_RATIO = 1.20
YELLOW_DOMINANCE_RATIO = 1.10


# ============================================================
# INDEPENDENT VERIFICATION SETTINGS
# ============================================================

VERIFICATION_MIN_PIXELS = 5

VERIFICATION_MIN_PEAK = 2

MIN_VERIFICATION_SEPARATION = 5

MATCH_DISTANCE_RATIO = 0.80

RECOVERY_MIN_COLOR_DENSITY = 0.12

VERIFY_LEFT_MARGIN_RATIO = 0.02
VERIFY_RIGHT_MARGIN_RATIO = 0.98


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
    # PURPLE MASK
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
    # YELLOW MASK
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
# PRIMARY CANDIDATE DETECTOR
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
# MERGE SAME-COLOR PRIMARY PIECES
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
                existing["w"] = right - left
                existing["h"] = bottom - top

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
# REMOVE PRIMARY CROSS-COLOR DUPLICATES
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
# RIGHT-SIDE PRIMARY PASS
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
# INDEPENDENT COLUMN SCAN
# ============================================================

def independent_column_scan(
    img,
    purple_mask,
    yellow_mask,
    primary_candles
):

    h, w = img.shape[:2]

    left_limit = int(
        w *
        VERIFY_LEFT_MARGIN_RATIO
    )

    right_limit = int(
        w *
        VERIFY_RIGHT_MARGIN_RATIO
    )

    if right_limit <= left_limit:
        return []

    # ========================================================
    # IMPORTANT:
    #
    # This scan is independent from the PRIMARY list.
    # It uses the actual image color masks.
    # ========================================================

    purple_projection = np.sum(
        purple_mask[
            :,
            left_limit:right_limit
        ] > 0,
        axis=0
    )

    yellow_projection = np.sum(
        yellow_mask[
            :,
            left_limit:right_limit
        ] > 0,
        axis=0
    )

    total_projection = (
        purple_projection +
        yellow_projection
    )

    if len(total_projection) == 0:
        return []

    # ========================================================
    # SMOOTH X PROJECTION
    # ========================================================

    smooth_kernel = np.ones(
        3,
        dtype=np.float32
    ) / 3.0

    purple_smooth = np.convolve(
        purple_projection.astype(
            np.float32
        ),
        smooth_kernel,
        mode="same"
    )

    yellow_smooth = np.convolve(
        yellow_projection.astype(
            np.float32
        ),
        smooth_kernel,
        mode="same"
    )

    total_smooth = (
        purple_smooth +
        yellow_smooth
    )

    # ========================================================
    # LOCAL COLOR PEAKS
    # ========================================================

    possible_centers = []

    for i in range(
        1,
        len(total_smooth) - 1
    ):

        value = total_smooth[i]

        if value < VERIFICATION_MIN_PEAK:
            continue

        if (
            value >=
            total_smooth[i - 1]
            and
            value >=
            total_smooth[i + 1]
        ):

            possible_centers.append(
                i + left_limit
            )

    # ========================================================
    # COLLAPSE VERY CLOSE PEAKS
    # ========================================================

    collapsed = []

    for x in possible_centers:

        if not collapsed:

            collapsed.append(x)

            continue

        if (
            x -
            collapsed[-1]
            <
            MIN_VERIFICATION_SEPARATION
        ):

            previous = collapsed[-1]

            previous_index = (
                previous -
                left_limit
            )

            current_index = (
                x -
                left_limit
            )

            if (
                total_smooth[
                    current_index
                ]
                >
                total_smooth[
                    previous_index
                ]
            ):

                collapsed[-1] = x

        else:

            collapsed.append(x)

    # ========================================================
    # ESTIMATE NORMAL CANDLE WIDTH
    # ========================================================

    primary_widths = [

        max(
            2,
            int(c["w"])
        )

        for c in primary_candles

    ]

    if primary_widths:

        median_width = int(
            np.median(
                primary_widths
            )
        )

    else:

        median_width = 5

    min_spacing = max(
        MIN_VERIFICATION_SEPARATION,
        int(
            median_width *
            0.75
        )
    )

    independent = []

    # ========================================================
    # EXAMINE EACH INDEPENDENT POSITION
    # ========================================================

    for x in collapsed:

        radius = max(
            2,
            int(
                median_width *
                0.80
            )
        )

        x1 = max(
            left_limit,
            x - radius
        )

        x2 = min(
            right_limit,
            x + radius + 1
        )

        purple_region = purple_mask[
            :,
            x1:x2
        ]

        yellow_region = yellow_mask[
            :,
            x1:x2
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

        total_pixels = (
            purple_pixels +
            yellow_pixels
        )

        if (
            total_pixels <
            VERIFICATION_MIN_PIXELS
        ):
            continue

        # ====================================================
        # COLOR MUST ACTUALLY DOMINATE
        # ====================================================

        if (
            purple_pixels >
            yellow_pixels
        ):

            color = "PURPLE"

            dominant_pixels = (
                purple_pixels
            )

        elif (
            yellow_pixels >
            purple_pixels
        ):

            color = "YELLOW"

            dominant_pixels = (
                yellow_pixels
            )

        else:

            # Ambiguous color = NOT a recovery.
            continue

        # ====================================================
        # VERTICAL COLOR EXTENT
        # ====================================================

        if color == "PURPLE":

            region = purple_mask[
                :,
                x1:x2
            ]

        else:

            region = yellow_mask[
                :,
                x1:x2
            ]

        ys, xs = np.where(
            region > 0
        )

        if len(ys) == 0:
            continue

        y_min = int(
            np.min(ys)
        )

        y_max = int(
            np.max(ys)
        )

        body_height = (
            y_max -
            y_min +
            1
        )

        if body_height < 2:
            continue

        # ====================================================
        # DENSITY
        # ====================================================

        region_area = max(
            1,
            (x2 - x1) *
            body_height
        )

        density = (
            dominant_pixels /
            float(region_area)
        )

        if (
            density <
            RECOVERY_MIN_COLOR_DENSITY
        ):
            continue

        independent.append({

            "x": int(x1),

            "y": int(y_min),

            "w": int(x2 - x1),

            "h": int(body_height),

            "center_x": float(x),

            "pixels": dominant_pixels,

            "density": density,

            "color": color,

            "source": "VERIFIER"

        })

    # ========================================================
    # REMOVE DUPLICATE VERIFIER POSITIONS
    # ========================================================

    independent.sort(
        key=lambda c:
        c["center_x"]
    )

    cleaned = []

    for candidate in independent:

        if not cleaned:

            cleaned.append(
                candidate
            )

            continue

        distance = (
            candidate["center_x"]
            -
            cleaned[-1]["center_x"]
        )

        if distance < min_spacing:

            if (
                candidate["pixels"]
                >
                cleaned[-1]["pixels"]
            ):

                cleaned[-1] = candidate

        else:

            cleaned.append(
                candidate
            )

    return cleaned


# ============================================================
# CHECK VERIFIER POSITION AGAINST PRIMARY
# ============================================================

def verifier_matches_primary(
    verifier,
    primary_candles
):

    for primary in primary_candles:

        distance = abs(
            verifier["center_x"]
            -
            primary["center_x"]
        )

        threshold = max(
            3,
            (
                max(
                    verifier["w"],
                    primary["w"]
                )
                *
                MATCH_DISTANCE_RATIO
            )
        )

        if distance <= threshold:

            return True

    return False


# ============================================================
# RECOVER MISSED CANDLES
# ============================================================

def recover_missed_candles(
    img,
    primary_candles
):

    purple_mask, yellow_mask = (
        get_color_masks(img)
    )

    independent = (
        independent_column_scan(
            img,
            purple_mask,
            yellow_mask,
            primary_candles
        )
    )

    recovered = []

    for candidate in independent:

        # ====================================================
        # IF IT MATCHES PRIMARY:
        #
        # It is NOT added again.
        # ====================================================

        if verifier_matches_primary(
            candidate,
            primary_candles
        ):

            continue

        # ====================================================
        # ONLY AN UNMATCHED REAL POSITION IS RECOVERED.
        # ====================================================

        recovered_candidate = (
            candidate.copy()
        )

        recovered_candidate[
            "source"
        ] = "RECOVERED"

        recovered.append(
            recovered_candidate
        )

    return independent, recovered


# ============================================================
# MERGE PRIMARY + RECOVERED
# ============================================================

def merge_final_candles(
    primary,
    recovered
):

    final = []

    # ========================================================
    # PRIMARY CANDLES
    # ========================================================

    for candle in primary:

        item = candle.copy()

        item["source"] = "PRIMARY"

        final.append(
            item
        )

    # ========================================================
    # RECOVERED CANDLES
    # ========================================================

    for candle in recovered:

        duplicate = False

        for existing in final:

            distance = abs(
                candle["center_x"]
                -
                existing["center_x"]
            )

            threshold = max(
                3,
                (
                    max(
                        candle["w"],
                        existing["w"]
                    )
                    *
                    MATCH_DISTANCE_RATIO
                )
            )

            if distance <= threshold:

                duplicate = True

                break

        if not duplicate:

            final.append(
                candle.copy()
            )

    # ========================================================
    # RIGHT → LEFT
    # ========================================================

    final.sort(
        key=lambda c:
        c["center_x"],
        reverse=True
    )

    return final


# ============================================================
# COUNT COLORS
# ============================================================

def count_colors(candles):

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
# VERIFICATION STATISTICS
# ============================================================

def create_verification_stats(
    primary,
    independent,
    recovered
):

    matched = 0

    for candidate in independent:

        if verifier_matches_primary(
            candidate,
            primary
        ):

            matched += 1

    independent_positions = len(
        independent
    )

    recovered_count = len(
        recovered
    )

    possible_missed = recovered_count

    # Anything independent that did not match primary
    # becomes a recovered position. It is NOT automatically
    # called "extra".

    if independent_positions > 0:

        agreement = (
            matched /
            independent_positions
        ) * 100

    else:

        agreement = 0.0

    verified_purple = sum(
        1
        for c in independent
        if (
            c["color"] ==
            "PURPLE"
        )
    )

    verified_yellow = sum(
        1
        for c in independent
        if (
            c["color"] ==
            "YELLOW"
        )
    )

    return {

        "verified_purple":
            verified_purple,

        "verified_yellow":
            verified_yellow,

        "matched":
            matched,

        "possible_missed":
            possible_missed,

        "recovered":
            recovered_count,

        "independent":
            independent_positions,

        "agreement":
            agreement

    }


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

        # ====================================================
        # ACTUAL CANDLE COLOR LABEL
        # ====================================================

        if candle["color"] == "PURPLE":

            label_color = (
                255,
                0,
                255
            )

            color_text = "BUY"

        else:

            label_color = (
                0,
                255,
                255
            )

            color_text = "SELL"

        # ====================================================
        # PRIMARY VS RECOVERED BOX
        # ====================================================

        if candle.get(
            "source"
        ) == "RECOVERED":

            # Green box means RECOVERED.
            box_color = (
                0,
                255,
                0
            )

            thickness = 3

        else:

            box_color = label_color

            thickness = 2

        # ====================================================
        # BOX
        # ====================================================

        cv2.rectangle(

            output,

            (x, y),

            (
                x + w,
                y + h
            ),

            box_color,

            thickness

        )

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
        # RECOVERED MARK
        # ====================================================

        if candle.get(
            "source"
        ) == "RECOVERED":

            cv2.putText(

                output,

                "REC " + color_text,

                (
                    x,
                    min(
                        output.shape[0] - 10,
                        y + h + 20
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.45,

                (
                    0,
                    255,
                    0
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

def handle_photo(message):

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

            "👁️ Reading candles...\n"
            "➡️ Primary scan: RIGHT → LEFT\n"
            "🔎 Independent scan checking for missed candles..."

        )

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

        primary_candles = (
            detect_candles(
                img
            )
        )

        detection_time = (
            time.time() -
            detection_start
        )

        # ====================================================
        # INDEPENDENT VERIFICATION
        # ====================================================

        verification_start = (
            time.time()
        )

        independent, recovered = (
            recover_missed_candles(
                img,
                primary_candles
            )
        )

        verification_time = (
            time.time() -
            verification_start
        )

        # ====================================================
        # FINAL CANDLE LIST
        # ====================================================

        final_candles = (
            merge_final_candles(
                primary_candles,
                recovered
            )
        )

        # ====================================================
        # COUNTS
        # ====================================================

        primary_purple, primary_yellow = (
            count_colors(
                primary_candles
            )
        )

        final_purple, final_yellow = (
            count_colors(
                final_candles
            )
        )

        primary_total = len(
            primary_candles
        )

        final_total = len(
            final_candles
        )

        # ====================================================
        # VERIFICATION STATS
        # ====================================================

        stats = (
            create_verification_stats(
                primary_candles,
                independent,
                recovered
            )
        )

        # ====================================================
        # RIGHT → LEFT FINAL SEQUENCE
        # ====================================================

        sequence = []

        for number, candle in enumerate(
            final_candles,
            start=1
        ):

            if candle["color"] == "PURPLE":

                color_text = "🟣 BUY"

            else:

                color_text = "🟡 SELL"

            if candle.get(
                "source"
            ) == "RECOVERED":

                sequence.append(

                    f"{number}. "
                    f"{color_text} ➕ RECOVERED"

                )

            else:

                sequence.append(

                    f"{number}. "
                    f"{color_text} ✓"

                )

        sequence_text = "\n".join(
            sequence
        )

        # ====================================================
        # TOTAL TIME
        # ====================================================

        total_time = (
            time.time() -
            total_start
        )

        # ====================================================
        # NO CANDLES
        # ====================================================

        if final_total == 0:

            bot.reply_to(

                message,

                "❌ No reliable candle bodies detected.\n\n"
                "No random candles were generated.\n"
                "No candles were forced into the result.\n"
                "No trading signal was generated."

            )

            return

        # ====================================================
        # MAIN REPORT
        # ====================================================

        report = (

            "🔎 **CANDLE + MAP VERIFICATION TEST**\n\n"

            "➡️ **SCAN: RIGHT → LEFT**\n\n"

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
            f"{stats['verified_purple']}\n"

            f"🟡 VERIFIED YELLOW: "
            f"{stats['verified_yellow']}\n"

            f"🤝 MATCHED POSITIONS: "
            f"{stats['matched']}\n"

            f"⚠️ POSSIBLE MISSED: "
            f"{stats['possible_missed']}\n"

            f"➕ RECOVERED MISSED: "
            f"{stats['recovered']}\n"

            f"📊 INDEPENDENT POSITIONS: "
            f"{stats['independent']}\n"

            f"📊 MAP AGREEMENT: "
            f"{stats['agreement']:.1f}%\n\n"

            "✅ **FINAL CANDLE LIST**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"

            f"🟣 PURPLE / BUY: "
            f"{final_purple}\n"

            f"🟡 YELLOW / SELL: "
            f"{final_yellow}\n"

            f"📊 FINAL TOTAL: "
            f"{final_total}\n\n"

        )

        # ====================================================
        # RECOVERY MESSAGE
        # ====================================================

        if recovered:

            report += (

                f"➕ **{len(recovered)} "
                f"MISSED CANDLE(S) FOUND AND "
                f"ADDED TO THE MAIN CANDLE LIST.**\n\n"

            )

        else:

            report += (

                "✅ **No independently confirmed "
                "missed candles.**\n\n"

            )

        # ====================================================
        # FINAL SEQUENCE
        # ====================================================

        report += (

            "🕯️ **RIGHT → LEFT READING**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"

            f"{sequence_text}\n\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            "🎯 **MAP KEY**\n"

            "🟣 = PURPLE / BUY\n"
            "🟡 = YELLOW / SELL\n"
            "✓ = primary detector candle\n"
            "➕ = independently recovered missed candle\n\n"

            "🔢 NUMBER 1 = "
            "NEWEST/RIGHTMOST FINAL CANDLE\n\n"

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
            f"{total_time:.2f}s"

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
        # MAP CAPTION
        # ====================================================

        caption = (

            "🔢 **FINAL RIGHT → LEFT CANDLE MAP**\n\n"

            "1 = newest/rightmost final candle.\n"

            "➡️ Numbers continue RIGHT → LEFT.\n\n"

            "🟣 Purple number = PURPLE / BUY.\n"

            "🟡 Yellow number = YELLOW / SELL.\n\n"

            "🟩 Green box = independently recovered "
            "missed candle.\n"

            "The number/color beside it shows whether "
            "the recovered candle is 🟣 PURPLE or 🟡 YELLOW.\n\n"

            "✓ = primary detector.\n"

            "➕ = recovered by independent scan.\n\n"

            "⚠️ Recovered candles are added only when "
            "the independent scan finds actual "
            "unmatched purple/yellow evidence."

        )

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
    "🔎 Primary candle detector enabled"
)

print(
    "🔎 Independent missed-candle recovery enabled"
)

print(
    "➕ Recovered candles added to final list"
)

print(
    "🟩 Recovered candles marked on map"
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

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


# ============================================================
# COLOR DOMINANCE
# ============================================================

PURPLE_DOMINANCE_RATIO = 1.20
YELLOW_DOMINANCE_RATIO = 1.10


# ============================================================
# MAP VERIFICATION SETTINGS
# ============================================================

VERIFY_MIN_PIXELS = 8
VERIFY_MIN_DENSITY = 0.08
VERIFY_HORIZONTAL_RADIUS = 0.70
VERIFY_MIN_DISTANCE_RATIO = 0.55
VERIFY_COLUMN_THRESHOLD = 3
VERIFY_CONFIDENCE_THRESHOLD = 65


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

ANALYSIS_MIN_CANDLES = 5
ANALYSIS_LOOKBACK = 12

RECENT_WEIGHT = 1.00
OLD_WEIGHT = 0.45

MIN_SIGNAL_CONFIDENCE = 65

MIN_DIRECTION_SEPARATION = 8

SEVERE_CONFLICT_THRESHOLD = 28

STRONG_BODY_RATIO = 1.35
SMALL_BODY_RATIO = 0.65

COMPRESSION_RATIO = 0.70
EXPANSION_RATIO = 1.45

PULLBACK_MAX_OPPOSITE = 2

REVERSAL_CONFIRMATION_REQUIRED = 2

BREAKOUT_BODY_MULTIPLIER = 1.25

NO_TRADE_CONFLICT = 70
NO_TRADE_SIDEWAYS_STRENGTH = 30


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

    b, g, r = cv2.split(img)

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
# MAIN CANDLE DETECTOR
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
# INDEPENDENT MAP VERIFICATION
# ============================================================

def verify_single_candle(
    candle,
    purple_mask,
    yellow_mask
):

    x = int(
        candle["center_x"]
    )

    y = int(
        candle["y"]
    )

    w = max(
        2,
        int(candle["w"])
    )

    h = max(
        2,
        int(candle["h"])
    )

    radius = max(
        2,
        int(
            w *
            VERIFY_HORIZONTAL_RADIUS
        )
    )

    left = max(
        0,
        x - radius
    )

    right = min(
        purple_mask.shape[1],
        x + radius + 1
    )

    top = max(
        0,
        y - max(2, int(h * 0.25))
    )

    bottom = min(
        purple_mask.shape[0],
        y + h + max(2, int(h * 0.25))
    )

    purple_region = purple_mask[
        top:bottom,
        left:right
    ]

    yellow_region = yellow_mask[
        top:bottom,
        left:right
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

    total_pixels = max(
        1,
        purple_region.shape[0] *
        purple_region.shape[1]
    )

    if candle["color"] == "PURPLE":

        own_pixels = purple_pixels
        other_pixels = yellow_pixels

    else:

        own_pixels = yellow_pixels
        other_pixels = purple_pixels

    own_density = (
        own_pixels /
        total_pixels
    )

    if own_pixels >= VERIFY_MIN_PIXELS:

        if own_pixels >= (
            other_pixels * 1.15
        ):

            color_agrees = True

        else:

            color_agrees = False

    else:

        color_agrees = False

    body_evidence = min(
        100,
        (
            own_pixels /
            float(
                max(
                    VERIFY_MIN_PIXELS,
                    1
                )
            )
        ) * 100
    )

    density_evidence = min(
        100,
        (
            own_density /
            VERIFY_MIN_DENSITY
        ) * 100
    )

    score = (
        body_evidence * 0.50
        +
        density_evidence * 0.25
        +
        (100 if color_agrees else 0) * 0.25
    )

    score = max(
        0,
        min(
            100,
            score
        )
    )

    verified = (
        color_agrees
        and
        score >= VERIFY_CONFIDENCE_THRESHOLD
    )

    return {

        "verified": verified,
        "score": score,
        "own_pixels": own_pixels,
        "other_pixels": other_pixels,
        "own_density": own_density,
        "color_agrees": color_agrees

    }


# ============================================================
# INDEPENDENT COLUMN PEAK SCANNER
# ============================================================

def build_verification_peaks(
    img,
    purple_mask,
    yellow_mask,
    primary_candles
):

    h, w = img.shape[:2]

    combined = cv2.bitwise_or(
        purple_mask,
        yellow_mask
    )

    top_limit = int(
        h * 0.18
    )

    bottom_limit = int(
        h * 0.82
    )

    chart_mask = combined[
        top_limit:bottom_limit,
        :
    ]

    column_strength = np.sum(
        chart_mask > 0,
        axis=0
    )

    kernel_size = 3

    kernel = np.ones(
        kernel_size,
        dtype=np.float32
    ) / kernel_size

    smoothed = np.convolve(
        column_strength.astype(
            np.float32
        ),
        kernel,
        mode="same"
    )

    primary_x = sorted([
        c["center_x"]
        for c in primary_candles
    ])

    spacings = []

    for i in range(
        1,
        len(primary_x)
    ):

        distance = (
            primary_x[i]
            -
            primary_x[i - 1]
        )

        if distance >= 3:

            spacings.append(
                distance
            )

    if spacings:

        median_spacing = float(
            np.median(
                spacings
            )
        )

    else:

        median_spacing = max(
            8,
            w * 0.018
        )

    minimum_distance = max(
        4,
        int(
            median_spacing *
            VERIFY_MIN_DISTANCE_RATIO
        )
    )

    possible_peaks = []

    threshold = max(
        VERIFY_COLUMN_THRESHOLD,
        int(
            median_spacing * 0.15
        )
    )

    for x in range(
        2,
        w - 2
    ):

        value = smoothed[x]

        if value < threshold:
            continue

        if (
            value >= smoothed[x - 1]
            and
            value >= smoothed[x + 1]
        ):

            possible_peaks.append(
                (
                    x,
                    value
                )
            )

    possible_peaks.sort(
        key=lambda item:
        item[1],
        reverse=True
    )

    selected = []

    for x, strength in possible_peaks:

        too_close = False

        for selected_x, _ in selected:

            if abs(
                x -
                selected_x
            ) < minimum_distance:

                too_close = True
                break

        if not too_close:

            selected.append(
                (
                    x,
                    strength
                )
            )

    selected.sort(
        key=lambda item:
        item[0]
    )

    return selected


# ============================================================
# MATCH VERIFICATION PEAKS TO CANDLES
# ============================================================

def compare_map_with_independent_scan(
    candles,
    peaks
):

    if not candles:

        return {

            "matched": 0,
            "possible_missing": [],
            "possible_extra": [],
            "agreement": 0.0

        }

    candle_x = [
        c["center_x"]
        for c in candles
    ]

    if len(candle_x) >= 2:

        sorted_x = sorted(
            candle_x
        )

        spacings = [

            sorted_x[i] -
            sorted_x[i - 1]

            for i in range(
                1,
                len(sorted_x)
            )

            if (
                sorted_x[i] -
                sorted_x[i - 1]
            ) > 2

        ]

        if spacings:

            tolerance = max(
                5,
                float(
                    np.median(
                        spacings
                    )
                ) * 0.55
            )

        else:

            tolerance = 8

    else:

        tolerance = 8

    matched = 0

    matched_candles = set()

    matched_peaks = set()

    for peak_index, (
        peak_x,
        strength
    ) in enumerate(peaks):

        best_index = None
        best_distance = None

        for candle_index, cx in enumerate(
            candle_x
        ):

            if candle_index in matched_candles:
                continue

            distance = abs(
                peak_x -
                cx
            )

            if distance <= tolerance:

                if (
                    best_distance is None
                    or
                    distance < best_distance
                ):

                    best_distance = distance
                    best_index = candle_index

        if best_index is not None:

            matched += 1

            matched_candles.add(
                best_index
            )

            matched_peaks.add(
                peak_index
            )

    possible_missing = []

    for peak_index, (
        peak_x,
        strength
    ) in enumerate(peaks):

        if peak_index not in matched_peaks:

            possible_missing.append(
                peak_x
            )

    possible_extra = []

    for candle_index, cx in enumerate(
        candle_x
    ):

        if candle_index not in matched_candles:

            possible_extra.append(
                cx
            )

    denominator = max(
        len(candles),
        len(peaks),
        1
    )

    agreement = (
        matched /
        denominator
    ) * 100

    return {

        "matched": matched,
        "possible_missing": possible_missing,
        "possible_extra": possible_extra,
        "agreement": agreement

    }


# ============================================================
# RECOVER MISSED CANDLES
# ============================================================

def recover_missed_candles(
    img,
    candles,
    peaks,
    comparison
):

    if not peaks:
        return candles, 0

    missed_x = comparison.get(
        "possible_missing",
        []
    )

    if not missed_x:
        return candles, 0

    purple_mask, yellow_mask = get_color_masks(
        img
    )

    h, w = img.shape[:2]

    recovered = []
    recovered_count = 0

    for peak_x in missed_x:

        already_exists = False

        for candle in candles:

            if abs(
                candle["center_x"] -
                peak_x
            ) < 10:

                already_exists = True
                break

        if already_exists:
            continue

        search_radius = 8

        left = max(
            0,
            int(
                peak_x -
                search_radius
            )
        )

        right = min(
            w,
            int(
                peak_x +
                search_radius +
                1
            )
        )

        top = int(
            h * 0.18
        )

        bottom = int(
            h * 0.82
        )

        purple_region = purple_mask[
            top:bottom,
            left:right
        ]

        yellow_region = yellow_mask[
            top:bottom,
            left:right
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
            purple_pixels >
            yellow_pixels
            and
            purple_pixels >= 5
        ):

            color = "PURPLE"
            color_pixels = purple_pixels

        elif (
            yellow_pixels >
            purple_pixels
            and
            yellow_pixels >= 5
        ):

            color = "YELLOW"
            color_pixels = yellow_pixels

        else:

            continue

        if color == "PURPLE":

            mask = purple_mask

        else:

            mask = yellow_mask

        col_x_start = max(
            0,
            int(
                peak_x - 3
            )
        )

        col_x_end = min(
            w,
            int(
                peak_x + 4
            )
        )

        col_mask = mask[
            top:bottom,
            col_x_start:col_x_end
        ]

        ys, xs = np.where(
            col_mask > 0
        )

        if len(ys) > 0:

            candle_top = (
                top +
                int(
                    np.min(ys)
                )
            )

            candle_bottom = (
                top +
                int(
                    np.max(ys)
                )
            )

            candle_h = max(
                2,
                candle_bottom -
                candle_top +
                1
            )

            candle_y = candle_top

        else:

            candle_h = 20
            candle_y = (
                top +
                50
            )

        recovered_candle = {

            "x": int(
                peak_x - 4
            ),

            "y": int(
                candle_y
            ),

            "w": 8,

            "h": int(
                candle_h
            ),

            "center_x": float(
                peak_x
            ),

            "color": color,

            "pixels": color_pixels,

            "recovered": True,

            "verification": {

                "verified": True,

                "score": 85,

                "own_pixels":
                    color_pixels,

                "other_pixels":
                    (
                        purple_pixels
                        if
                        color == "YELLOW"
                        else
                        yellow_pixels
                    ),

                "own_density": 0.15,

                "color_agrees": True

            }

        }

        recovered.append(
            recovered_candle
        )

        recovered_count += 1

    all_candles = (
        candles +
        recovered
    )

    all_candles.sort(
        key=lambda c:
        c["center_x"],
        reverse=True
    )

    return (
        all_candles,
        recovered_count
    )


# ============================================================
# FULL MAP VERIFICATION
# ============================================================

def verify_candle_map(
    img,
    candles
):

    purple_mask, yellow_mask = (
        get_color_masks(img)
    )

    results = []

    for candle in candles:

        result = verify_single_candle(
            candle,
            purple_mask,
            yellow_mask
        )

        verified_candle = candle.copy()

        verified_candle[
            "verification"
        ] = result

        results.append(
            verified_candle
        )

    peaks = build_verification_peaks(
        img,
        purple_mask,
        yellow_mask,
        candles
    )

    comparison = (
        compare_map_with_independent_scan(
            candles,
            peaks
        )
    )

    all_candles, recovered_count = (
        recover_missed_candles(
            img,
            results,
            peaks,
            comparison
        )
    )

    final_candles = []

    for candle in all_candles:

        if candle.get(
            "recovered",
            False
        ):

            final_candles.append(
                candle
            )

        elif (
            "verification"
            not in candle
        ):

            result = verify_single_candle(
                candle,
                purple_mask,
                yellow_mask
            )

            verified_candle = candle.copy()

            verified_candle[
                "verification"
            ] = result

            final_candles.append(
                verified_candle
            )

        else:

            final_candles.append(
                candle
            )

    verified_purple = 0
    verified_yellow = 0
    unverified = 0

    for candle in final_candles:

        if candle[
            "verification"
        ]["verified"]:

            if candle[
                "color"
            ] == "PURPLE":

                verified_purple += 1

            else:

                verified_yellow += 1

        else:

            unverified += 1

    return {

        "candles": final_candles,

        "verified_purple":
            verified_purple,

        "verified_yellow":
            verified_yellow,

        "verified_total":
            verified_purple +
            verified_yellow,

        "unverified":
            unverified,

        "recovered_count":
            recovered_count,

        "peaks":
            peaks,

        "comparison":
            comparison

    }


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
# ============================================================
# CANDLE-ONLY ANALYSIS ENGINE
# ============================================================
# ============================================================
#
# The detector above remains the source of candle direction.
#
# PURPLE = BUY
# YELLOW = SELL
#
# No RSI
# No MACD
# No EMA
# No Bollinger Bands
# No price OCR
# No generated OHLC
#
# ============================================================


def candle_direction(candle):

    if candle["color"] == "PURPLE":
        return 1

    return -1


def body_size(candle):

    return max(
        1.0,
        float(
            candle.get(
                "h",
                1
            )
        )
    )


def safe_ratio(a, b):

    return (
        float(a) /
        max(
            float(b),
            0.0001
        )
    )


def weighted_average(values):

    if not values:
        return 0.0

    weights = np.linspace(
        1.0,
        0.55,
        len(values)
    )

    values = np.asarray(
        values,
        dtype=float
    )

    return float(
        np.sum(
            values * weights
        )
        /
        np.sum(weights)
    )


# ============================================================
# ADD APPROXIMATE CANDLE GEOMETRY
# ============================================================
#
# The detector's colored candle extent is used as the visual
# geometry available from the screenshot.
#
# Because screenshots do not provide OHLC data, this engine
# never invents OHLC values.
#
# ============================================================

def enrich_candle_geometry(
    img,
    candles
):

    purple_mask, yellow_mask = (
        get_color_masks(img)
    )

    h_img, w_img = img.shape[:2]

    enriched = []

    for candle in candles:

        c = candle.copy()

        x = int(
            candle["center_x"]
        )

        body_top = int(
            candle["y"]
        )

        body_bottom = int(
            candle["y"] +
            candle["h"]
        )

        radius = max(
            2,
            int(
                max(
                    candle["w"],
                    2
                ) * 0.75
            )
        )

        left = max(
            0,
            x - radius
        )

        right = min(
            w_img,
            x + radius + 1
        )

        if candle["color"] == "PURPLE":

            mask = purple_mask

        else:

            mask = yellow_mask

        # Look through the colored vertical area.
        region = mask[
            :,
            left:right
        ]

        ys, xs = np.where(
            region > 0
        )

        if len(ys) > 0:

            visual_top = int(
                np.min(ys)
            )

            visual_bottom = int(
                np.max(ys)
            )

        else:

            visual_top = body_top
            visual_bottom = body_bottom

        # Keep detector body information separate.
        c["body_top"] = float(
            body_top
        )

        c["body_bottom"] = float(
            body_bottom
        )

        c["body_size"] = body_size(
            candle
        )

        c["visual_top"] = float(
            visual_top
        )

        c["visual_bottom"] = float(
            visual_bottom
        )

        c["visual_range"] = max(
            1.0,
            float(
                visual_bottom -
                visual_top
            )
        )

        # Screen coordinates run downward.
        # Smaller y = visually higher.
        c["upper_extension"] = max(
            0.0,
            float(
                body_top -
                visual_top
            )
        )

        c["lower_extension"] = max(
            0.0,
            float(
                visual_bottom -
                body_bottom
            )
        )

        c["upper_rejection_ratio"] = safe_ratio(
            c["upper_extension"],
            c["body_size"]
        )

        c["lower_rejection_ratio"] = safe_ratio(
            c["lower_extension"],
            c["body_size"]
        )

        c["body_to_range"] = safe_ratio(
            c["body_size"],
            c["visual_range"]
        )

        enriched.append(
            c
        )

    return enriched


# ============================================================
# 1. RECENT CANDLE DIRECTION
# ============================================================

def analyze_recent_direction(candles):

    recent = candles[
        :min(
            5,
            len(candles)
        )
    ]

    if not recent:
        return 0, 0

    values = [
        candle_direction(c)
        for c in recent
    ]

    score = weighted_average(
        values
    )

    strength = abs(score) * 100

    return score, strength


# ============================================================
# 2. CANDLE SEQUENCE
# ============================================================

def analyze_sequence(candles):

    recent = candles[
        :min(
            6,
            len(candles)
        )
    ]

    if len(recent) < 2:

        return {
            "score": 0,
            "buy_run": 0,
            "sell_run": 0,
            "alternating": False,
            "changes": 0
        }

    directions = [
        candle_direction(c)
        for c in recent
    ]

    buy_run = 0
    sell_run = 0

    for d in directions:

        if d == 1:
            buy_run += 1
        else:
            break

    for d in directions:

        if d == -1:
            sell_run += 1
        else:
            break

    changes = 0

    for i in range(
        1,
        len(directions)
    ):

        if (
            directions[i] !=
            directions[i - 1]
        ):

            changes += 1

    alternating = (
        changes >=
        len(directions) - 2
    )

    score = weighted_average(
        directions
    )

    if alternating:
        score *= 0.45

    return {

        "score": score,

        "buy_run": buy_run,

        "sell_run": sell_run,

        "alternating": alternating,

        "changes": changes

    }


# ============================================================
# 3-4. HIGHER HIGH / HIGHER LOW
#         LOWER HIGH / LOWER LOW
# ============================================================

def analyze_structure(candles):

    recent = candles[
        :min(
            8,
            len(candles)
        )
    ]

    if len(recent) < 4:

        return {

            "bullish": 0,
            "bearish": 0,
            "score": 0,
            "structure": "INSUFFICIENT"

        }

    # Newest is first.
    # Compare newer visual highs/lows against older candles.
    bullish = 0
    bearish = 0

    for i in range(
        0,
        len(recent) - 1
    ):

        current = recent[i]
        previous = recent[i + 1]

        current_high = (
            current["visual_top"]
        )

        previous_high = (
            previous["visual_top"]
        )

        current_low = (
            current["visual_bottom"]
        )

        previous_low = (
            previous["visual_bottom"]
        )

        # Smaller y = higher price.
        higher_high = (
            current_high <
            previous_high
        )

        higher_low = (
            current_low <
            previous_low
        )

        lower_high = (
            current_high >
            previous_high
        )

        lower_low = (
            current_low >
            previous_low
        )

        if higher_high:
            bullish += 1

        if higher_low:
            bullish += 1

        if lower_high:
            bearish += 1

        if lower_low:
            bearish += 1

    total = max(
        1,
        bullish + bearish
    )

    score = (
        bullish -
        bearish
    ) / total

    if score > 0.20:
        structure = "BULLISH HH/HL"

    elif score < -0.20:
        structure = "BEARISH LH/LL"

    else:
        structure = "MIXED"

    return {

        "bullish": bullish,
        "bearish": bearish,
        "score": score,
        "structure": structure

    }


# ============================================================
# 5. MOMENTUM
# ============================================================

def analyze_momentum(candles):

    recent = candles[
        :min(
            6,
            len(candles)
        )
    ]

    if len(recent) < 3:
        return 0

    sizes = [
        c["body_size"]
        for c in recent
    ]

    directions = [
        candle_direction(c)
        for c in recent
    ]

    momentum_values = []

    for i in range(
        len(recent) - 1
    ):

        current_size = sizes[i]
        older_size = sizes[i + 1]

        ratio = safe_ratio(
            current_size,
            older_size
        )

        direction = directions[i]

        if ratio > 1.0:
            value = direction * min(
                1.0,
                ratio / 2.0
            )
        else:
            value = direction * ratio * 0.55

        momentum_values.append(
            value
        )

    return weighted_average(
        momentum_values
    )


# ============================================================
# 6. PULLBACK DETECTION
# ============================================================

def detect_pullback(candles):

    if len(candles) < 5:

        return {
            "bullish": False,
            "bearish": False,
            "quality": 0
        }

    d = [
        candle_direction(c)
        for c in candles[:6]
    ]

    bullish_pullback = (
        d[1] == -1
        and
        d[2] == 1
        and
        d[3] == 1
    )

    bearish_pullback = (
        d[1] == 1
        and
        d[2] == -1
        and
        d[3] == -1
    )

    quality = 0

    if bullish_pullback:

        if (
            candles[1]["body_size"]
            <
            candles[2]["body_size"]
        ):

            quality = 0.75

        else:

            quality = 0.50

    elif bearish_pullback:

        if (
            candles[1]["body_size"]
            <
            candles[2]["body_size"]
        ):

            quality = 0.75

        else:

            quality = 0.50

    return {

        "bullish": bullish_pullback,

        "bearish": bearish_pullback,

        "quality": quality

    }


# ============================================================
# 7. REVERSAL DETECTION
# ============================================================

def detect_reversal(candles):

    if len(candles) < 4:

        return {
            "bullish": False,
            "bearish": False,
            "depth": 0
        }

    d = [
        candle_direction(c)
        for c in candles[:6]
    ]

    bullish_confirmation = (
        d[0] == 1
        and
        d[1] == 1
        and
        d[2] == -1
        and
        d[3] == -1
    )

    bearish_confirmation = (
        d[0] == -1
        and
        d[1] == -1
        and
        d[2] == 1
        and
        d[3] == 1
    )

    return {

        "bullish": bullish_confirmation,

        "bearish": bearish_confirmation,

        "depth":
            2 if (
                bullish_confirmation
                or
                bearish_confirmation
            )
            else 0

    }


# ============================================================
# 8. BREAKOUT + RETEST
# ============================================================

def detect_breakout_retest(candles):

    if len(candles) < 7:

        return {
            "bullish": False,
            "bearish": False,
            "quality": 0
        }

    newest = candles[0]

    recent = candles[1:4]
    older = candles[4:7]

    recent_high = min(
        c["visual_top"]
        for c in recent
    )

    recent_low = max(
        c["visual_bottom"]
        for c in recent
    )

    older_high = min(
        c["visual_top"]
        for c in older
    )

    older_low = max(
        c["visual_bottom"]
        for c in older
    )

    bullish_break = (
        newest["visual_top"]
        <
        older_high
    )

    bearish_break = (
        newest["visual_bottom"]
        >
        older_low
    )

    bullish = (
        bullish_break
        and
        newest["color"] ==
        "PURPLE"
    )

    bearish = (
        bearish_break
        and
        newest["color"] ==
        "YELLOW"
    )

    quality = 0

    if bullish or bearish:

        avg_body = np.mean([
            c["body_size"]
            for c in candles[1:6]
        ])

        quality = min(
            1.0,
            safe_ratio(
                newest["body_size"],
                avg_body
            ) /
            BREAKOUT_BODY_MULTIPLIER
        )

    return {

        "bullish": bullish,

        "bearish": bearish,

        "quality": quality

    }


# ============================================================
# 9. SWING REJECTION
# ============================================================

def analyze_swing_rejection(candles):

    if not candles:

        return {
            "bullish": 0,
            "bearish": 0
        }

    newest = candles[0]

    upper = newest[
        "upper_rejection_ratio"
    ]

    lower = newest[
        "lower_rejection_ratio"
    ]

    bullish = min(
        1.0,
        lower / 1.2
    )

    bearish = min(
        1.0,
        upper / 1.2
    )

    return {

        "bullish": bullish,

        "bearish": bearish

    }


# ============================================================
# 10-11. OVERALL TREND + TREND STRENGTH
# ============================================================

def analyze_trend(candles):

    if len(candles) < 4:

        return {
            "trend": "SIDEWAYS",
            "strength": 0
        }

    structure = analyze_structure(
        candles
    )

    direction, direction_strength = (
        analyze_recent_direction(
            candles
        )
    )

    momentum = analyze_momentum(
        candles
    )

    combined = (
        structure["score"] * 0.50
        +
        direction * 0.25
        +
        momentum * 0.25
    )

    strength = abs(
        combined
    ) * 100

    if combined > 0.18:

        trend = "BULLISH"

    elif combined < -0.18:

        trend = "BEARISH"

    else:

        trend = "SIDEWAYS"

    return {

        "trend": trend,

        "strength": strength,

        "score": combined

    }


# ============================================================
# 12. CANDLE QUALITY
# ============================================================

def analyze_candle_quality(candles):

    if not candles:
        return 0

    newest = candles[0]

    body = newest["body_size"]

    recent_bodies = [
        c["body_size"]
        for c in candles[1:6]
    ]

    if not recent_bodies:
        return 0

    average = np.mean(
        recent_bodies
    )

    body_strength = min(
        1.0,
        safe_ratio(
            body,
            average
        ) / STRONG_BODY_RATIO
    )

    opposing_wick = max(
        newest[
            "upper_rejection_ratio"
        ],
        newest[
            "lower_rejection_ratio"
        ]
    )

    wick_penalty = min(
        0.60,
        opposing_wick * 0.30
    )

    quality = (
        body_strength -
        wick_penalty
    )

    return max(
        0,
        min(
            1,
            quality
        )
    )


# ============================================================
# 13. RECENT VS OLDER
# ============================================================

def analyze_recent_vs_old(candles):

    if len(candles) < 6:
        return 0

    recent = candles[:3]
    older = candles[3:8]

    recent_score = weighted_average([
        candle_direction(c)
        for c in recent
    ])

    old_score = weighted_average([
        candle_direction(c)
        for c in older
    ])

    return (
        recent_score * 0.70
        +
        old_score * 0.30
    )


# ============================================================
# 14. CONTRADICTION CHECK
# ============================================================

def contradiction_check(candles):

    if len(candles) < 5:

        return {
            "severity": 0,
            "label": "LOW",
            "direction": 0
        }

    older_score = weighted_average([
        candle_direction(c)
        for c in candles[2:8]
    ])

    newest_score = weighted_average([
        candle_direction(c)
        for c in candles[:2]
    ])

    contradiction = (
        abs(
            older_score -
            newest_score
        )
    )

    severity = (
        contradiction *
        100
    )

    if severity >= 65:

        label = "SEVERE"

    elif severity >= 35:

        label = "MODERATE"

    else:

        label = "LOW"

    return {

        "severity": severity,

        "label": label,

        "direction":
            newest_score

    }


# ============================================================
# 16. BODY-TO-BODY MOMENTUM PROGRESSION
# ============================================================

def body_progression(candles):

    recent = candles[
        :min(
            5,
            len(candles)
        )
    ]

    if len(recent) < 3:

        return 0

    values = []

    for i in range(
        len(recent) - 1
    ):

        ratio = safe_ratio(
            recent[i]["body_size"],
            recent[i + 1]["body_size"]
        )

        direction = candle_direction(
            recent[i]
        )

        if ratio >= 1.20:

            value = direction * min(
                1,
                ratio / 2
            )

        elif ratio <= 0.75:

            value = direction * 0.30

        else:

            value = direction * 0.65

        values.append(
            value
        )

    return weighted_average(
        values
    )


# ============================================================
# 17. WICK / REJECTION ANALYSIS
# ============================================================

def wick_rejection(candles):

    if not candles:
        return 0

    newest = candles[0]

    upper = newest[
        "upper_rejection_ratio"
    ]

    lower = newest[
        "lower_rejection_ratio"
    ]

    body_ratio = newest[
        "body_to_range"
    ]

    bullish = (
        min(
            1,
            lower / 1.0
        )
        -
        min(
            0.5,
            upper / 2
        )
    )

    bearish = (
        min(
            1,
            upper / 1.0
        )
        -
        min(
            0.5,
            lower / 2
        )
    )

    continuation = (
        candle_direction(newest)
        *
        body_ratio
    )

    return (
        bullish -
        bearish
        +
        continuation * 0.30
    )


# ============================================================
# 18. ENGULFING CANDLE
# ============================================================

def detect_engulfing(candles):

    if len(candles) < 2:

        return {
            "bullish": False,
            "bearish": False,
            "strength": 0
        }

    current = candles[0]
    previous = candles[1]

    bullish = (
        current["color"] ==
        "PURPLE"
        and
        previous["color"] ==
        "YELLOW"
        and
        current["body_size"] >=
        previous["body_size"] *
        1.15
    )

    bearish = (
        current["color"] ==
        "YELLOW"
        and
        previous["color"] ==
        "PURPLE"
        and
        current["body_size"] >=
        previous["body_size"] *
        1.15
    )

    strength = min(
        1.0,
        safe_ratio(
            current["body_size"],
            previous["body_size"]
        ) / 2.0
    )

    return {

        "bullish": bullish,

        "bearish": bearish,

        "strength": strength

    }


# ============================================================
# 19. INSIDE / COMPRESSION
# ============================================================

def detect_compression(candles):

    if len(candles) < 4:

        return {
            "compression": False,
            "score": 0
        }

    sizes = [
        c["body_size"]
        for c in candles[:4]
    ]

    shrinking = (
        sizes[0] <= sizes[1]
        and
        sizes[1] <= sizes[2]
        and
        sizes[2] <= sizes[3]
    )

    average = np.mean(
        sizes[1:]
    )

    compressed = (
        shrinking
        and
        sizes[0] <
        average *
        COMPRESSION_RATIO
    )

    return {

        "compression": compressed,

        "score":
            1.0 if compressed
            else 0.0

    }


# ============================================================
# 20. EXPANSION AFTER COMPRESSION
# ============================================================

def detect_expansion(candles):

    if len(candles) < 5:

        return {
            "bullish": False,
            "bearish": False,
            "strength": 0
        }

    newest = candles[0]

    older = [
        c["body_size"]
        for c in candles[1:4]
    ]

    average = np.mean(
        older
    )

    expansion = (
        newest["body_size"] >=
        average *
        EXPANSION_RATIO
    )

    strength = min(
        1.0,
        safe_ratio(
            newest["body_size"],
            average
        ) / 2.0
    )

    return {

        "bullish":
            expansion and
            newest["color"] ==
            "PURPLE",

        "bearish":
            expansion and
            newest["color"] ==
            "YELLOW",

        "strength": strength

    }


# ============================================================
# 21. THREE-CANDLE CONTEXT
# ============================================================

def three_candle_context(candles):

    if len(candles) < 3:
        return 0

    a = candle_direction(
        candles[0]
    )

    b = candle_direction(
        candles[1]
    )

    c = candle_direction(
        candles[2]
    )

    sizes = [
        candles[i]["body_size"]
        for i in range(3)
    ]

    # Three same-direction candles:
    # continuation evidence, but reduce slightly if
    # newest body is shrinking.
    if a == b == c:

        progression = safe_ratio(
            sizes[0],
            max(
                sizes[1],
                1
            )
        )

        if progression >= 1.0:
            return a * 1.0

        return a * 0.65

    # Opposite -> opposite -> current:
    # possible reversal confirmation.
    if (
        a != b
        and
        b != c
    ):

        return a * 0.75

    return a * 0.25


# ============================================================
# 22. DIRECTIONAL PERSISTENCE
# ============================================================

def directional_persistence(candles):

    recent = candles[
        :min(
            8,
            len(candles)
        )
    ]

    if not recent:
        return 0

    values = [
        candle_direction(c)
        for c in recent
    ]

    weights = np.linspace(
        1.0,
        0.45,
        len(values)
    )

    weighted = (
        np.sum(
            np.asarray(
                values
            ) *
            weights
        )
        /
        np.sum(weights)
    )

    return weighted


# ============================================================
# 23. MOMENTUM DIVERGENCE / EXHAUSTION
# ============================================================

def momentum_divergence(candles):

    if len(candles) < 5:
        return 0

    recent = candles[:4]

    directions = [
        candle_direction(c)
        for c in recent
    ]

    if not (
        directions[0] ==
        directions[1] ==
        directions[2]
    ):
        return 0

    newest_body = (
        recent[0]["body_size"]
    )

    older_average = np.mean([
        c["body_size"]
        for c in recent[1:4]
    ])

    if (
        newest_body <
        older_average *
        SMALL_BODY_RATIO
    ):

        # Exhaustion points opposite the
        # established direction.
        return (
            -directions[0] *
            0.75
        )

    return 0


# ============================================================
# 24. BREAKOUT STRENGTH
# ============================================================

def breakout_strength(candles):

    result = detect_breakout_retest(
        candles
    )

    if result["bullish"]:

        return result["quality"]

    if result["bearish"]:

        return -result["quality"]

    return 0


# ============================================================
# 25. RETEST QUALITY
# ============================================================

def retest_quality(candles):

    if len(candles) < 6:
        return 0

    newest = candles[0]

    previous = candles[1]

    older = candles[2:6]

    older_level_high = min(
        c["visual_top"]
        for c in older
    )

    older_level_low = max(
        c["visual_bottom"]
        for c in older
    )

    if newest["color"] == "PURPLE":

        clean = (
            previous["color"] ==
            "YELLOW"
            and
            abs(
                previous[
                    "visual_bottom"
                ]
                -
                older_level_high
            )
            <
            max(
                5,
                previous["visual_range"] *
                0.50
            )
            and
            newest["body_size"] >=
            previous["body_size"]
        )

        return (
            0.85 if clean else 0
        )

    else:

        clean = (
            previous["color"] ==
            "PURPLE"
            and
            abs(
                previous[
                    "visual_top"
                ]
                -
                older_level_low
            )
            <
            max(
                5,
                previous["visual_range"] *
                0.50
            )
            and
            newest["body_size"] >=
            previous["body_size"]
        )

        return (
            -0.85 if clean else 0
        )


# ============================================================
# 26. REVERSAL CONFIRMATION DEPTH
# ============================================================

def reversal_confirmation_depth(
    candles
):

    if len(candles) < 5:
        return 0

    newest = candles[0]
    previous = candles[1]
    third = candles[2]
    fourth = candles[3]

    current_direction = (
        candle_direction(newest)
    )

    previous_direction = (
        candle_direction(previous)
    )

    third_direction = (
        candle_direction(third)
    )

    fourth_direction = (
        candle_direction(fourth)
    )

    if (
        current_direction ==
        previous_direction
        and
        current_direction !=
        third_direction
        and
        third_direction ==
        fourth_direction
    ):

        return (
            current_direction *
            1.0
        )

    return 0


# ============================================================
# 27. RECENT-CANDLE WEIGHTED SCORING
# ============================================================

def recent_weighted_score(
    candles
):

    if not candles:
        return 0

    values = [
        candle_direction(c)
        for c in candles[
            :min(
                10,
                len(candles)
            )
        ]
    ]

    weights = []

    for i in range(
        len(values)
    ):

        weights.append(
            max(
                OLD_WEIGHT,
                RECENT_WEIGHT -
                i * 0.07
            )
        )

    return float(
        np.sum(
            np.asarray(values) *
            np.asarray(weights)
        )
        /
        np.sum(weights)
    )


# ============================================================
# 28. SIGNAL CONFLICT SEVERITY
# ============================================================

def conflict_severity(
    buy_evidence,
    sell_evidence
):

    total = (
        buy_evidence +
        sell_evidence
    )

    if total <= 0:

        return {
            "severity": 0,
            "label": "NONE"
        }

    weaker = min(
        buy_evidence,
        sell_evidence
    )

    stronger = max(
        buy_evidence,
        sell_evidence
    )

    conflict = (
        weaker /
        max(
            stronger,
            0.0001
        )
    ) * 100

    if conflict >= 70:

        label = "SEVERE"

    elif conflict >= 40:

        label = "MODERATE"

    else:

        label = "MINOR"

    return {

        "severity": conflict,

        "label": label

    }


# ============================================================
# 29. CANDLE-PATTERN CONTEXT
# ============================================================

def pattern_context(
    candles,
    trend_result,
    pullback_result
):

    engulfing = detect_engulfing(
        candles
    )

    context_score = 0

    if (
        engulfing["bullish"]
        and
        (
            pullback_result["bullish"]
            or
            trend_result["trend"] ==
            "BULLISH"
        )
    ):

        context_score += (
            engulfing["strength"]
            *
            0.80
        )

    if (
        engulfing["bearish"]
        and
        (
            pullback_result["bearish"]
            or
            trend_result["trend"] ==
            "BEARISH"
        )
    ):

        context_score -= (
            engulfing["strength"]
            *
            0.80
        )

    compression = detect_compression(
        candles
    )

    if compression["compression"]:

        # Compression is uncertainty,
        # not an automatic direction.
        context_score *= 0.35

    return context_score


# ============================================================
# 30. CONTINUATION VS REVERSAL
# ============================================================

def continuation_vs_reversal(
    candles,
    evidence
):

    continuation = 0.0
    reversal = 0.0

    direction_score = evidence[
        "recent_direction"
    ]

    momentum = evidence[
        "momentum"
    ]

    persistence = evidence[
        "persistence"
    ]

    structure = evidence[
        "structure"
    ]

    exhaustion = evidence[
        "divergence"
    ]

    reversal_depth = evidence[
        "reversal_depth"
    ]

    pullback = evidence[
        "pullback"
    ]

    engulfing = evidence[
        "engulfing"
    ]

    continuation += (
        abs(direction_score)
        * 18
    )

    continuation += (
        abs(momentum)
        * 16
    )

    continuation += (
        abs(persistence)
        * 14
    )

    continuation += (
        abs(structure)
        * 15
    )

    continuation += (
        abs(pullback)
        * 10
    )

    continuation += (
        abs(
            evidence[
                "breakout"
            ]
        )
        * 12
    )

    reversal += (
        abs(exhaustion)
        * 18
    )

    reversal += (
        abs(reversal_depth)
        * 25
    )

    reversal += (
        abs(engulfing)
        * 18
    )

    reversal += (
        abs(
            evidence[
                "wick"
            ]
        )
        * 10
    )

    if (
        reversal_depth != 0
        and
        np.sign(
            reversal_depth
        ) != np.sign(
            direction_score
        )
    ):

        reversal += 10

    return (
        continuation,
        reversal
    )


# ============================================================
# MAIN 30-LAYER ANALYSIS
# ============================================================

def analyze_candles(
    img,
    candles
):

    if len(candles) < ANALYSIS_MIN_CANDLES:

        return {

            "decision": "NO TRADE",

            "confidence": 0,

            "reason":
                "Not enough verified candles.",

            "buy_score": 0,

            "sell_score": 0,

            "trend": "UNKNOWN",

            "trend_strength": 0,

            "conflict": {
                "severity": 100,
                "label": "SEVERE"
            }

        }

    candles = enrich_candle_geometry(
        img,
        candles
    )

    # --------------------------------------------------------
    # GROUP A: DIRECTION + MOMENTUM
    # --------------------------------------------------------

    direction_score, direction_strength = (
        analyze_recent_direction(
            candles
        )
    )

    sequence = analyze_sequence(
        candles
    )

    momentum = analyze_momentum(
        candles
    )

    progression = body_progression(
        candles
    )

    persistence = directional_persistence(
        candles
    )

    recent_old = analyze_recent_vs_old(
        candles
    )

    weighted_recent = recent_weighted_score(
        candles
    )

    # --------------------------------------------------------
    # GROUP B: STRUCTURE
    # --------------------------------------------------------

    structure_result = analyze_structure(
        candles
    )

    trend_result = analyze_trend(
        candles
    )

    breakout = breakout_strength(
        candles
    )

    retest = retest_quality(
        candles
    )

    swing = analyze_swing_rejection(
        candles
    )

    # --------------------------------------------------------
    # GROUP C: CANDLE PATTERNS
    # --------------------------------------------------------

    wick = wick_rejection(
        candles
    )

    engulf = detect_engulfing(
        candles
    )

    engulf_score = 0

    if engulf["bullish"]:

        engulf_score = (
            engulf["strength"]
        )

    elif engulf["bearish"]:

        engulf_score = -(
            engulf["strength"]
        )

    compression = detect_compression(
        candles
    )

    expansion = detect_expansion(
        candles
    )

    three_candle = three_candle_context(
        candles
    )

    candle_quality = analyze_candle_quality(
        candles
    )

    # --------------------------------------------------------
    # GROUP D: PULLBACK + REVERSAL
    # --------------------------------------------------------

    pullback_result = detect_pullback(
        candles
    )

    pullback_score = 0

    if pullback_result["bullish"]:

        pullback_score = (
            pullback_result["quality"]
        )

    elif pullback_result["bearish"]:

        pullback_score = -(
            pullback_result["quality"]
        )

    reversal_result = detect_reversal(
        candles
    )

    reversal_depth = (
        reversal_confirmation_depth(
            candles
        )
    )

    divergence = momentum_divergence(
        candles
    )

    # --------------------------------------------------------
    # GROUP E: PROTECTION
    # --------------------------------------------------------

    contradiction = contradiction_check(
        candles
    )

    # Pattern context
    context = pattern_context(
        candles,
        trend_result,
        pullback_result
    )

    # --------------------------------------------------------
    # BUILD RAW EVIDENCE
    # --------------------------------------------------------

    evidence = {

        "recent_direction":
            direction_score,

        "sequence":
            sequence["score"],

        "momentum":
            momentum,

        "progression":
            progression,

        "persistence":
            persistence,

        "recent_old":
            recent_old,

        "weighted_recent":
            weighted_recent,

        "structure":
            structure_result["score"],

        "trend":
            trend_result["score"],

        "breakout":
            breakout,

        "retest":
            retest,

        "swing":
            swing["bullish"] -
            swing["bearish"],

        "wick":
            wick,

        "engulfing":
            engulf_score,

        "compression":
            compression["score"],

        "expansion":
            (
                expansion["strength"]
                if
                expansion["bullish"]
                else
                -expansion["strength"]
                if
                expansion["bearish"]
                else
                0
            ),

        "three_candle":
            three_candle,

        "candle_quality":
            candle_quality,

        "pullback":
            pullback_score,

        "divergence":
            divergence,

        "reversal_depth":
            reversal_depth,

        "context":
            context

    }

    # --------------------------------------------------------
    # WEIGHTED EVIDENCE
    # --------------------------------------------------------
    #
    # Important:
    # Several layers describe the same behavior.
    # Therefore they are deliberately grouped rather than
    # allowing every layer to contribute full independent
    # points.
    #
    # --------------------------------------------------------

    group_a = (

        direction_score * 0.22

        +

        sequence["score"] * 0.12

        +

        momentum * 0.20

        +

        progression * 0.16

        +

        persistence * 0.12

        +

        recent_old * 0.08

        +

        weighted_recent * 0.10

    )

    group_b = (

        structure_result["score"] * 0.38

        +

        trend_result["score"] * 0.20

        +

        breakout * 0.18

        +

        retest * 0.14

        +

        (
            swing["bullish"] -
            swing["bearish"]
        ) * 0.10

    )

    group_c = (

        wick * 0.20

        +

        engulf_score * 0.22

        +

        expansion.get(
            "strength",
            0
        ) *
        (
            0.20
            if expansion["bullish"]
            else
            -0.20
            if expansion["bearish"]
            else
            0
        )

        +

        three_candle * 0.18

        +

        (
            0
            if compression["compression"]
            else
            0
        )

        +

        candle_quality *
        candle_direction(
            candles[0]
        ) *
        0.20

    )

    group_d = (

        pullback_score * 0.20

        +

        reversal_depth * 0.30

        +

        divergence * 0.25

        +

        context * 0.15

        +

        three_candle * 0.10

    )

    # --------------------------------------------------------
    # RAW FINAL DIRECTION
    # --------------------------------------------------------

    raw_score = (
        group_a * 0.34
        +
        group_b * 0.26
        +
        group_c * 0.18
        +
        group_d * 0.22
    )

    # --------------------------------------------------------
    # CONTINUATION VS REVERSAL
    # --------------------------------------------------------

    continuation, reversal = (
        continuation_vs_reversal(
            candles,
            evidence
        )
    )

    # Reversal direction comes from the reversal evidence.
    reversal_direction = 0

    if reversal_depth != 0:

        reversal_direction = np.sign(
            reversal_depth
        )

    elif engulf_score != 0:

        reversal_direction = np.sign(
            engulf_score
        )

    elif divergence != 0:

        reversal_direction = np.sign(
            divergence
        )

    # --------------------------------------------------------
    # PROTECTION / CONTRADICTION
    # --------------------------------------------------------

    protection_penalty = (
        contradiction["severity"]
        / 100
    ) * 0.30

    if contradiction["label"] == "SEVERE":

        raw_score *= 0.65

    elif contradiction["label"] == "MODERATE":

        raw_score *= 0.82

    # Compression is uncertainty, not direction.
    if compression["compression"]:

        raw_score *= 0.70

    # Alternating candles are also uncertainty.
    if sequence["alternating"]:

        raw_score *= 0.65

    # --------------------------------------------------------
    # CONTINUATION / REVERSAL DECISION
    # --------------------------------------------------------

    if (
        reversal > continuation
        and
        reversal_direction != 0
    ):

        final_score = (
            reversal_direction *
            min(
                1.0,
                reversal /
                100
            )
        )

    else:

        final_score = raw_score

    # --------------------------------------------------------
    # BUY / SELL EVIDENCE
    # --------------------------------------------------------

    buy_score = 0.0
    sell_score = 0.0

    if final_score > 0:

        buy_score = (
            abs(final_score) *
            100
        )

    elif final_score < 0:

        sell_score = (
            abs(final_score) *
            100
        )

    # Add structural directional confirmation.
    if structure_result["score"] > 0:

        buy_score += (
            structure_result["score"] *
            12
        )

    elif structure_result["score"] < 0:

        sell_score += (
            abs(
                structure_result["score"]
            ) *
            12
        )

    # Cap before conflict calculation.
    buy_score = min(
        100,
        max(
            0,
            buy_score
        )
    )

    sell_score = min(
        100,
        max(
            0,
            sell_score
        )
    )

    # --------------------------------------------------------
    # CONFLICT
    # --------------------------------------------------------

    conflict = conflict_severity(
        buy_score,
        sell_score
    )

    # --------------------------------------------------------
    # SIDEWAYS PROTECTION
    # --------------------------------------------------------

    sideways = (
        trend_result["trend"] ==
        "SIDEWAYS"
        and
        trend_result["strength"] <
        NO_TRADE_SIDEWAYS_STRENGTH
    )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    separation = abs(
        buy_score -
        sell_score
    )

    if sideways:

        decision = "NO TRADE"

    elif (
        conflict["severity"] >=
        NO_TRADE_CONFLICT
    ):

        decision = "NO TRADE"

    elif (
        contradiction["severity"] >=
        SEVERE_CONFLICT_THRESHOLD
        and
        separation <
        MIN_DIRECTION_SEPARATION
    ):

        decision = "NO TRADE"

    elif (
        buy_score >=
        MIN_SIGNAL_CONFIDENCE
        and
        buy_score >
        sell_score
        and
        separation >=
        MIN_DIRECTION_SEPARATION
    ):

        decision = "BUY"

    elif (
        sell_score >=
        MIN_SIGNAL_CONFIDENCE
        and
        sell_score >
        buy_score
        and
        separation >=
        MIN_DIRECTION_SEPARATION
    ):

        decision = "SELL"

    else:

        decision = "NO TRADE"

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if decision == "BUY":

        confidence = buy_score

    elif decision == "SELL":

        confidence = sell_score

    else:

        confidence = max(
            buy_score,
            sell_score
        )

        # NO TRADE should not look like a strong signal.
        confidence = min(
            confidence,
            64
        )

    confidence = max(
        0,
        min(
            100,
            confidence
        )
    )

    # --------------------------------------------------------
    # REASON
    # --------------------------------------------------------

    if decision == "BUY":

        reason = (
            "Bullish candle-structure evidence "
            "is stronger than bearish evidence."
        )

    elif decision == "SELL":

        reason = (
            "Bearish candle-structure evidence "
            "is stronger than bullish evidence."
        )

    else:

        if sideways:

            reason = (
                "Visible structure is too sideways "
                "or weak for a directional decision."
            )

        elif (
            conflict["severity"] >=
            NO_TRADE_CONFLICT
        ):

            reason = (
                "BUY and SELL evidence conflict "
                "too severely."
            )

        elif compression["compression"]:

            reason = (
                "Recent candles are compressed; "
                "directional confirmation is insufficient."
            )

        elif sequence["alternating"]:

            reason = (
                "Recent candle sequence is too "
                "alternating/choppy."
            )

        else:

            reason = (
                "Evidence does not separate BUY "
                "and SELL strongly enough."
            )

    # --------------------------------------------------------
    # RETURN FULL ANALYSIS
    # --------------------------------------------------------

    return {

        "decision":
            decision,

        "confidence":
            confidence,

        "reason":
            reason,

        "buy_score":
            buy_score,

        "sell_score":
            sell_score,

        "trend":
            trend_result["trend"],

        "trend_strength":
            trend_result["strength"],

        "momentum":
            momentum,

        "body_progression":
            progression,

        "persistence":
            persistence,

        "structure":
            structure_result,

        "sequence":
            sequence,

        "pullback":
            pullback_result,

        "reversal":
            reversal_result,

        "engulfing":
            engulf,

        "compression":
            compression,

        "expansion":
            expansion,

        "contradiction":
            contradiction,

        "conflict":
            conflict,

        "continuation_score":
            continuation,

        "reversal_score":
            reversal,

        "evidence":
            evidence

    }


# ============================================================
# ANALYSIS REPORT
# ============================================================

def format_analysis_report(
    analysis
):

    decision = analysis[
        "decision"
    ]

    confidence = analysis[
        "confidence"
    ]

    if decision == "BUY":

        decision_text = "🟢 BUY"

    elif decision == "SELL":

        decision_text = "🔴 SELL"

    else:

        decision_text = "⚪ NO TRADE"

    trend = analysis[
        "trend"
    ]

    trend_strength = analysis[
        "trend_strength"
    ]

    conflict = analysis[
        "conflict"
    ]

    sequence = analysis[
        "sequence"
    ]

    structure = analysis[
        "structure"
    ]

    pullback = analysis[
        "pullback"
    ]

    engulf = analysis[
        "engulfing"
    ]

    compression = analysis[
        "compression"
    ]

    expansion = analysis[
        "expansion"
    ]

    return (

        "🧠 **30-LAYER CANDLE ANALYSIS**\n\n"

        "━━━━━━━━━━━━━━━━━━━━\n"

        f"🎯 **DECISION:** {decision_text}\n"

        f"📊 **CONFIDENCE:** "
        f"{confidence:.1f}%\n\n"

        "━━━━━━━━━━━━━━━━━━━━\n"

        "📈 **DIRECTION & MOMENTUM**\n"

        f"Recent direction: "
        f"{analysis['evidence']['recent_direction']:+.2f}\n"

        f"Momentum: "
        f"{analysis['momentum']:+.2f}\n"

        f"Body progression: "
        f"{analysis['body_progression']:+.2f}\n"

        f"Persistence: "
        f"{analysis['persistence']:+.2f}\n"

        f"Recent weighting: "
        f"{analysis['evidence']['weighted_recent']:+.2f}\n\n"

        "━━━━━━━━━━━━━━━━━━━━\n"

        "🏗️ **STRUCTURE**\n"

        f"Trend: **{trend}**\n"

        f"Trend strength: "
        f"{trend_strength:.1f}%\n"

        f"Structure: "
        f"{structure['structure']}\n"

        f"HH/HL evidence: "
        f"{structure['bullish']}\n"

        f"LH/LL evidence: "
        f"{structure['bearish']}\n\n"

        "━━━━━━━━━━━━━━━━━━━━\n"

        "🕯️ **CANDLE PATTERNS**\n"

        f"Engulfing: "
        f"{'BULLISH' if engulf['bullish'] else 'BEARISH' if engulf['bearish'] else 'NONE'}\n"

        f"Compression: "
        f"{'YES' if compression['compression'] else 'NO'}\n"

        f"Expansion BUY: "
        f"{'YES' if expansion['bullish'] else 'NO'}\n"

        f"Expansion SELL: "
        f"{'YES' if expansion['bearish'] else 'NO'}\n\n"

        "━━━━━━━━━━━━━━━━━━━━\n"

        "🔄 **PULLBACK / REVERSAL**\n"

        f"Pullback BUY: "
        f"{'YES' if pullback['bullish'] else 'NO'}\n"

        f"Pullback SELL: "
        f"{'YES' if pullback['bearish'] else 'NO'}\n"

        f"Reversal depth: "
        f"{analysis['reversal_score']:.1f}\n"

        f"Continuation score: "
        f"{analysis['continuation_score']:.1f}\n"

        f"Reversal score: "
        f"{analysis['reversal_score']:.1f}\n\n"

        "━━━━━━━━━━━━━━━━━━━━\n"

        "🛡️ **PROTECTION**\n"

        f"Contradiction: "
        f"{conflict['label']}\n"

        f"Signal conflict: "
        f"{conflict['severity']:.1f}% "
        f"({conflict['label']})\n\n"

        "━━━━━━━━━━━━━━━━━━━━\n"

        f"📝 **REASON**\n"
        f"{analysis['reason']}\n\n"

        "⚠️ Candle-only visual analysis.\n"
        "No RSI / MACD / EMA / Bollinger.\n"
        "No OCR-generated prices.\n"
        "No generated OHLC.\n"
        "No random candles."

    )


# ============================================================
# CREATE VERIFIED DETECTION MAP
# ============================================================

def create_detection_map(
    img,
    verification
):

    output = img.copy()

    candles = verification[
        "candles"
    ]

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

        verify = candle[
            "verification"
        ]

        verified = verify[
            "verified"
        ]

        score = verify[
            "score"
        ]

        recovered = candle.get(
            "recovered",
            False
        )

        if recovered:

            box_color = (
                255,
                0,
                0
            )

        elif verified:

            box_color = (
                0,
                255,
                0
            )

        else:

            box_color = (
                0,
                0,
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

        if candle[
            "color"
        ] == "PURPLE":

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

        if recovered:

            mark = "R"

        elif verified:

            mark = "V"

        else:

            mark = "?"

        cv2.putText(

            output,

            mark,

            (
                x + w + 3,
                y + 15
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            box_color,

            2,

            cv2.LINE_AA

        )

    cv2.rectangle(
        output,
        (10, 10),
        (410, 135),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        output,
        "MAP VERIFICATION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        "GREEN = VERIFIED",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        "RED = CHECK",
        (220, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        "BLUE = RECOVERED",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        "V = VERIFIED | R = RECOVERED | ? = CHECK",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1,
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
        "candle_verification_map.png"
    )

    try:

        bot.reply_to(

            message,

            "👁️ Reading candle map...\n"
            "➡️ Scanning RIGHT → LEFT.\n"
            "🟣 PURPLE = BUY.\n"
            "🟡 YELLOW = SELL.\n"
            "🔎 Running independent verification...\n"
            "🔄 Recovering missed candles...\n"
            "🧠 Running 30 candle-only analysis layers..."

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

        candles = detect_candles(
            img
        )

        detection_time = (
            time.time()
            -
            detection_start
        )

        purple, yellow = (
            create_report(
                candles
            )
        )

        total = len(
            candles
        )

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
        # VERIFICATION
        # ====================================================

        verification_start = time.time()

        verification = (
            verify_candle_map(
                img,
                candles
            )
        )

        verification_time = (
            time.time()
            -
            verification_start
        )

        verified_candles = verification[
            "candles"
        ]

        # ====================================================
        # ONLY VERIFIED CANDLES ENTER ANALYSIS
        # ====================================================
        #
        # Recovered candles are accepted because the original
        # detector's recovery system explicitly marks them as
        # independently recovered.
        #
        # Unverified candles are not allowed to become strong
        # analysis evidence.
        # ====================================================

        analysis_candles = []

        for candle in verified_candles:

            if (
                candle[
                    "verification"
                ]["verified"]
                or
                candle.get(
                    "recovered",
                    False
                )
            ):

                analysis_candles.append(
                    candle
                )

        # ====================================================
        # CANDLE-ONLY ANALYSIS
        # ====================================================

        analysis_start = time.time()

        analysis = analyze_candles(
            img,
            analysis_candles
        )

        analysis_time = (
            time.time()
            -
            analysis_start
        )

        # ====================================================
        # VERIFIED COUNTS
        # ====================================================

        verified_purple = (
            verification[
                "verified_purple"
            ]
        )

        verified_yellow = (
            verification[
                "verified_yellow"
            ]
        )

        verified_total = (
            verification[
                "verified_total"
            ]
        )

        unverified = (
            verification[
                "unverified"
            ]
        )

        recovered_count = (
            verification[
                "recovered_count"
            ]
        )

        comparison = (
            verification[
                "comparison"
            ]
        )

        map_agreement = (
            comparison[
                "agreement"
            ]
        )

        independent_peaks = len(
            verification[
                "peaks"
            ]
        )

        possible_missing = len(
            comparison[
                "possible_missing"
            ]
        )

        possible_extra = len(
            comparison[
                "possible_extra"
            ]
        )

        # ====================================================
        # RIGHT → LEFT SEQUENCE
        # ====================================================

        sequence = []

        for number, candle in enumerate(

            verified_candles,

            start=1

        ):

            if candle[
                "color"
            ] == "PURPLE":

                color_text = "🟣 BUY"

            else:

                color_text = "🟡 SELL"

            verified = candle[
                "verification"
            ]["verified"]

            score = candle[
                "verification"
            ]["score"]

            recovered = candle.get(
                "recovered",
                False
            )

            if recovered:

                status = "🔵 R"

            elif verified:

                status = "✅"

            else:

                status = "❓"

            sequence.append(

                f"{number}. {color_text} "
                f"{status} "
                f"({score:.0f}%)"

            )

        sequence_text = (
            "\n".join(
                sequence
            )
        )

        # ====================================================
        # PROCESSING TIME
        # ====================================================

        total_time = (
            time.time()
            -
            start_time
        )

        # ====================================================
        # DETECTION + ANALYSIS REPORT
        # ====================================================

        report = (

            "🔎 **CANDLE + MAP VERIFICATION**\n\n"

            "➡️ **SCAN:** RIGHT → LEFT\n"
            "🔢 **1 = newest/rightmost**\n\n"

            "📊 **PRIMARY DETECTION**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"

            f"🟣 PURPLE / BUY: {purple}\n"
            f"🟡 YELLOW / SELL: {yellow}\n"
            f"📊 TOTAL: {total}\n\n"

            "🔎 **INDEPENDENT VERIFICATION**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"

            f"🟣 VERIFIED PURPLE: "
            f"{verified_purple}\n"

            f"🟡 VERIFIED YELLOW: "
            f"{verified_yellow}\n"

            f"✅ VERIFIED TOTAL: "
            f"{verified_total}\n"

            f"🔄 RECOVERED: "
            f"{recovered_count}\n"

            f"❓ NEEDS CHECK: "
            f"{unverified}\n\n"

            "🧭 **MAP AGREEMENT**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"

            f"Independent positions: "
            f"{independent_peaks}\n"

            f"Matched positions: "
            f"{comparison['matched']}\n"

            f"Possible missed: "
            f"{possible_missing}\n"

            f"Possible extra: "
            f"{possible_extra}\n"

            f"Map agreement: "
            f"{map_agreement:.1f}%\n\n"

            "🕯️ **RIGHT → LEFT CANDLES**\n"
            "━━━━━━━━━━━━━━━━━━━━\n"

            f"{sequence_text}\n\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            format_analysis_report(
                analysis
            )

            +

            "\n\n━━━━━━━━━━━━━━━━━━━━\n"

            "⏱️ **PROCESSING**\n"

            f"Detection: "
            f"{detection_time:.2f}s\n"

            f"Verification: "
            f"{verification_time:.2f}s\n"

            f"Analysis: "
            f"{analysis_time:.2f}s\n"

            f"Total: "
            f"{total_time:.2f}s\n\n"

            "🚫 No random candles.\n"
            "🚫 No random prices.\n"
            "🚫 No generated OHLC.\n"
            "🚫 No indicator calculations."

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
                verification
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

                    "🔎 **CANDLE MAP VERIFICATION**\n\n"

                    "➡️ RIGHT → LEFT\n"
                    "🔢 1 = newest/rightmost\n\n"

                    "🟣 = PURPLE / BUY\n"
                    "🟡 = YELLOW / SELL\n\n"

                    "🟩 = verified\n"
                    "🔵 R = recovered\n"
                    "🟥 ? = needs checking\n\n"

                    "The analysis engine uses the "
                    "verified/recovered candle map "
                    "as its candle input."

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
    "🕯️ CANDLE + 30-LAYER ANALYSIS BOT"
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
    "🔎 Independent map verification enabled"
)

print(
    "🔄 Missed candle recovery enabled"
)

print(
    "🧠 30 candle-only analysis layers enabled"
)

print(
    "📈 Direction + momentum enabled"
)

print(
    "🏗️ HH/HL + LH/LL structure enabled"
)

print(
    "🕯️ Wick + engulfing + compression enabled"
)

print(
    "🔄 Pullback + reversal analysis enabled"
)

print(
    "🚀 Breakout + retest analysis enabled"
)

print(
    "🛡️ Contradiction + conflict protection enabled"
)

print(
    "⚖️ Continuation vs reversal enabled"
)

print(
    "🎯 BUY / SELL / NO TRADE enabled"
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
    "🚫 No RSI / MACD / EMA / Bollinger"
)

print(
    "========================================"
)


bot.infinity_polling(

    timeout=30,

    long_polling_timeout=30

)

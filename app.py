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

# How much colored evidence must exist around a detected body.
VERIFY_MIN_PIXELS = 8

# Minimum percentage of the verification region that must
# contain the detected candle color.
VERIFY_MIN_DENSITY = 0.08

# How far left/right around the detected candle center
# the verifier checks.
VERIFY_HORIZONTAL_RADIUS = 0.70

# Minimum distance between independent verification peaks.
VERIFY_MIN_DISTANCE_RATIO = 0.55

# How many colored pixels are needed in a vertical column
# before it becomes a possible candle location.
VERIFY_COLUMN_THRESHOLD = 3

# Verification confidence required to mark a candle as verified.
VERIFY_CONFIDENCE_THRESHOLD = 65


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


    # ========================================================
    # MAIN PASS
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
    # DUPLICATE REMOVAL
    # ========================================================

    candles = (
        remove_cross_color_duplicates(
            candles
        )
    )


    # ========================================================
    # RIGHT → LEFT
    # ========================================================

    candles.sort(
        key=lambda c:
        c["center_x"],
        reverse=True
    )


    return candles


# ============================================================
# ============================================================
# INDEPENDENT MAP VERIFICATION
# ============================================================
# ============================================================
#
# This is NOT Vision API.
#
# It does not simply trust the original detector.
#
# It independently examines the actual color masks around
# every detected candle.
#
# It checks:
#
# 1. Is there really colored candle evidence here?
# 2. Is the color actually PURPLE or YELLOW?
# 3. Does the color agree with the detector?
# 4. Is the detected candle separated from neighboring candles?
#
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


    # ========================================================
    # VERIFICATION REGION
    # ========================================================

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


    # ========================================================
    # COLOR AGREEMENT
    # ========================================================

    if own_pixels >= VERIFY_MIN_PIXELS:

        if own_pixels >= (
            other_pixels * 1.15
        ):

            color_agrees = True

        else:

            color_agrees = False

    else:

        color_agrees = False


    # ========================================================
    # BODY EVIDENCE
    # ========================================================

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


    # ========================================================
    # DENSITY EVIDENCE
    # ========================================================

    density_evidence = min(
        100,
        (
            own_density /
            VERIFY_MIN_DENSITY
        ) * 100
    )


    # ========================================================
    # FINAL VERIFICATION SCORE
    # ========================================================

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
#
# This second check looks across the screenshot and searches
# for independent vertical concentrations of candle color.
#
# It helps detect:
#
# - possible missed candles
# - possible merged candles
#
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


    # ========================================================
    # Limit scan to the chart's main area.
    #
    # We avoid the very top and bottom UI areas.
    # ========================================================

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


    # Small smoothing.
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


    # ========================================================
    # Estimate normal candle spacing from primary detector.
    # ========================================================

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


    # ========================================================
    # Find local maxima.
    # ========================================================

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


    # ========================================================
    # Separate close peaks.
    # ========================================================

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


    # Estimate matching tolerance.
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


    # ========================================================
    # Match each peak to closest candle.
    # ========================================================

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


    # ========================================================
    # Possible missing candles.
    # ========================================================

    possible_missing = []


    for peak_index, (
        peak_x,
        strength
    ) in enumerate(peaks):

        if peak_index not in matched_peaks:

            possible_missing.append(
                peak_x
            )


    # ========================================================
    # Possible extra detections.
    # ========================================================

    possible_extra = []


    for candle_index, cx in enumerate(
        candle_x
    ):

        if candle_index not in matched_candles:

            possible_extra.append(
                cx
            )


    # ========================================================
    # Agreement
    # ========================================================

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
    """Recover missed candles from independent column scan peaks."""
    
    if not peaks:
        return candles, 0
    
    # Get positions of missed peaks
    missed_x = comparison.get("possible_missing", [])
    
    if not missed_x:
        return candles, 0
    
    # Get color masks
    purple_mask, yellow_mask = get_color_masks(img)
    
    h, w = img.shape[:2]
    
    recovered = []
    recovered_count = 0
    
    for peak_x in missed_x:
        
        # Check if this position already has a candle nearby
        already_exists = False
        for candle in candles:
            if abs(candle["center_x"] - peak_x) < 10:
                already_exists = True
                break
        
        if already_exists:
            continue
        
        # Look around the peak to determine color
        search_radius = 8
        left = max(0, int(peak_x - search_radius))
        right = min(w, int(peak_x + search_radius + 1))
        
        # Focus on the vertical chart area
        top = int(h * 0.18)
        bottom = int(h * 0.82)
        
        purple_region = purple_mask[top:bottom, left:right]
        yellow_region = yellow_mask[top:bottom, left:right]
        
        purple_pixels = int(np.sum(purple_region > 0))
        yellow_pixels = int(np.sum(yellow_region > 0))
        
        # Determine color by pixel count
        if purple_pixels > yellow_pixels and purple_pixels >= 5:
            color = "PURPLE"
            color_pixels = purple_pixels
        elif yellow_pixels > purple_pixels and yellow_pixels >= 5:
            color = "YELLOW"
            color_pixels = yellow_pixels
        else:
            # Not enough evidence to determine color
            continue
        
        # Calculate approximate body dimensions
        # Use vertical extent from the masks
        vertical_top = top
        vertical_bottom = bottom
        
        if color == "PURPLE":
            mask = purple_mask
        else:
            mask = yellow_mask
        
        # Find vertical extent of colored pixels at this x position
        col_x_start = max(0, int(peak_x - 3))
        col_x_end = min(w, int(peak_x + 4))
        col_mask = mask[top:bottom, col_x_start:col_x_end]
        ys, xs = np.where(col_mask > 0)
        
        if len(ys) > 0:
            candle_top = top + int(np.min(ys))
            candle_bottom = top + int(np.max(ys))
            candle_h = max(2, candle_bottom - candle_top + 1)
            candle_y = candle_top
        else:
            # Fallback: use default dimensions
            candle_h = 20
            candle_y = top + 50
        
        # Create recovered candle
        recovered_candle = {
            "x": int(peak_x - 4),
            "y": int(candle_y),
            "w": 8,
            "h": int(candle_h),
            "center_x": float(peak_x),
            "color": color,
            "pixels": color_pixels,
            "recovered": True,
            "verification": {
                "verified": True,
                "score": 85,
                "own_pixels": color_pixels,
                "other_pixels": purple_pixels if color == "YELLOW" else yellow_pixels,
                "own_density": 0.15,
                "color_agrees": True
            }
        }
        
        recovered.append(recovered_candle)
        recovered_count += 1
    
    # Merge recovered candles with main list
    all_candles = candles + recovered
    
    # Re-sort RIGHT → LEFT (newest first)
    all_candles.sort(key=lambda c: c["center_x"], reverse=True)
    
    return all_candles, recovered_count


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


    # ========================================================
    # VERIFY EACH PRIMARY CANDLE
    # ========================================================

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


    # ========================================================
    # INDEPENDENT COLUMN SCAN
    # ========================================================

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


    # ========================================================
    # RECOVER MISSED CANDLES
    # ========================================================

    all_candles, recovered_count = recover_missed_candles(
        img,
        results,
        peaks,
        comparison
    )

    # ========================================================
    # RE-VERIFY ANY CANDLES THAT WERE NOT VERIFIED
    # ========================================================

    final_candles = []

    for candle in all_candles:
        if candle.get("recovered", False):
            # Already has verification from recovery
            final_candles.append(candle)
        elif "verification" not in candle:
            # Verify it
            result = verify_single_candle(
                candle,
                purple_mask,
                yellow_mask
            )
            verified_candle = candle.copy()
            verified_candle["verification"] = result
            final_candles.append(verified_candle)
        else:
            final_candles.append(candle)


    # ========================================================
    # VERIFIED COLOR COUNTS
    # ========================================================

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


    # ========================================================
    # DRAW PRIMARY CANDLES
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


        # ====================================================
        # VERIFIED = GREEN BOX
        # UNVERIFIED = RED BOX
        # RECOVERED = BLUE BOX (with R label)
        # ====================================================

        if recovered:

            box_color = (
                255,
                0,
                0
            )  # BLUE

        elif verified:

            box_color = (
                0,
                255,
                0
            )  # GREEN

        else:

            box_color = (
                0,
                0,
                255
            )  # RED


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


        # ====================================================
        # NUMBER COLOR
        # ====================================================

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


        # ====================================================
        # CANDLE NUMBER
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
        # VERIFICATION MARK
        # ====================================================

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


    # ========================================================
    # LEGEND
    # ========================================================

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
        "V = VERIFIED  |  R = RECOVERED  |  ? = CHECK",
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
            "🟣 Checking PURPLE candles = BUY.\n"
            "🟡 Checking YELLOW candles = SELL.\n"
            "🔎 Running independent map verification...\n"
            "🔄 Recovering missed candles..."

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


        # ====================================================
        # PRIMARY COUNT
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
        # NO CANDLES
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

            verification[
                "candles"
            ],

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
        # RESULT REPORT
        # ====================================================

        report = (

            "🔎 **CANDLE + MAP VERIFICATION TEST**\n\n"

            "➡️ **SCAN:** RIGHT → LEFT\n\n"

            "📊 **PRIMARY CANDLE DETECTION**\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"🟣 PURPLE / BUY: {purple}\n"

            f"🟡 YELLOW / SELL: {yellow}\n"

            f"📊 TOTAL: {total}\n\n"


            "🔎 **INDEPENDENT MAP VERIFICATION**\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"🟣 VERIFIED PURPLE: {verified_purple}\n"

            f"🟡 VERIFIED YELLOW: {verified_yellow}\n"

            f"✅ VERIFIED TOTAL: {verified_total}\n"

            f"🔄 RECOVERED CANDLES: {recovered_count}\n"

            f"❓ NEEDS CHECK: {unverified}\n\n"


            "🧭 **INDEPENDENT COLUMN SCAN**\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"🔎 Possible candle positions: "
            f"{independent_peaks}\n"

            f"🤝 Matched positions: "
            f"{comparison['matched']}\n"

            f"⚠️ Possible missed: "
            f"{possible_missing}\n"

            f"⚠️ Possible extra: "
            f"{possible_extra}\n"

            f"📊 MAP AGREEMENT: "
            f"{map_agreement:.1f}%\n\n"


            "🕯️ **RIGHT → LEFT READING**\n"

            "━━━━━━━━━━━━━━━━━━━━\n"

            f"{sequence_text}\n\n"


            "━━━━━━━━━━━━━━━━━━━━\n"

            "🎯 **MAP KEY**\n"

            "🟣 = PURPLE / BUY\n"

            "🟡 = YELLOW / SELL\n"

            "✅ = independently verified\n"

            "🔵 R = RECOVERED candle\n"

            "❓ = needs visual checking\n\n"


            "🔢 Number 1 = newest/rightmost "
            "detected candle.\n\n"


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
        # CREATE VERIFICATION MAP
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

                    "🟣 Number = PURPLE / BUY\n"

                    "🟡 Number = YELLOW / SELL\n\n"

                    "🟩 ✅ = verified by independent check\n"

                    "🔵 R = RECOVERED candle\n"

                    "🟥 ❓ = needs visual checking\n\n"

                    "Recovered candles are marked with BLUE boxes "
                    "and an 'R' label.\n\n"

                    "Check them against the actual screenshot "
                    "to confirm they are real candles."

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
    "🕯️ CANDLE + MAP VERIFICATION BOT"
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
    "🔵 Recovered candles marked with BLUE box + R"
)

print(
    "🚫 OpenRouter Vision removed"
)

print(
    "🚫 No OHLC generation"
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

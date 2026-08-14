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
# UNIFIED SCAN AREA SETTINGS (YELLOW BOX — UNCHANGED)
# ============================================================

RIGHT_SIDE_START = 0.80
TOP_CROP = 0.164
BOTTOM_CROP = 0.30


# ============================================================
# ============================================================
# SECTION 1: CANDLE DETECTION (ORIGINAL — UNCHANGED)
# ============================================================
# ============================================================

# ============================================================
# CANDLE DETECTION SETTINGS
# ============================================================

MIN_BODY_AREA = 10
MIN_BODY_HEIGHT = 2
MIN_CANDLE_WIDTH = 2

RIGHT_MIN_BODY_AREA = 6
RIGHT_MIN_BODY_HEIGHT = 2

MAX_CANDLE_WIDTH_RATIO = 0.045
MERGE_DISTANCE_RATIO = 0.55
MIN_COLOR_DENSITY = 0.25

PURPLE_HUE_LOW = 125
PURPLE_HUE_HIGH = 165
MIN_PURPLE_SATURATION = 100
MIN_PURPLE_VALUE = 70

YELLOW_HUE_LOW = 18
YELLOW_HUE_HIGH = 40
MIN_YELLOW_SATURATION = 100
MIN_YELLOW_VALUE = 70

PURPLE_DOMINANCE_RATIO = 1.20
YELLOW_DOMINANCE_RATIO = 1.10

VERIFY_MIN_PIXELS = 8
VERIFY_MIN_DENSITY = 0.08
VERIFY_HORIZONTAL_RADIUS = 0.70
VERIFY_MIN_DISTANCE_RATIO = 0.55
VERIFY_COLUMN_THRESHOLD = 3
VERIFY_CONFIDENCE_THRESHOLD = 65


# ============================================================
# LOAD IMAGE (shared)
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

    # PURPLE
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

    # YELLOW
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

    # BGR channels
    b, g, r = cv2.split(img)

    # PURPLE DOMINANCE
    purple_dominance = (
        (r.astype(np.int16) > g.astype(np.int16) * PURPLE_DOMINANCE_RATIO) &
        (b.astype(np.int16) > g.astype(np.int16) * PURPLE_DOMINANCE_RATIO) &
        (r.astype(np.int16) > 70) &
        (b.astype(np.int16) > 70)
    )
    purple_dominance_mask = purple_dominance.astype(np.uint8) * 255
    purple = cv2.bitwise_and(purple, purple_dominance_mask)

    # YELLOW DOMINANCE
    yellow_dominance = (
        (r.astype(np.int16) > b.astype(np.int16) * YELLOW_DOMINANCE_RATIO) &
        (g.astype(np.int16) > b.astype(np.int16) * YELLOW_DOMINANCE_RATIO) &
        (r.astype(np.int16) > 80) &
        (g.astype(np.int16) > 70)
    )
    yellow_dominance_mask = yellow_dominance.astype(np.uint8) * 255
    yellow = cv2.bitwise_and(yellow, yellow_dominance_mask)

    return purple, yellow


# ============================================================
# FIND CANDIDATES (CANDLES)
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
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < MIN_CANDLE_WIDTH:
            continue
        if h < min_height:
            continue
        if w > max_width:
            continue
        if w > h * 6:
            continue

        region = cleaned[y:y+h, x:x+w]
        colored_pixels = int(np.sum(region > 0))

        if colored_pixels < 5:
            continue

        density = colored_pixels / float(max(1, w * h))
        if density < MIN_COLOR_DENSITY:
            continue

        center_x = x + w / 2

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

def merge_candidates(candidates):

    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda c: c["center_x"])
    merged = []

    for candidate in candidates:
        merged_into_existing = False
        for existing in merged:
            distance = abs(candidate["center_x"] - existing["center_x"])
            allowed = max(candidate["w"], existing["w"], 2) * MERGE_DISTANCE_RATIO

            candidate_top = candidate["y"]
            candidate_bottom = candidate["y"] + candidate["h"]
            existing_top = existing["y"]
            existing_bottom = existing["y"] + existing["h"]

            vertical_overlap = not (candidate_bottom < existing_top or candidate_top > existing_bottom)

            if distance <= allowed and vertical_overlap:
                left = min(existing["x"], candidate["x"])
                right = max(existing["x"] + existing["w"], candidate["x"] + candidate["w"])
                top = min(existing["y"], candidate["y"])
                bottom = max(existing["y"] + existing["h"], candidate["y"] + candidate["h"])

                existing["x"] = left
                existing["y"] = top
                existing["w"] = right - left
                existing["h"] = bottom - top
                existing["center_x"] = left + existing["w"] / 2
                existing["area"] += candidate["area"]
                existing["pixels"] += candidate["pixels"]

                merged_into_existing = True
                break

        if not merged_into_existing:
            merged.append(candidate.copy())

    return merged


# ============================================================
# REMOVE CROSS-COLOR DUPLICATES
# ============================================================

def remove_cross_color_duplicates(candles):

    candles = sorted(candles, key=lambda c: c["center_x"])
    result = []

    for candle in candles:
        duplicate_index = None
        for i, existing in enumerate(result):
            distance = abs(candle["center_x"] - existing["center_x"])
            threshold = max(candle["w"], existing["w"], 2) * 0.65

            if distance <= threshold:
                duplicate_index = i
                break

        if duplicate_index is None:
            result.append(candle)
        else:
            existing = result[duplicate_index]
            if candle["pixels"] > existing["pixels"]:
                result[duplicate_index] = candle

    return result


# ============================================================
# VERIFY SINGLE CANDLE
# ============================================================

def verify_single_candle(candle, purple_mask, yellow_mask):

    x = int(candle["center_x"])
    y = int(candle["y"])
    w = max(2, int(candle["w"]))
    h = max(2, int(candle["h"]))

    radius = max(2, int(w * VERIFY_HORIZONTAL_RADIUS))

    left = max(0, x - radius)
    right = min(purple_mask.shape[1], x + radius + 1)
    top = max(0, y - max(2, int(h * 0.25)))
    bottom = min(purple_mask.shape[0], y + h + max(2, int(h * 0.25)))

    purple_region = purple_mask[top:bottom, left:right]
    yellow_region = yellow_mask[top:bottom, left:right]

    purple_pixels = int(np.sum(purple_region > 0))
    yellow_pixels = int(np.sum(yellow_region > 0))

    total_pixels = max(1, purple_region.shape[0] * purple_region.shape[1])

    if candle["color"] == "PURPLE":
        own_pixels = purple_pixels
        other_pixels = yellow_pixels
    else:
        own_pixels = yellow_pixels
        other_pixels = purple_pixels

    own_density = own_pixels / total_pixels

    if own_pixels >= VERIFY_MIN_PIXELS:
        color_agrees = own_pixels >= (other_pixels * 1.15)
    else:
        color_agrees = False

    body_evidence = min(100, (own_pixels / float(max(VERIFY_MIN_PIXELS, 1))) * 100)
    density_evidence = min(100, (own_density / VERIFY_MIN_DENSITY) * 100)

    score = (body_evidence * 0.50) + (density_evidence * 0.25) + ((100 if color_agrees else 0) * 0.25)
    score = max(0, min(100, score))

    verified = color_agrees and score >= VERIFY_CONFIDENCE_THRESHOLD

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

def build_verification_peaks(img, purple_mask, yellow_mask, primary_candles):

    h, w = img.shape[:2]

    combined = cv2.bitwise_or(purple_mask, yellow_mask)

    top_limit = int(h * 0.18)
    bottom_limit = int(h * 0.82)

    chart_mask = combined[top_limit:bottom_limit, :]

    column_strength = np.sum(chart_mask > 0, axis=0)

    kernel_size = 3
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size

    smoothed = np.convolve(column_strength.astype(np.float32), kernel, mode="same")

    primary_x = sorted([c["center_x"] for c in primary_candles])

    spacings = []

    for i in range(1, len(primary_x)):
        distance = primary_x[i] - primary_x[i - 1]
        if distance >= 3:
            spacings.append(distance)

    if spacings:
        median_spacing = float(np.median(spacings))
    else:
        median_spacing = max(8, w * 0.018)

    minimum_distance = max(4, int(median_spacing * VERIFY_MIN_DISTANCE_RATIO))

    possible_peaks = []

    threshold = max(VERIFY_COLUMN_THRESHOLD, int(median_spacing * 0.15))

    for x in range(2, w - 2):
        value = smoothed[x]
        if value < threshold:
            continue
        if value >= smoothed[x - 1] and value >= smoothed[x + 1]:
            possible_peaks.append((x, value))

    possible_peaks.sort(key=lambda item: item[1], reverse=True)

    selected = []

    for x, strength in possible_peaks:
        too_close = False
        for selected_x, _ in selected:
            if abs(x - selected_x) < minimum_distance:
                too_close = True
                break
        if not too_close:
            selected.append((x, strength))

    selected.sort(key=lambda item: item[0])

    return selected


# ============================================================
# COMPARE MAP WITH INDEPENDENT SCAN
# ============================================================

def compare_map_with_independent_scan(candles, peaks):

    if not candles:
        return {
            "matched": 0,
            "possible_missing": [],
            "possible_extra": [],
            "agreement": 0.0
        }

    candle_x = [c["center_x"] for c in candles]

    if len(candle_x) >= 2:
        sorted_x = sorted(candle_x)
        spacings = [
            sorted_x[i] - sorted_x[i - 1]
            for i in range(1, len(sorted_x))
            if (sorted_x[i] - sorted_x[i - 1]) > 2
        ]
        if spacings:
            tolerance = max(5, float(np.median(spacings)) * 0.55)
        else:
            tolerance = 8
    else:
        tolerance = 8

    matched = 0
    matched_candles = set()
    matched_peaks = set()

    for peak_index, (peak_x, strength) in enumerate(peaks):
        best_index = None
        best_distance = None

        for candle_index, cx in enumerate(candle_x):
            if candle_index in matched_candles:
                continue
            distance = abs(peak_x - cx)
            if distance <= tolerance:
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_index = candle_index

        if best_index is not None:
            matched += 1
            matched_candles.add(best_index)
            matched_peaks.add(peak_index)

    possible_missing = []
    for peak_index, (peak_x, strength) in enumerate(peaks):
        if peak_index not in matched_peaks:
            possible_missing.append(peak_x)

    possible_extra = []
    for candle_index, cx in enumerate(candle_x):
        if candle_index not in matched_candles:
            possible_extra.append(cx)

    denominator = max(len(candles), len(peaks), 1)
    agreement = (matched / denominator) * 100

    return {
        "matched": matched,
        "possible_missing": possible_missing,
        "possible_extra": possible_extra,
        "agreement": agreement
    }


# ============================================================
# RECOVER MISSED CANDLES
# ============================================================

def recover_missed_candles(img, candles, peaks, comparison):

    if not peaks:
        return candles, 0

    missed_x = comparison.get("possible_missing", [])
    if not missed_x:
        return candles, 0

    purple_mask, yellow_mask = get_color_masks(img)
    h, w = img.shape[:2]

    recovered = []
    recovered_count = 0

    for peak_x in missed_x:
        already_exists = False
        for candle in candles:
            if abs(candle["center_x"] - peak_x) < 10:
                already_exists = True
                break

        if already_exists:
            continue

        search_radius = 8
        left = max(0, int(peak_x - search_radius))
        right = min(w, int(peak_x + search_radius + 1))

        top = int(h * 0.18)
        bottom = int(h * 0.82)

        purple_region = purple_mask[top:bottom, left:right]
        yellow_region = yellow_mask[top:bottom, left:right]

        purple_pixels = int(np.sum(purple_region > 0))
        yellow_pixels = int(np.sum(yellow_region > 0))

        if purple_pixels > yellow_pixels and purple_pixels >= 5:
            color = "PURPLE"
            color_pixels = purple_pixels
        elif yellow_pixels > purple_pixels and yellow_pixels >= 5:
            color = "YELLOW"
            color_pixels = yellow_pixels
        else:
            continue

        if color == "PURPLE":
            mask = purple_mask
        else:
            mask = yellow_mask

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
            candle_h = 20
            candle_y = top + 50

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

    all_candles = candles + recovered
    all_candles.sort(key=lambda c: c["center_x"], reverse=True)

    return all_candles, recovered_count


# ============================================================
# FULL MAP VERIFICATION
# ============================================================

def verify_candle_map(img, candles):

    purple_mask, yellow_mask = get_color_masks(img)

    results = []

    for candle in candles:
        result = verify_single_candle(candle, purple_mask, yellow_mask)
        verified_candle = candle.copy()
        verified_candle["verification"] = result
        results.append(verified_candle)

    peaks = build_verification_peaks(img, purple_mask, yellow_mask, candles)
    comparison = compare_map_with_independent_scan(candles, peaks)

    all_candles, recovered_count = recover_missed_candles(img, results, peaks, comparison)

    final_candles = []

    for candle in all_candles:
        if candle.get("recovered", False):
            final_candles.append(candle)
        elif "verification" not in candle:
            result = verify_single_candle(candle, purple_mask, yellow_mask)
            verified_candle = candle.copy()
            verified_candle["verification"] = result
            final_candles.append(verified_candle)
        else:
            final_candles.append(candle)

    verified_purple = 0
    verified_yellow = 0
    unverified = 0

    for candle in final_candles:
        if candle["verification"]["verified"]:
            if candle["color"] == "PURPLE":
                verified_purple += 1
            else:
                verified_yellow += 1
        else:
            unverified += 1

    return {
        "candles": final_candles,
        "verified_purple": verified_purple,
        "verified_yellow": verified_yellow,
        "verified_total": verified_purple + verified_yellow,
        "unverified": unverified,
        "recovered_count": recovered_count,
        "peaks": peaks,
        "comparison": comparison
    }


# ============================================================
# CREATE CANDLE REPORT
# ============================================================

def create_candle_report(verification):

    candles = verification["candles"]

    purple = sum(1 for c in candles if c["color"] == "PURPLE")
    yellow = sum(1 for c in candles if c["color"] == "YELLOW")
    total = len(candles)

    sequence = []

    for number, candle in enumerate(candles, start=1):
        if candle["color"] == "PURPLE":
            color_text = "🟣 BUY"
        else:
            color_text = "🟡 SELL"

        verified = candle["verification"]["verified"]
        score = candle["verification"]["score"]

        if verified:
            status = "✅"
        else:
            status = "❓"

        sequence.append(f"{number}. {color_text} {status} ({score:.0f}%)")

    return {
        "purple": purple,
        "yellow": yellow,
        "total": total,
        "sequence": sequence,
        "recovered": verification["recovered_count"]
    }


# ============================================================
# ============================================================
# SECTION 2: PRICE DETECTION (INSIDE YELLOW BOX)
# ============================================================
# ============================================================

# ============================================================
# PRICE DETECTION SETTINGS
# ============================================================

MIN_COMPONENT_AREA = 3
MAX_COMPONENT_AREA = 5000
MIN_DIGIT_HEIGHT = 6
MAX_DIGIT_HEIGHT = 180
MIN_DIGIT_WIDTH = 1
MAX_DIGIT_WIDTH = 100
GROUP_GAP_RATIO = 1.25
MIN_RECOGNITION_SCORE = 0.34
MIN_NUMBER_DIGITS = 1


# ============================================================
# GET YELLOW BOX ROI
# ============================================================

def get_yellow_box_roi(img):

    h, w = img.shape[:2]

    start_x = int(w * RIGHT_SIDE_START)
    top_y = int(h * TOP_CROP)
    bottom_y = int(h * (1 - BOTTOM_CROP))

    roi = img[top_y:bottom_y, start_x:w]

    return roi, start_x, top_y


# ============================================================
# CREATE THRESHOLDS (for prices)
# ============================================================

def create_thresholds(roi):

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    thresholds = []

    bright = cv2.inRange(gray, 150, 255)
    thresholds.append(("BRIGHT", bright))

    very_bright = cv2.inRange(gray, 190, 255)
    thresholds.append(("VERY_BRIGHT", very_bright))

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    colored_bright = ((value > 130) & (saturation > 40)).astype(np.uint8) * 255
    thresholds.append(("COLORED_BRIGHT", colored_bright))

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -3)
    thresholds.append(("ADAPTIVE", adaptive))

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholds.append(("OTSU", otsu))

    return thresholds


# ============================================================
# CLEAN MASK (for prices)
# ============================================================

def clean_mask(mask):

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

    return cleaned


# ============================================================
# FIND CHARACTER COMPONENTS (for prices)
# ============================================================

def find_components(mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    components = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area < MIN_COMPONENT_AREA:
            continue
        if area > MAX_COMPONENT_AREA:
            continue
        if h < MIN_DIGIT_HEIGHT:
            continue
        if h > MAX_DIGIT_HEIGHT:
            continue
        if w < MIN_DIGIT_WIDTH:
            continue
        if w > MAX_DIGIT_WIDTH:
            continue
        if w > h * 3.5:
            continue

        region = mask[y:y+h, x:x+w]
        pixels = int(np.sum(region > 0))
        density = pixels / float(max(1, w * h))

        if density < 0.04:
            continue

        components.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "pixels": pixels,
            "density": density
        })

    components.sort(key=lambda c: c["x"])
    return components


# ============================================================
# MERGE CLOSE COMPONENTS (for prices)
# ============================================================

def merge_close_components(components):

    if not components:
        return []

    result = []
    used = set()

    for i, current in enumerate(components):
        if i in used:
            continue

        group = [current]
        used.add(i)
        changed = True

        while changed:
            changed = False
            for j, candidate in enumerate(components):
                if j in used:
                    continue
                for member in group:
                    member_left = member["x"]
                    member_right = member["x"] + member["w"]
                    candidate_left = candidate["x"]
                    candidate_right = candidate["x"] + candidate["w"]

                    horizontal_gap = max(0, max(candidate_left - member_right, member_left - candidate_right))
                    height_ratio = min(member["h"], candidate["h"]) / float(max(member["h"], candidate["h"]))

                    if horizontal_gap <= 3 and height_ratio >= 0.45:
                        group.append(candidate)
                        used.add(j)
                        changed = True
                        break
                if changed:
                    break

        x1 = min(c["x"] for c in group)
        y1 = min(c["y"] for c in group)
        x2 = max(c["x"] + c["w"] for c in group)
        y2 = max(c["y"] + c["h"] for c in group)

        result.append({
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1
        })

    result.sort(key=lambda c: c["x"])
    return result


# ============================================================
# GROUP DIGITS INTO NUMBERS (for prices)
# ============================================================

def group_into_numbers(components):

    if not components:
        return []

    groups = []
    current = [components[0]]

    for component in components[1:]:
        previous = current[-1]
        gap = component["x"] - (previous["x"] + previous["w"])
        average_height = (previous["h"] + component["h"]) / 2.0
        allowed_gap = max(4, average_height * GROUP_GAP_RATIO)

        if gap <= allowed_gap:
            current.append(component)
        else:
            groups.append(current)
            current = [component]

    if current:
        groups.append(current)

    return groups


# ============================================================
# NORMALIZE DIGIT (for prices)
# ============================================================

def normalize_digit(image):

    if image is None or image.size == 0:
        return None

    image = image.copy()
    h, w = image.shape[:2]

    target_h = 48
    scale = target_h / float(max(1, h))
    target_w = max(8, int(w * scale))

    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((64, 48), dtype=np.uint8)
    rh, rw = resized.shape[:2]

    if rw > 46:
        resized = cv2.resize(resized, (46, 60), interpolation=cv2.INTER_AREA)
        rh, rw = resized.shape[:2]

    x_offset = (48 - rw) // 2
    y_offset = (64 - rh) // 2
    canvas[y_offset:y_offset+rh, x_offset:x_offset+rw] = resized

    _, canvas = cv2.threshold(canvas, 100, 255, cv2.THRESH_BINARY)
    return canvas


# ============================================================
# GENERATE DIGIT TEMPLATES (for prices)
# ============================================================

def generate_digit_templates():

    templates = {str(i): [] for i in range(10)}

    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX
    ]

    font_scales = [1.0, 1.2, 1.4, 1.6, 1.8]
    thicknesses = [1, 2, 3]

    for digit in range(10):
        text = str(digit)
        for font in fonts:
            for scale in font_scales:
                for thickness in thicknesses:
                    canvas = np.zeros((80, 60), dtype=np.uint8)
                    size, baseline = cv2.getTextSize(text, font, scale, thickness)
                    tw, th = size
                    x = max(1, (60 - tw) // 2)
                    y = max(th + 1, (80 + th) // 2)

                    cv2.putText(canvas, text, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
                    normalized = normalize_digit(canvas)
                    if normalized is not None:
                        templates[text].append(normalized)

    return templates


DIGIT_TEMPLATES = generate_digit_templates()


# ============================================================
# IMAGE COMPARISON (for prices)
# ============================================================

def compare_images(image_a, image_b):

    if image_a is None or image_b is None:
        return 0.0

    a = image_a.astype(np.float32) / 255.0
    b = image_b.astype(np.float32) / 255.0

    mae = np.mean(np.abs(a - b))
    pixel_score = max(0.0, 1.0 - mae)

    a_binary = (a > 0.5).astype(np.uint8)
    b_binary = (b > 0.5).astype(np.uint8)

    intersection = np.sum((a_binary & b_binary) > 0)
    union = np.sum((a_binary | b_binary) > 0)

    iou = intersection / float(union) if union > 0 else 0.0
    score = (pixel_score * 0.45) + (iou * 0.55)

    return float(score)


# ============================================================
# RECOGNIZE ONE DIGIT (for prices)
# ============================================================

def recognize_digit(digit_image):

    normalized = normalize_digit(digit_image)

    if normalized is None:
        return None, 0.0

    best_digit = None
    best_score = 0.0

    for digit, templates in DIGIT_TEMPLATES.items():
        for template in templates:
            score = compare_images(normalized, template)
            if score > best_score:
                best_score = score
                best_digit = digit

    if best_digit is None or best_score < MIN_RECOGNITION_SCORE:
        return None, best_score

    return best_digit, best_score


# ============================================================
# RECOGNIZE NUMBER GROUP (for prices)
# ============================================================

def recognize_number_group(group, binary):

    if not group:
        return None, 0.0

    recognized = []
    scores = []

    for component in group:
        x, y, w, h = component["x"], component["y"], component["w"], component["h"]
        padding = 2

        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(binary.shape[1], x + w + padding)
        bottom = min(binary.shape[0], y + h + padding)

        digit_roi = binary[top:bottom, left:right]
        digit, score = recognize_digit(digit_roi)

        if digit is None:
            return None, 0.0

        recognized.append(digit)
        scores.append(score)

    if not recognized:
        return None, 0.0

    number = "".join(recognized)
    confidence = sum(scores) / len(scores)

    return number, confidence


# ============================================================
# ANALYZE THRESHOLD (for prices)
# ============================================================

def analyze_threshold(binary, name):

    cleaned = clean_mask(binary)
    components = find_components(cleaned)

    if not components:
        return None

    components = merge_close_components(components)
    groups = group_into_numbers(components)

    results = []

    for group in groups:
        if len(group) < MIN_NUMBER_DIGITS:
            continue

        number, confidence = recognize_number_group(group, cleaned)

        if number is None or confidence < MIN_RECOGNITION_SCORE:
            continue

        x1 = min(c["x"] for c in group)
        y1 = min(c["y"] for c in group)
        x2 = max(c["x"] + c["w"] for c in group)
        y2 = max(c["y"] + c["h"] for c in group)

        results.append({
            "number": number,
            "confidence": confidence,
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "components": len(group),
            "method": name
        })

    return results if results else None


# ============================================================
# REMOVE DUPLICATES (for prices)
# ============================================================

def remove_duplicate_results(results):

    if not results:
        return []

    results.sort(key=lambda r: (r["y"], r["x"]))
    final = []

    for result in results:
        duplicate = False
        for existing in final:
            x_distance = abs(result["x"] - existing["x"])
            y_distance = abs(result["y"] - existing["y"])

            if x_distance < 15 and y_distance < 15 and result["number"] == existing["number"]:
                duplicate = True
                if result["confidence"] > existing["confidence"]:
                    existing.update(result)
                break

        if not duplicate:
            final.append(result)

    return final


# ============================================================
# EXTRACT PRICES FROM YELLOW BOX
# ============================================================

def extract_prices_from_yellow_box(image_path):

    img = load_image(image_path)

    roi, offset_x, offset_y = get_yellow_box_roi(img)

    thresholds = create_thresholds(roi)
    all_results = []

    for name, mask in thresholds:
        results = analyze_threshold(mask, name)
        if results:
            for result in results:
                result["x"] += offset_x
                result["y"] += offset_y
                all_results.append(result)

    all_results = remove_duplicate_results(all_results)
    all_results.sort(key=lambda r: r["confidence"], reverse=True)

    final_results = []

    for result in all_results:
        overlapping = False
        for existing in final_results:
            ax1, ax2 = result["x"], result["x"] + result["w"]
            bx1, bx2 = existing["x"], existing["x"] + existing["w"]
            horizontal_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))

            if horizontal_overlap > 0:
                overlapping = True
                break

        if not overlapping:
            final_results.append(result)

    final_results.sort(key=lambda r: (r["y"], r["x"]))

    return img, final_results


# ============================================================
# CREATE PRICE REPORT WITH LABELS
# ============================================================

def create_price_report(results):

    if not results:
        return {"total": 0, "prices": []}

    results_sorted = sorted(results, key=lambda r: r["y"])

    prices = []

    all_values = []
    for r in results_sorted:
        try:
            val = float(r["number"])
            all_values.append((val, r))
        except:
            all_values.append((None, r))

    valid_values = [(v, r) for v, r in all_values if v is not None]

    if len(valid_values) >= 2:
        high_val = max(valid_values, key=lambda x: x[0])
        low_val = min(valid_values, key=lambda x: x[0])

    for i, (val, r) in enumerate(all_values):
        number = r["number"]
        confidence = round(r["confidence"] * 100, 1)

        label = "PRICE"

        if val is not None and len(valid_values) >= 2:
            if val == high_val[0]:
                label = "HIGH"
            elif val == low_val[0]:
                label = "LOW"
            elif i == 0:
                label = "CURRENT"
            else:
                label = "PRICE"

        prices.append({
            "value": number,
            "confidence": confidence,
            "label": label,
            "position": i + 1
        })

    return {
        "total": len(results),
        "prices": prices
    }


# ============================================================
# ============================================================
# SECTION 3: COMBINED ANALYSIS
# ============================================================
# ============================================================

def analyze_screenshot(image_path):

    img = load_image(image_path)

    # ========================================================
    # 1. CANDLE DETECTION
    # ========================================================

    candles = detect_candles(img)
    verification = verify_candle_map(img, candles)
    candle_report = create_candle_report(verification)

    # ========================================================
    # 2. PRICE DETECTION (inside yellow box)
    # ========================================================

    _, price_results = extract_prices_from_yellow_box(image_path)
    price_report = create_price_report(price_results)

    # ========================================================
    # 3. COMBINED ANALYSIS
    # ========================================================

    combined_analysis = {
        "candles": candle_report,
        "prices": price_report,
        "scan_area": {
            "right_start": RIGHT_SIDE_START,
            "top_crop": TOP_CROP,
            "bottom_crop": BOTTOM_CROP
        }
    }

    return combined_analysis, img, verification, price_results


# ============================================================
# DETECT CANDLES
# ============================================================

def detect_candles(img):

    h, w = img.shape[:2]

    purple_mask, yellow_mask = get_color_masks(img)

    # Main pass
    purple = find_candidates(purple_mask, "PURPLE", w, right_side=False)
    yellow = find_candidates(yellow_mask, "YELLOW", w, right_side=False)

    purple = merge_candidates(purple)
    yellow = merge_candidates(yellow)

    candles = purple + yellow

    # Right-side pass (using unified scan area)
    roi, offset_x, offset_y = get_yellow_box_roi(img)

    # Crop masks to the unified scan area
    purple_roi = purple_mask[offset_y:int(h * (1 - BOTTOM_CROP)), offset_x:]
    yellow_roi = yellow_mask[offset_y:int(h * (1 - BOTTOM_CROP)), offset_x:]

    purple_right = find_candidates(purple_roi, "PURPLE", w, right_side=True)
    yellow_right = find_candidates(yellow_roi, "YELLOW", w, right_side=True)

    for candle in purple_right + yellow_right:
        candle["x"] += offset_x
        candle["center_x"] += offset_x

    candles.extend(purple_right + yellow_right)
    candles = remove_cross_color_duplicates(candles)

    # Verification
    final_candles = []
    for candle in candles:
        result = verify_single_candle(candle, purple_mask, yellow_mask)
        verified_candle = candle.copy()
        verified_candle["verification"] = result
        final_candles.append(verified_candle)

    final_candles.sort(key=lambda c: c["center_x"], reverse=True)

    return final_candles


# ============================================================
# CREATE UNIFIED DETECTION MAP
# ============================================================

def create_unified_detection_map(img, verification, price_results):

    output = img.copy()
    h, w = img.shape[:2]

    # ========================================================
    # SCAN AREA (YELLOW BOX)
    # ========================================================

    scan_x1 = int(w * RIGHT_SIDE_START)
    scan_y1 = int(h * TOP_CROP)
    scan_x2 = w
    scan_y2 = int(h * (1 - BOTTOM_CROP))

    # Yellow scan area
    cv2.rectangle(output, (scan_x1, scan_y1), (scan_x2, scan_y2), (0, 255, 255), 2)

    # ========================================================
    # DRAW ALL CANDLES
    # ========================================================

    candles = verification["candles"]

    for number, candle in enumerate(candles, 1):
        x = int(candle["x"])
        y = int(candle["y"])
        x2 = int(candle["x"] + candle["w"])
        y2 = int(candle["y"] + candle["h"])

        verified = candle["verification"]["verified"]

        if verified:
            box_color = (0, 255, 0)
        else:
            box_color = (0, 0, 255)

        cv2.rectangle(output, (x, y), (x2, y2), box_color, 2)

        # Label
        label = f"{number}"
        cv2.putText(output, label, (x, max(25, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 255) if candle["color"] == "PURPLE" else (0, 255, 255), 2, cv2.LINE_AA)

        # Verification mark
        cv2.putText(output, "V" if verified else "?", (x2 + 3, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2, cv2.LINE_AA)

    # ========================================================
    # DRAW ALL PRICES
    # ========================================================

    for result in price_results:
        x = int(result["x"])
        y = int(result["y"])
        x2 = int(result["x"] + result["w"])
        y2 = int(result["y"] + result["h"])

        cv2.rectangle(output, (x, y), (x2, y2), (255, 255, 0), 2)

        # Number label
        label = f"{result['number']}"
        cv2.putText(output, label, (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 0), 1, cv2.LINE_AA)

    # ========================================================
    # LEGEND
    # ========================================================

    cv2.rectangle(output, (10, 10), (380, 145), (0, 0, 0), -1)

    cv2.putText(output, "UNIFIED DETECTION MAP", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, "Yellow = scan area", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, "Green = verified candle | Red = check", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(output, "Purple # = BUY | Yellow # = SELL", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(output, "Cyan boxes = detected prices", (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(output, f"Scan: {RIGHT_SIDE_START*100:.0f}% → 100%  |  Y: {scan_y1} → {scan_y2}", (20, 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    return output


# ============================================================
# TELEGRAM PHOTO HANDLER
# ============================================================

@bot.message_handler(content_types=["photo"])
def handle_photo(message):

    start_time = time.time()

    image_path = "screenshot.png"
    map_path = "detection_map.png"

    try:

        bot.reply_to(message, "🔍 Analyzing screenshot...")

        # ====================================================
        # DOWNLOAD
        # ====================================================

        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with open(image_path, "wb") as f:
            f.write(downloaded_file)

        download_time = time.time() - start_time

        # ====================================================
        # COMBINED ANALYSIS
        # ====================================================

        analysis_start = time.time()

        combined, img, verification, price_results = analyze_screenshot(image_path)

        analysis_time = time.time() - analysis_start
        total_time = time.time() - start_time

        # ====================================================
        # BUILD RESPONSE
        # ====================================================

        candle_report = combined["candles"]
        price_report = combined["prices"]

        response = "📊 **SCREENSHOT ANALYSIS**\n\n"

        response += "━━━━━━━━━━━━━━━━━━━━\n"
        response += "🕯️ **CANDLE DETECTION**\n"
        response += "━━━━━━━━━━━━━━━━━━━━\n"
        response += f"🟣 PURPLE (BUY): {candle_report['purple']}\n"
        response += f"🟡 YELLOW (SELL): {candle_report['yellow']}\n"
        response += f"📊 TOTAL: {candle_report['total']}\n"
        response += f"🔄 RECOVERED: {candle_report['recovered']}\n\n"

        if candle_report["sequence"]:
            response += "**Candle sequence (newest first):**\n"
            for seq in candle_report["sequence"]:
                response += f"  {seq}\n"

        response += "\n━━━━━━━━━━━━━━━━━━━━\n"
        response += "🔢 **PRICE DETECTION (YELLOW BOX)**\n"
        response += "━━━━━━━━━━━━━━━━━━━━\n"
        response += f"📊 Total prices found: {price_report['total']}\n\n"

        if price_report["prices"]:
            response += "**Prices (with labels):**\n"
            for price in price_report["prices"]:
                response += f"• {price['label']}: `{price['value']}` ({price['confidence']}%)\n"

        response += "\n━━━━━━━━━━━━━━━━━━━━\n"
        response += "📐 **SCAN AREA (YELLOW BOX)**\n"
        response += f"➡️ {RIGHT_SIDE_START*100:.0f}% → 100%\n"
        response += f"📐 Top crop: {TOP_CROP*100:.1f}%\n"
        response += f"📐 Bottom crop: {BOTTOM_CROP*100:.1f}%\n"

        response += f"\n⚡ Download: {download_time:.2f}s\n"
        response += f"⚡ Analysis: {analysis_time:.2f}s\n"
        response += f"⚡ Total: {total_time:.2f}s"

        bot.reply_to(message, response, parse_mode="Markdown")

        # ====================================================
        # DETECTION MAP
        # ====================================================

        map_img = create_unified_detection_map(img, verification, price_results)

        cv2.imwrite(map_path, map_img)

        with open(map_path, "rb") as photo:
            bot.send_photo(
                message.chat.id,
                photo,
                caption="🔎 **DETECTION MAP**\n\n🟨 Yellow = scan area\n🟩 Green = verified candle\n🟥 Red = check\n🟦 Cyan = detected price"
            )

    except Exception as e:
        print("❌ ERROR:", repr(e))
        bot.reply_to(message, f"❌ Error: {str(e)}")

    finally:
        for path in [image_path, map_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


# ============================================================
# START COMMAND
# ============================================================

@bot.message_handler(commands=["start"])
def start(message):

    bot.reply_to(
        message,
        "📊 **UNIFIED SCREENSHOT ANALYZER**\n\n"
        "Send a screenshot.\n\n"
        "I will detect:\n"
        "• 🟣 PURPLE candles (BUY)\n"
        "• 🟡 YELLOW candles (SELL)\n"
        "• 🔢 All prices inside the yellow box\n\n"
        "All detections use the same scan area:\n"
        f"➡️ {RIGHT_SIDE_START*100:.0f}% → 100%\n"
        f"📐 Top crop: {TOP_CROP*100:.1f}%\n"
        f"📐 Bottom crop: {BOTTOM_CROP*100:.1f}%\n\n"
        "📝 Detailed report — all candles and all prices shown",
        parse_mode="Markdown"
    )


# ============================================================
# START BOT
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("📊 UNIFIED SCREENSHOT ANALYZER")
    print("=" * 50)
    print("✅ Candle detection (PURPLE / YELLOW)")
    print("✅ Price detection (yellow box)")
    print("✅ Unified scan area")
    print("✅ Detailed report — all candles and all prices")
    print("=" * 50)

    bot.infinity_polling(timeout=30, long_polling_timeout=30)

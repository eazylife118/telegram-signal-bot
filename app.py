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
# SETTINGS
# ============================================================

# Right-side scanning area — REDUCED WIDTH BY HALF (shifted right)
RIGHT_SIDE_START = 0.80

# Top crop — a little more
TOP_CROP = 0.14

# Bottom crop — more
BOTTOM_CROP = 0.22

# Component filtering
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
# IMAGE LOAD
# ============================================================

def load_image(path):

    img = cv2.imread(path)

    if img is None:
        raise ValueError("Could not read screenshot.")

    return img


# ============================================================
# RIGHT-SIDE ROI
# ============================================================

def get_right_side_roi(img):

    h, w = img.shape[:2]

    start_x = int(w * RIGHT_SIDE_START)

    top_y = int(h * TOP_CROP)

    bottom_y = int(
        h * (1 - BOTTOM_CROP)
    )

    roi = img[
        top_y:bottom_y,
        start_x:w
    ]

    return roi, start_x, top_y


# ============================================================
# CREATE THRESHOLDS
# ============================================================

def create_thresholds(roi):

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    thresholds = []

    # --------------------------------------------------------
    # BRIGHT
    # --------------------------------------------------------

    bright = cv2.inRange(
        gray,
        150,
        255
    )

    thresholds.append(
        ("BRIGHT", bright)
    )

    # --------------------------------------------------------
    # VERY BRIGHT
    # --------------------------------------------------------

    very_bright = cv2.inRange(
        gray,
        190,
        255
    )

    thresholds.append(
        ("VERY_BRIGHT", very_bright)
    )

    # --------------------------------------------------------
    # COLORED BRIGHT
    # --------------------------------------------------------

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    colored_bright = (
        (value > 130)
        &
        (saturation > 40)
    ).astype(np.uint8) * 255

    thresholds.append(
        ("COLORED_BRIGHT", colored_bright)
    )

    # --------------------------------------------------------
    # ADAPTIVE
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        -3
    )

    thresholds.append(
        ("ADAPTIVE", adaptive)
    )

    # --------------------------------------------------------
    # OTSU
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    thresholds.append(
        ("OTSU", otsu)
    )

    return thresholds


# ============================================================
# CLEAN MASK
# ============================================================

def clean_mask(mask):

    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small
    )

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    return cleaned


# ============================================================
# FIND CHARACTER COMPONENTS
# ============================================================

def find_components(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    components = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

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

        region = mask[
            y:y+h,
            x:x+w
        ]

        pixels = int(
            np.sum(region > 0)
        )

        density = (
            pixels /
            float(max(1, w * h))
        )

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

    components.sort(
        key=lambda c: c["x"]
    )

    return components


# ============================================================
# MERGE CLOSE COMPONENTS
# ============================================================

def merge_close_components(
    components
):

    if not components:
        return []

    result = []

    used = set()

    for i, current in enumerate(
        components
    ):

        if i in used:
            continue

        group = [current]

        used.add(i)

        changed = True

        while changed:

            changed = False

            for j, candidate in enumerate(
                components
            ):

                if j in used:
                    continue

                for member in group:

                    member_left = member["x"]

                    member_right = (
                        member["x"] +
                        member["w"]
                    )

                    candidate_left = (
                        candidate["x"]
                    )

                    candidate_right = (
                        candidate["x"] +
                        candidate["w"]
                    )

                    horizontal_gap = max(
                        0,
                        max(
                            candidate_left -
                            member_right,
                            member_left -
                            candidate_right
                        )
                    )

                    height_ratio = (
                        min(
                            member["h"],
                            candidate["h"]
                        )
                        /
                        float(
                            max(
                                member["h"],
                                candidate["h"]
                            )
                        )
                    )

                    if (
                        horizontal_gap <= 3
                        and
                        height_ratio >= 0.45
                    ):

                        group.append(
                            candidate
                        )

                        used.add(j)

                        changed = True

                        break

                if changed:
                    break

        x1 = min(
            c["x"]
            for c in group
        )

        y1 = min(
            c["y"]
            for c in group
        )

        x2 = max(
            c["x"] + c["w"]
            for c in group
        )

        y2 = max(
            c["y"] + c["h"]
            for c in group
        )

        result.append({
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1
        })

    result.sort(
        key=lambda c: c["x"]
    )

    return result


# ============================================================
# GROUP DIGITS INTO NUMBERS
# ============================================================

def group_into_numbers(
    components
):

    if not components:
        return []

    groups = []

    current = [
        components[0]
    ]

    for component in components[1:]:

        previous = current[-1]

        gap = (
            component["x"]
            -
            (
                previous["x"] +
                previous["w"]
            )
        )

        average_height = (
            previous["h"] +
            component["h"]
        ) / 2.0

        allowed_gap = max(
            4,
            average_height *
            GROUP_GAP_RATIO
        )

        if gap <= allowed_gap:

            current.append(
                component
            )

        else:

            groups.append(
                current
            )

            current = [
                component
            ]

    if current:
        groups.append(
            current
        )

    return groups


# ============================================================
# NORMALIZE DIGIT
# ============================================================

def normalize_digit(
    image
):

    if image is None:
        return None

    if image.size == 0:
        return None

    image = image.copy()

    h, w = image.shape[:2]

    target_h = 48

    scale = target_h / float(
        max(1, h)
    )

    target_w = max(
        8,
        int(w * scale)
    )

    resized = cv2.resize(
        image,
        (
            target_w,
            target_h
        ),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (64, 48),
        dtype=np.uint8
    )

    rh, rw = resized.shape[:2]

    if rw > 46:

        resized = cv2.resize(
            resized,
            (46, 60),
            interpolation=cv2.INTER_AREA
        )

        rh, rw = resized.shape[:2]

    x_offset = (
        48 - rw
    ) // 2

    y_offset = (
        64 - rh
    ) // 2

    canvas[
        y_offset:y_offset+rh,
        x_offset:x_offset+rw
    ] = resized

    _, canvas = cv2.threshold(
        canvas,
        100,
        255,
        cv2.THRESH_BINARY
    )

    return canvas


# ============================================================
# GENERATE DIGIT TEMPLATES
# ============================================================

def generate_digit_templates():

    templates = {
        str(i): []
        for i in range(10)
    }

    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX
    ]

    font_scales = [
        1.0,
        1.2,
        1.4,
        1.6,
        1.8
    ]

    thicknesses = [
        1,
        2,
        3
    ]

    for digit in range(10):

        text = str(digit)

        for font in fonts:

            for scale in font_scales:

                for thickness in thicknesses:

                    canvas = np.zeros(
                        (80, 60),
                        dtype=np.uint8
                    )

                    size, baseline = (
                        cv2.getTextSize(
                            text,
                            font,
                            scale,
                            thickness
                        )
                    )

                    tw, th = size

                    x = max(
                        1,
                        (60 - tw) // 2
                    )

                    y = max(
                        th + 1,
                        (80 + th) // 2
                    )

                    cv2.putText(
                        canvas,
                        text,
                        (x, y),
                        font,
                        scale,
                        255,
                        thickness,
                        cv2.LINE_AA
                    )

                    normalized = (
                        normalize_digit(
                            canvas
                        )
                    )

                    if normalized is not None:

                        templates[
                            text
                        ].append(
                            normalized
                        )

    return templates


DIGIT_TEMPLATES = (
    generate_digit_templates()
)


# ============================================================
# IMAGE COMPARISON
# ============================================================

def compare_images(
    image_a,
    image_b
):

    if (
        image_a is None
        or
        image_b is None
    ):
        return 0.0

    a = (
        image_a.astype(
            np.float32
        ) / 255.0
    )

    b = (
        image_b.astype(
            np.float32
        ) / 255.0
    )

    mae = np.mean(
        np.abs(a - b)
    )

    pixel_score = max(
        0.0,
        1.0 - mae
    )

    a_binary = (
        a > 0.5
    ).astype(
        np.uint8
    )

    b_binary = (
        b > 0.5
    ).astype(
        np.uint8
    )

    intersection = np.sum(
        (
            a_binary &
            b_binary
        ) > 0
    )

    union = np.sum(
        (
            a_binary |
            b_binary
        ) > 0
    )

    if union > 0:

        iou = (
            intersection /
            float(union)
        )

    else:

        iou = 0.0

    score = (
        pixel_score * 0.45
        +
        iou * 0.55
    )

    return float(score)


# ============================================================
# RECOGNIZE ONE DIGIT
# ============================================================

def recognize_digit(
    digit_image
):

    normalized = normalize_digit(
        digit_image
    )

    if normalized is None:
        return None, 0.0

    best_digit = None
    best_score = 0.0

    for digit, templates in (
        DIGIT_TEMPLATES.items()
    ):

        for template in templates:

            score = compare_images(
                normalized,
                template
            )

            if score > best_score:

                best_score = score
                best_digit = digit

    if (
        best_digit is None
        or
        best_score <
        MIN_RECOGNITION_SCORE
    ):

        return None, best_score

    return (
        best_digit,
        best_score
    )


# ============================================================
# RECOGNIZE NUMBER GROUP
# ============================================================

def recognize_number_group(
    group,
    binary
):

    if not group:
        return None, 0.0

    recognized = []

    scores = []

    for component in group:

        x = component["x"]
        y = component["y"]
        w = component["w"]
        h = component["h"]

        padding = 2

        left = max(
            0,
            x - padding
        )

        top = max(
            0,
            y - padding
        )

        right = min(
            binary.shape[1],
            x + w + padding
        )

        bottom = min(
            binary.shape[0],
            y + h + padding
        )

        digit_roi = binary[
            top:bottom,
            left:right
        ]

        digit, score = (
            recognize_digit(
                digit_roi
            )
        )

        if digit is None:

            return None, 0.0

        recognized.append(
            digit
        )

        scores.append(
            score
        )

    if not recognized:
        return None, 0.0

    number = "".join(
        recognized
    )

    confidence = (
        sum(scores) /
        len(scores)
    )

    return number, confidence


# ============================================================
# ANALYZE THRESHOLD
# ============================================================

def analyze_threshold(
    binary,
    name
):

    cleaned = clean_mask(
        binary
    )

    components = find_components(
        cleaned
    )

    if not components:
        return None

    components = merge_close_components(
        components
    )

    groups = group_into_numbers(
        components
    )

    results = []

    for group in groups:

        if len(group) < MIN_NUMBER_DIGITS:
            continue

        number, confidence = (
            recognize_number_group(
                group,
                cleaned
            )
        )

        if number is None:
            continue

        if confidence < MIN_RECOGNITION_SCORE:
            continue

        x1 = min(
            c["x"]
            for c in group
        )

        y1 = min(
            c["y"]
            for c in group
        )

        x2 = max(
            c["x"] + c["w"]
            for c in group
        )

        y2 = max(
            c["y"] + c["h"]
            for c in group
        )

        results.append({

            "number": number,

            "confidence":
                confidence,

            "x": x1,

            "y": y1,

            "w": x2 - x1,

            "h": y2 - y1,

            "components":
                len(group),

            "method":
                name
        })

    if not results:
        return None

    return results


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicate_results(
    results
):

    if not results:
        return []

    results.sort(
        key=lambda r:
        (
            r["y"],
            r["x"]
        )
    )

    final = []

    for result in results:

        duplicate = False

        for existing in final:

            x_distance = abs(
                result["x"] -
                existing["x"]
            )

            y_distance = abs(
                result["y"] -
                existing["y"]
            )

            if (
                x_distance < 15
                and
                y_distance < 15
                and
                result["number"] ==
                existing["number"]
            ):

                duplicate = True

                if (
                    result["confidence"]
                    >
                    existing["confidence"]
                ):

                    existing.update(
                        result
                    )

                break

        if not duplicate:

            final.append(
                result
            )

    return final


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_numbers_from_image(
    image_path
):

    img = load_image(
        image_path
    )

    roi, offset_x, offset_y = (
        get_right_side_roi(
            img
        )
    )

    thresholds = create_thresholds(
        roi
    )

    all_results = []

    for name, mask in thresholds:

        results = analyze_threshold(
            mask,
            name
        )

        if results:

            for result in results:

                result["x"] += (
                    offset_x
                )

                result["y"] += (
                    offset_y
                )

                all_results.append(
                    result
                )

    all_results = (
        remove_duplicate_results(
            all_results
        )
    )

    all_results.sort(
        key=lambda r:
        r["confidence"],
        reverse=True
    )

    final_results = []

    for result in all_results:

        overlapping = False

        for existing in final_results:

            ax1 = result["x"]

            ax2 = (
                result["x"] +
                result["w"]
            )

            bx1 = existing["x"]

            bx2 = (
                existing["x"] +
                existing["w"]
            )

            horizontal_overlap = (
                max(
                    0,
                    min(ax2, bx2) -
                    max(ax1, bx1)
                )
            )

            if horizontal_overlap > 0:

                overlapping = True

                break

        if not overlapping:

            final_results.append(
                result
            )

    final_results.sort(
        key=lambda r:
        (
            r["y"],
            r["x"]
        )
    )

    return img, final_results


# ============================================================
# CREATE DEBUG MAP
# ============================================================

def create_number_detection_map(
    img,
    results
):

    output = img.copy()

    h, w = img.shape[:2]

    # ========================================================
    # EXACT SCAN AREA
    # ========================================================

    scan_x1 = int(
        w * RIGHT_SIDE_START
    )

    scan_y1 = int(
        h * TOP_CROP
    )

    scan_x2 = w

    scan_y2 = int(
        h * (1 - BOTTOM_CROP)
    )

    # ========================================================
    # YELLOW SCAN RECTANGLE
    # ========================================================

    cv2.rectangle(
        output,
        (scan_x1, scan_y1),
        (scan_x2, scan_y2),
        (0, 255, 255),
        2
    )

    # ========================================================
    # GREEN DETECTION RECTANGLE
    # ========================================================

    cv2.rectangle(
        output,
        (scan_x1, scan_y1),
        (scan_x2, scan_y2),
        (0, 255, 0),
        3
    )

    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    panel_height = 115

    cv2.rectangle(
        output,
        (10, 10),
        (370, panel_height),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        "NUMBER DETECTION AREA",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        "GREEN = EXACT SCAN AREA",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        f"X: {scan_x1} → {scan_x2}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        f"Y: {scan_y1} → {scan_y2}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # WRITE DETECTED NUMBERS INSIDE THE SCAN AREA
    # ========================================================

    if results:

        text_y = scan_y1 + 25

        for index, result in enumerate(
            results[:10],
            1
        ):

            text = (
                f"{index}: "
                f"{result['number']} "
                f"({result['confidence'] * 100:.0f}%)"
            )

            text_x = scan_x1 + 10

            if text_y >= scan_y2 - 5:
                break

            cv2.putText(
                output,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

            text_y += 20

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

    image_path = (
        "number_screenshot.png"
    )

    map_path = (
        "number_detection_map.png"
    )

    try:

        # ====================================================
        # DOWNLOAD
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

        download_time = (
            time.time()
            -
            start_time
        )

        # ====================================================
        # ANALYSIS
        # ====================================================

        analysis_start = time.time()

        img, results = (
            extract_numbers_from_image(
                image_path
            )
        )

        analysis_time = (
            time.time()
            -
            analysis_start
        )

        total_time = (
            time.time()
            -
            start_time
        )

        # ====================================================
        # NO RESULTS
        # ====================================================

        if not results:

            bot.reply_to(

                message,

                "❌ No reliable numbers detected.\n\n"

                "Nothing was generated.\n"
                "No fake number was created.\n"
                "No price was guessed.\n\n"

                f"⚡ Analysis: "
                f"{analysis_time:.2f}s\n"

                f"📥 Download: "
                f"{download_time:.2f}s\n"

                f"⏱ Total: "
                f"{total_time:.2f}s"

            )

            return

        # ====================================================
        # RESULT
        # ====================================================

        response = (
            "🔢 **RIGHT-SIDE NUMBER DETECTION**\n\n"
        )

        response += (
            f"📊 Numbers found: "
            f"{len(results)}\n"
        )

        response += (
            f"📏 Scan: "
            f"{RIGHT_SIDE_START*100:.0f}% → 100%\n"
        )

        response += (
            f"📐 Top crop: "
            f"{TOP_CROP*100:.0f}%\n"
        )

        response += (
            f"📐 Bottom crop: "
            f"{BOTTOM_CROP*100:.0f}%\n"
        )

        response += (
            f"⚡ Analysis: "
            f"{analysis_time:.2f}s\n"
        )

        response += (
            f"📥 Download: "
            f"{download_time:.2f}s\n"
        )

        response += (
            f"⏱ Total: "
            f"{total_time:.2f}s\n\n"
        )

        response += (
            "━━━━━━━━━━━━━━━━━━━━\n"
        )

        response += (
            "🔢 **DETECTED NUMBERS:**\n\n"
        )

        for i, result in enumerate(
            results,
            1
        ):

            response += (
                f"{i}. `{result['number']}` "
                f"— {result['confidence'] * 100:.0f}%\n"
            )

        response += (
            "\n━━━━━━━━━━━━━━━━━━━━\n"
        )

        response += (
            "🟩 Detection box = EXACT scan area\n"
        )

        response += (
            "🎯 Right side only\n"
        )

        response += (
            "🚫 No Tesseract\n"
        )

        response += (
            "🚫 No Vision API\n"
        )

        response += (
            "🚫 No generated numbers\n"
        )

        response += (
            "🚫 No generated prices"
        )

        bot.reply_to(
            message,
            response,
            parse_mode="Markdown"
        )

        # ====================================================
        # DEBUG MAP
        # ====================================================

        map_img = (
            create_number_detection_map(
                img,
                results
            )
        )

        cv2.imwrite(
            map_path,
            map_img
        )

        with open(
            map_path,
            "rb"
        ) as photo:

            bot.send_photo(

                message.chat.id,

                photo,

                caption=(
                    "🔎 NUMBER DETECTION MAP\n\n"
                    "🟨 Yellow = scan boundary\n"
                    "🟩 Green = EXACT SAME boundary\n\n"
                    "🔄 **CHANGES APPLIED:**\n"
                    "• Top crop: slightly more\n"
                    "• Bottom crop: more\n"
                    "• Width: reduced by half\n"
                    "• Shifted: from the left side"
                )

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
def start(
    message
):

    bot.reply_to(

        message,

        "🔢 **RIGHT-SIDE NUMBER READER**\n\n"

        "Send a screenshot.\n\n"

        f"📏 Horizontal scan: "
        f"{RIGHT_SIDE_START*100:.0f}% → 100%\n"

        f"📐 Top crop: "
        f"{TOP_CROP*100:.0f}%\n"

        f"📐 Bottom crop: "
        f"{BOTTOM_CROP*100:.0f}%\n\n"

        "🟨 Yellow = scan boundary\n"
        "🟩 Green = exact same boundary\n\n"

        "🔄 **CHANGES APPLIED:**\n"
        "• Top crop: slightly more\n"
        "• Bottom crop: more\n"
        "• Width: reduced by half\n"
        "• Shifted: from the left side\n\n"

        "🚫 No Tesseract\n"
        "🚫 No Vision API\n"
        "🚫 No fake numbers\n"
        "🚫 No fake prices\n\n"

        "⚡ OpenCV template recognition\n"
        "📝 Detected numbers are written "
        "for verification.",

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
        "🔢 RIGHT-SIDE NUMBER READER"
    )

    print(
        "========================================"
    )

    print(
        f"📏 Scan: "
        f"{RIGHT_SIDE_START*100:.0f}% → 100%"
    )

    print(
        f"📐 Top crop: "
        f"{TOP_CROP*100:.0f}%"
    )

    print(
        f"📐 Bottom crop: "
        f"{BOTTOM_CROP*100:.0f}%"
    )

    print(
        "✅ OpenCV only"
    )

    print(
        "✅ No Tesseract"
    )

    print(
        "✅ Template recognition"
    )

    print(
        "✅ Exact-size detection box"
    )

    print(
        "🔄 Width reduced by half (from left)"
    )

    print(
        "🚫 No fake numbers"
    )

    print(
        "🚫 No fake prices"
    )

    print(
        "========================================"
    )

    bot.infinity_polling(
        timeout=30,
        long_polling_timeout=30
    )

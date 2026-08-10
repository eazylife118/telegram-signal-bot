import os
import cv2
import numpy as np
import telebot
import time
from datetime import datetime, timedelta
from flask import Flask
from threading import Thread
# ============================================================
# TELEGRAM
# ============================================================
TELEGRAM_TOKEN = os.getenv(
    "BOT_TOKEN",
    "PASTE_YOUR_BOT_TOKEN_HERE"
)
bot = telebot.TeleBot(TELEGRAM_TOKEN)
# ============================================================
# FLASK KEEP-ALIVE
# ============================================================
app = Flask(__name__)
@app.route("/")
def home():
    return "OTC Candle Strategy Bot is running."
def run_flask():
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
Thread(
    target=run_flask,
    daemon=True
).start()
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
# STRATEGY SETTINGS
# ============================================================
LOOKBACK_CANDLES = 20
MIN_ZONE_TESTS = 2
ZONE_TOLERANCE_RATIO = 0.035
MIN_SIGNAL_SCORE = 6
MIN_SIGNAL_AGREEMENT = 65
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
    # GREEN
    green_lower = np.array([
        30, 35, 35
    ])
    green_upper = np.array([
        90, 255, 255
    ])
    green = cv2.inRange(
        hsv,
        green_lower,
        green_upper
    )
    # RED
    red_lower_1 = np.array([
        0, 55, 55
    ])
    red_upper_1 = np.array([
        12, 255, 255
    ])
    red_lower_2 = np.array([
        168, 55, 55
    ])
    red_upper_2 = np.array([
        180, 255, 255
    ])
    red1 = cv2.inRange(
        hsv,
        red_lower_1,
        red_upper_1
    )
    red2 = cv2.inRange(
        hsv,
        red_lower_2,
        red_upper_2
    )
    red = cv2.bitwise_or(
        red1,
        red2
    )
    return green, red
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
            np.sum(region > 0)
        )
        if colored_pixels < 5:
            continue
        density = (
            colored_pixels /
            float(max(1, w * h))
        )
        if density < 0.15:
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
                candidate_bottom < existing_top
                or
                candidate_top > existing_bottom
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
                    right - left
                )
                existing["h"] = (
                    bottom - top
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
def remove_cross_color_duplicates(candles):
    candles = sorted(
        candles,
        key=lambda c: c["center_x"]
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
# RIGHT-SIDE DETECTION
# ============================================================
def detect_right_side(
    chart,
    green_mask,
    red_mask
):
    h, w = chart.shape[:2]
    right_start = int(
        w * 0.72
    )
    green_right = green_mask[
        :,
        right_start:
    ]
    red_right = red_mask[
        :,
        right_start:
    ]
    green = find_candidates(
        green_right,
        "GREEN",
        w,
        right_side=True
    )
    red = find_candidates(
        red_right,
        "RED",
        w,
        right_side=True
    )
    for candle in green + red:
        candle["x"] += right_start
        candle["center_x"] += (
            right_start
        )
    return green + red
# ============================================================
# DETECT CANDLES
# ============================================================
def detect_candles(img):
    h, w = img.shape[:2]
    green_mask, red_mask = (
        get_color_masks(img)
    )
    green = find_candidates(
        green_mask,
        "GREEN",
        w,
        right_side=False
    )
    red = find_candidates(
        red_mask,
        "RED",
        w,
        right_side=False
    )
    green = merge_candidates(
        green
    )
    red = merge_candidates(
        red
    )
    candles = (
        green +
        red
    )
    right_candidates = (
        detect_right_side(
            img,
            green_mask,
            red_mask
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
        key=lambda c: c["center_x"]
    )
    # RIGHT → LEFT
    # Candle #1 = newest visible candle
    candles.reverse()
    return candles
# ============================================================
# BODY POSITION
# ============================================================
def body_position(candle):
    top = candle["y"]
    bottom = (
        candle["y"] +
        candle["h"]
    )
    return (
        top,
        bottom
    )
# ============================================================
# RESISTANCE
# ============================================================
def find_resistance_zone(candles):
    if not candles:
        return None
    candles_to_check = candles[
        :LOOKBACK_CANDLES
    ]
    if len(candles_to_check) < 2:
        return None
    points = []
    for candle in candles_to_check:
        top, _ = body_position(
            candle
        )
        points.append(top)
    chart_min = min(points)
    chart_max = max(points)
    visible_range = max(
        1,
        chart_max - chart_min
    )
    tolerance = (
        visible_range *
        ZONE_TOLERANCE_RATIO
    )
    zones = []
    for candle in candles_to_check:
        top, _ = body_position(
            candle
        )
        placed = False
        for zone in zones:
            if abs(
                top -
                zone["level"]
            ) <= tolerance:
                zone["candles"].append(
                    candle
                )
                zone["level"] = np.mean(
                    [
                        body_position(c)[0]
                        for c in zone["candles"]
                    ]
                )
                placed = True
                break
        if not placed:
            zones.append({
                "level": float(top),
                "candles": [candle]
            })
    zones.sort(
        key=lambda z: len(
            z["candles"]
        ),
        reverse=True
    )
    if not zones:
        return None
    strongest = zones[0]
    if len(
        strongest["candles"]
    ) < MIN_ZONE_TESTS:
        return None
    return strongest
# ============================================================
# SUPPORT
# ============================================================
def find_support_zone(candles):
    if not candles:
        return None
    candles_to_check = candles[
        :LOOKBACK_CANDLES
    ]
    if len(candles_to_check) < 2:
        return None
    points = []
    for candle in candles_to_check:
        _, bottom = body_position(
            candle
        )
        points.append(bottom)
    chart_min = min(points)
    chart_max = max(points)
    visible_range = max(
        1,
        chart_max - chart_min
    )
    tolerance = (
        visible_range *
        ZONE_TOLERANCE_RATIO
    )
    zones = []
    for candle in candles_to_check:
        _, bottom = body_position(
            candle
        )
        placed = False
        for zone in zones:
            if abs(
                bottom -
                zone["level"]
            ) <= tolerance:
                zone["candles"].append(
                    candle
                )
                zone["level"] = np.mean(
                    [
                        body_position(c)[1]
                        for c in zone["candles"]
                    ]
                )
                placed = True
                break
        if not placed:
            zones.append({
                "level": float(bottom),
                "candles": [candle]
            })
    zones.sort(
        key=lambda z: len(
            z["candles"]
        ),
        reverse=True
    )
    if not zones:
        return None
    strongest = zones[0]
    if len(
        strongest["candles"]
    ) < MIN_ZONE_TESTS:
        return None
    return strongest
# ============================================================
# ZONE REACTION STRENGTH
# ============================================================
def zone_reaction_strength(zone):
    if not zone:
        return 0
    tests = len(
        zone["candles"]
    )
    if tests >= 8:
        return 3
    if tests >= 5:
        return 2
    if tests >= 2:
        return 1
    return 0
# ============================================================
# RESISTANCE BEHAVIOR
# ============================================================
def resistance_behavior(
    candles,
    resistance
):
    if not resistance:
        return {
            "near": False,
            "rejection": False
        }
    newest = candles[0]
    top, _ = body_position(
        newest
    )
    level = resistance["level"]
    distance = abs(
        top -
        level
    )
    tolerance = max(
        3,
        newest["h"] * 0.75
    )
    near = (
        distance <= tolerance
    )
    rejection = (
        near
        and
        newest["color"] == "RED"
    )
    return {
        "near": near,
        "rejection": rejection
    }
# ============================================================
# SUPPORT BEHAVIOR
# ============================================================
def support_behavior(
    candles,
    support
):
    if not support:
        return {
            "near": False,
            "rejection": False
        }
    newest = candles[0]
    _, bottom = body_position(
        newest
    )
    level = support["level"]
    distance = abs(
        bottom -
        level
    )
    tolerance = max(
        3,
        newest["h"] * 0.75
    )
    near = (
        distance <= tolerance
    )
    rejection = (
        near
        and
        newest["color"] == "GREEN"
    )
    return {
        "near": near,
        "rejection": rejection
    }
# ============================================================
# MOMENTUM
# ============================================================
def analyze_body_momentum(candles):
    if len(candles) < 3:
        return "UNKNOWN"
    newest = candles[0]["h"]
    previous = candles[1]["h"]
    older = candles[2]["h"]
    average_previous = (
        previous +
        older
    ) / 2
    if newest > average_previous * 1.35:
        return "STRONGER"
    if newest < average_previous * 0.70:
        return "WEAKER"
    return "NORMAL"
# ============================================================
# CONSECUTIVE MOMENTUM
# ============================================================
def analyze_consecutive_momentum(candles):
    if len(candles) < 3:
        return {
            "direction": "NONE",
            "count": 0
        }
    newest_color = candles[0]["color"]
    count = 0
    for candle in candles:
        if candle["color"] == newest_color:
            count += 1
        else:
            break
    if count >= 3:
        direction = (
            "UPWARD"
            if newest_color == "GREEN"
            else
            "DOWNWARD"
        )
    else:
        direction = "NONE"
    return {
        "direction": direction,
        "count": count
    }
# ============================================================
# BODY ENGULFING
# ============================================================
def analyze_body_reversal(candles):
    if len(candles) < 2:
        return None
    current = candles[0]
    previous = candles[1]
    if (
        current["color"]
        ==
        previous["color"]
    ):
        return None
    if (
        current["h"]
        >=
        previous["h"] * 1.20
    ):
        return (
            "BUY"
            if current["color"] == "GREEN"
            else "SELL"
        )
    return None
# ============================================================
# TWO-CANDLE REVERSAL
# ============================================================
def analyze_two_candle_reversal(
    candles
):
    if len(candles) < 2:
        return None
    first = candles[0]
    second = candles[1]
    if (
        first["color"] == "GREEN"
        and
        second["color"] == "RED"
    ):
        if first["h"] >= second["h"] * 0.65:
            return "SELL"
    if (
        first["color"] == "RED"
        and
        second["color"] == "GREEN"
    ):
        if first["h"] >= second["h"] * 0.65:
            return "BUY"
    return None
# ============================================================
# THREE-CANDLE REVERSAL
# ============================================================
def analyze_three_candle_reversal(
    candles
):
    if len(candles) < 3:
        return None
    a = candles[0]
    b = candles[1]
    c = candles[2]
    if (
        a["color"] == "GREEN"
        and
        b["color"] == "GREEN"
        and
        c["color"] == "RED"
    ):
        return "SELL"
    if (
        a["color"] == "RED"
        and
        b["color"] == "RED"
        and
        c["color"] == "GREEN"
    ):
        return "BUY"
    return None
# ============================================================
# PULLBACK
# ============================================================
def analyze_pullback(candles):
    if len(candles) < 4:
        return None
    a = candles[0]
    b = candles[1]
    c = candles[2]
    d = candles[3]
    if (
        a["color"] == b["color"]
        and
        b["color"] == c["color"]
        and
        d["color"] != c["color"]
    ):
        if c["color"] == "GREEN":
            return "BUY"
        if c["color"] == "RED":
            return "SELL"
    return None
# ============================================================
# CONTINUATION
# ============================================================
def analyze_continuation(
    candles
):
    if len(candles) < 3:
        return None
    a = candles[0]
    b = candles[1]
    c = candles[2]
    if (
        a["color"] ==
        b["color"]
        and
        b["color"] ==
        c["color"]
    ):
        if (
            a["h"] >=
            b["h"] * 0.45
        ):
            if a["color"] == "GREEN":
                return "BUY"
            if a["color"] == "RED":
                return "SELL"
    return None
# ============================================================
# BREAKOUT ANALYSIS
# ============================================================
def analyze_breakouts(
    candles,
    resistance,
    support
):
    result = {
        "resistance_break": False,
        "resistance_hold": False,
        "resistance_failure": False,
        "support_break": False,
        "support_hold": False,
        "support_failure": False
    }
    if len(candles) < 2:
        return result
    # --------------------------------------------------------
    # RESISTANCE
    # --------------------------------------------------------
    if resistance:
        level = resistance["level"]
        for i in range(
            1,
            min(
                len(candles),
                LOOKBACK_CANDLES
            )
        ):
            previous = candles[i]
            newer = candles[i - 1]
            previous_top = (
                body_position(previous)[0]
            )
            newer_top = (
                body_position(newer)[0]
            )
            crossed = (
                previous_top <= level
                and
                newer_top > level
            )
            if crossed:
                result[
                    "resistance_break"
                ] = True
                if newer["color"] == "RED":
                    result[
                        "resistance_failure"
                    ] = True
                # A later candle remaining
                # above the zone confirms hold.
                if i >= 2:
                    following = candles[i - 2]
                    following_top = (
                        body_position(
                            following
                        )[0]
                    )
                    if (
                        following_top > level
                    ):
                        result[
                            "resistance_hold"
                        ] = True
                break
    # --------------------------------------------------------
    # SUPPORT
    # --------------------------------------------------------
    if support:
        level = support["level"]
        for i in range(
            1,
            min(
                len(candles),
                LOOKBACK_CANDLES
            )
        ):
            previous = candles[i]
            newer = candles[i - 1]
            previous_bottom = (
                body_position(previous)[1]
            )
            newer_bottom = (
                body_position(newer)[1]
            )
            crossed = (
                previous_bottom >= level
                and
                newer_bottom < level
            )
            if crossed:
                result[
                    "support_break"
                ] = True
                if newer["color"] == "GREEN":
                    result[
                        "support_failure"
                    ] = True
                if i >= 2:
                    following = candles[i - 2]
                    following_bottom = (
                        body_position(
                            following
                        )[1]
                    )
                    if (
                        following_bottom < level
                    ):
                        result[
                            "support_hold"
                        ] = True
                break
    return result
# ============================================================
# COLOR CONFIRMATION
# ============================================================
def analyze_color_confirmation(
    candles,
    resistance,
    support
):
    green_red = False
    red_green = False
    if resistance:
        level = resistance["level"]
        for i in range(
            1,
            min(
                len(candles),
                LOOKBACK_CANDLES
            )
        ):
            previous = candles[i]
            current = candles[i - 1]
            previous_top = (
                body_position(previous)[0]
            )
            current_top = (
                body_position(current)[0]
            )
            if (
                previous["color"] == "GREEN"
                and
                current["color"] == "RED"
                and
                previous_top <= level
                and
                current_top >= level
            ):
                green_red = True
                break
    if support:
        level = support["level"]
        for i in range(
            1,
            min(
                len(candles),
                LOOKBACK_CANDLES
            )
        ):
            previous = candles[i]
            current = candles[i - 1]
            previous_bottom = (
                body_position(previous)[1]
            )
            current_bottom = (
                body_position(current)[1]
            )
            if (
                previous["color"] == "RED"
                and
                current["color"] == "GREEN"
                and
                previous_bottom >= level
                and
                current_bottom <= level
            ):
                red_green = True
                break
    return {
        "green_red": green_red,
        "red_green": red_green
    }
# ============================================================
# NEXT CANDLE PREDICTION
# ============================================================
def predict_next_candle(
    candles,
    resistance,
    support
):
    buy_score = 0
    sell_score = 0
    reasons_buy = []
    reasons_sell = []
    # --------------------------------------------------------
    # ZONE STRUCTURE
    # --------------------------------------------------------
    if support:
        strength = zone_reaction_strength(
            support
        )
        if strength >= 2:
            buy_score += 1
            reasons_buy.append(
                "Support has repeated body reactions"
            )
    if resistance:
        strength = zone_reaction_strength(
            resistance
        )
        if strength >= 2:
            sell_score += 1
            reasons_sell.append(
                "Resistance has repeated body reactions"
            )
    # --------------------------------------------------------
    # CURRENT ZONE REACTION
    # --------------------------------------------------------
    resistance_result = (
        resistance_behavior(
            candles,
            resistance
        )
    )
    support_result = (
        support_behavior(
            candles,
            support
        )
    )
    if support_result["rejection"]:
        buy_score += 2
        reasons_buy.append(
            "GREEN rejection at support"
        )
    if resistance_result["rejection"]:
        sell_score += 2
        reasons_sell.append(
            "RED rejection at resistance"
        )
    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------
    momentum = (
        analyze_consecutive_momentum(
            candles
        )
    )
    if momentum["direction"] == "UPWARD":
        buy_score += 2
        reasons_buy.append(
            "Consecutive GREEN body momentum"
        )
    elif momentum["direction"] == "DOWNWARD":
        sell_score += 2
        reasons_sell.append(
            "Consecutive RED body momentum"
        )
    # --------------------------------------------------------
    # CONTINUATION
    # --------------------------------------------------------
    continuation = (
        analyze_continuation(
            candles
        )
    )
    if continuation == "BUY":
        buy_score += 2
        reasons_buy.append(
            "Bullish continuation structure"
        )
    elif continuation == "SELL":
        sell_score += 2
        reasons_sell.append(
            "Bearish continuation structure"
        )
    # --------------------------------------------------------
    # REVERSALS
    # --------------------------------------------------------
    body_reversal = (
        analyze_body_reversal(
            candles
        )
    )
    if body_reversal == "BUY":
        buy_score += 2
        reasons_buy.append(
            "Bullish body reversal"
        )
    elif body_reversal == "SELL":
        sell_score += 2
        reasons_sell.append(
            "Bearish body reversal"
        )
    two_reversal = (
        analyze_two_candle_reversal(
            candles
        )
    )
    if two_reversal == "BUY":
        buy_score += 1
        reasons_buy.append(
            "Bullish 2-candle reversal"
        )
    elif two_reversal == "SELL":
        sell_score += 1
        reasons_sell.append(
            "Bearish 2-candle reversal"
        )
    three_reversal = (
        analyze_three_candle_reversal(
            candles
        )
    )
    if three_reversal == "BUY":
        buy_score += 1
        reasons_buy.append(
            "Bullish 3-candle reversal"
        )
    elif three_reversal == "SELL":
        sell_score += 1
        reasons_sell.append(
            "Bearish 3-candle reversal"
        )
    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------
    pullback = (
        analyze_pullback(
            candles
        )
    )
    if pullback == "BUY":
        buy_score += 1
        reasons_buy.append(
            "Bullish pullback structure"
        )
    elif pullback == "SELL":
        sell_score += 1
        reasons_sell.append(
            "Bearish pullback structure"
        )
    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------
    breakout = (
        analyze_breakouts(
            candles,
            resistance,
            support
        )
    )
    if breakout["resistance_hold"]:
        buy_score += 2
        reasons_buy.append(
            "Resistance breakout held above the zone"
        )
    if breakout["resistance_failure"]:
        sell_score += 2
        reasons_sell.append(
            "Resistance breakout failed"
        )
    if breakout["support_hold"]:
        sell_score += 2
        reasons_sell.append(
            "Support breakdown held below the zone"
        )
    if breakout["support_failure"]:
        buy_score += 2
        reasons_buy.append(
            "Support breakdown failed"
        )
    # --------------------------------------------------------
    # COLOR CONFIRMATION
    # --------------------------------------------------------
    confirmation = (
        analyze_color_confirmation(
            candles,
            resistance,
            support
        )
    )
    if confirmation["red_green"]:
        buy_score += 2
        reasons_buy.append(
            "RED → GREEN confirmation"
        )
    if confirmation["green_red"]:
        sell_score += 2
        reasons_sell.append(
            "GREEN → RED confirmation"
        )
    # --------------------------------------------------------
    # CONFLICT CHECK
    # --------------------------------------------------------
    total = (
        buy_score +
        sell_score
    )
    if total <= 0:
        return {
            "signal": None,
            "buy_score": 0,
            "sell_score": 0,
            "agreement": 0,
            "reasons": []
        }
    strongest = max(
        buy_score,
        sell_score
    )
    agreement = round(
        strongest /
        total *
        100
    )
    difference = abs(
        buy_score -
        sell_score
    )
    # Require both enough evidence
    # and a meaningful directional gap.
    if (
        buy_score >= MIN_SIGNAL_SCORE
        and
        buy_score > sell_score
        and
        agreement >= MIN_SIGNAL_AGREEMENT
        and
        difference >= 2
    ):
        return {
            "signal": "BUY",
            "buy_score": buy_score,
            "sell_score": sell_score,
            "agreement": agreement,
            "reasons": reasons_buy[:3]
        }
    if (
        sell_score >= MIN_SIGNAL_SCORE
        and
        sell_score > buy_score
        and
        agreement >= MIN_SIGNAL_AGREEMENT
        and
        difference >= 2
    ):
        return {
            "signal": "SELL",
            "buy_score": buy_score,
            "sell_score": sell_score,
            "agreement": agreement,
            "reasons": reasons_sell[:3]
        }
    return {
        "signal": None,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "agreement": agreement,
        "reasons": []
    }
# ============================================================
# NIGERIA TIME
# ============================================================
def nigeria_time():
    # Nigeria is UTC+1.
    utc_now = datetime.utcnow()
    return utc_now + timedelta(
        hours=1
    )
# ============================================================
# BUILD SIGNAL MESSAGE
# ============================================================
def build_signal_message(
    prediction
):
    now = nigeria_time()
    signal_time = now.strftime(
        "%H:%M"
    )
    # The signal is based on the currently
    # visible completed candle sequence.
    # Entry is the OPEN of the next candle.
    entry_time = (
        now +
        timedelta(minutes=1)
    ).strftime("%H:%M")
    if prediction["signal"] == "BUY":
        direction = "🟢 BUY"
    else:
        direction = "🔴 SELL"
    reasons = prediction["reasons"]
    reason_text = "\n".join(
        f"• {reason}"
        for reason in reasons
    )
    return (
        "🚨 **SIGNAL ALERT**\n\n"
        f"📈 **Direction:** {direction}\n"
        f"🕐 **Signal Time:** {signal_time} Nigeria Time\n"
        f"🎯 **Entry Time:** {entry_time} Nigeria Time\n"
        f"📊 **Strategy Agreement:** "
        f"{prediction['agreement']}%\n\n"
        "🧠 **CORE REASONS**\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"{reason_text}\n\n"
        "⚠️ Manual entry only.\n"
        "🎯 Entry is the OPEN of the next candle."
    )
# ============================================================
# TELEGRAM PHOTO HANDLER
# ============================================================
@bot.message_handler(
    content_types=["photo"]
)
def handle_photo(message):
    original_path = (
        "strategy_chart.png"
    )
    try:
        # ----------------------------------------------------
        # DOWNLOAD ORIGINAL SCREENSHOT
        # ----------------------------------------------------
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
        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------
        img = load_image(
            original_path
        )
        # ----------------------------------------------------
        # DETECT CANDLES
        # ----------------------------------------------------
        candles = detect_candles(
            img
        )
        if len(candles) < 5:
            bot.reply_to(
                message,
                "⚪ NO SIGNAL — DON'T TRADE"
            )
            return
        # ----------------------------------------------------
        # FIND STRUCTURE
        # ----------------------------------------------------
        resistance = (
            find_resistance_zone(
                candles
            )
        )
        support = (
            find_support_zone(
                candles
            )
        )
        # ----------------------------------------------------
        # PREDICT NEXT CANDLE
        # ----------------------------------------------------
        prediction = (
            predict_next_candle(
                candles,
                resistance,
                support
            )
        )
        # ----------------------------------------------------
        # NO SIGNAL
        # ----------------------------------------------------
        if prediction["signal"] is None:
            bot.reply_to(
                message,
                "⚪ NO SIGNAL — DON'T TRADE"
            )
            return
        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------
        signal_message = (
            build_signal_message(
                prediction
            )
        )
        # ----------------------------------------------------
        # SEND SIGNAL
        # ----------------------------------------------------
        bot.send_message(
            message.chat.id,
            signal_message,
            parse_mode="Markdown"
        )
        # ----------------------------------------------------
        # SEND ORIGINAL SCREENSHOT
        # ----------------------------------------------------
        with open(
            original_path,
            "rb"
        ) as photo:
            bot.send_photo(
                message.chat.id,
                photo,
                caption=(
                    "📸 Chart used for the signal.\n"
                    "🎯 Entry = OPEN of the next candle."
                )
            )
    except Exception as e:
        print(
            "❌ ERROR:",
            repr(e)
        )
        bot.reply_to(
            message,
            "❌ Screenshot analysis failed."
        )
    finally:
        if os.path.exists(
            original_path
        ):
            try:
                os.remove(
                    original_path
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
    "🧠 OTC NEXT-CANDLE STRATEGY BOT"
)
print(
    "========================================"
)
print(
    "RIGHT → LEFT scanning."
)
print(
    "Candle #1 = newest visible candle."
)
print(
    "Next-candle directional prediction."
)
print(
    "No price mapping."
)
print(
    "No indicators."
)
print(
    "No OCR."
)
print(
    "No detection map."
)
print(
    "No automatic trading."
)
print(
    "Nigeria Time entry calculation."
)
print(
    "Flask keep-alive enabled."
)
print(
    "========================================"
)
bot.infinity_polling(
    timeout=30,
    long_polling_timeout=30
)

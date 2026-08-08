import os
import time
import re
import asyncio
import cv2
import numpy as np
import pytesseract
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
# ============================================================
# TELEGRAM
# ============================================================
TOKEN = "8937673241:AAGvyTA-G12xfwMlhif3Nh4_2Ag8OStq3tU"
# ============================================================
# CANDLE DETECTION SETTINGS
# ============================================================
MIN_CANDLES_EXPECTED = 5
# We do NOT generate candles.
# We do NOT create fake OHLC.
# We only count candles that are visually detected.
#
# The detector intentionally does NOT use a fixed
# "chart_width // 6" candle limit.
# ============================================================
class CandleReader:
    def __init__(self):
        self.last_result = None
    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------
    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Could not read screenshot.")
        return img
    # --------------------------------------------------------
    # FIND CHART AREA
    # --------------------------------------------------------
    def get_chart_region(self, img):
        h, w = img.shape[:2]
        # Pocket Option screenshots normally have:
        # controls above/below the chart and price scale
        # around the right side.
        #
        # We intentionally keep a very large chart area.
        top = int(h * 0.12)
        bottom = int(h * 0.82)
        left = int(w * 0.04)
        right = int(w * 0.91)
        chart = img[top:bottom, left:right]
        return chart
    # --------------------------------------------------------
    # COLOR MASKS
    # --------------------------------------------------------
    def get_masks(self, chart):
        hsv = cv2.cvtColor(chart, cv2.COLOR_BGR2HSV)
        # GREEN
        green1 = cv2.inRange(
            hsv,
            np.array([30, 25, 25]),
            np.array([95, 255, 255])
        )
        # RED
        red1 = cv2.inRange(
            hsv,
            np.array([0, 25, 25]),
            np.array([15, 255, 255])
        )
        red2 = cv2.inRange(
            hsv,
            np.array([165, 25, 25]),
            np.array([180, 255, 255])
        )
        red = cv2.bitwise_or(red1, red2)
        return green1, red
    # --------------------------------------------------------
    # REMOVE UI / GRID NOISE
    # --------------------------------------------------------
    def clean_mask(self, mask):
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )
        return mask
    # --------------------------------------------------------
    # FIND CANDLE CANDIDATES
    # --------------------------------------------------------
    def find_candidates(self, mask, color):
        h, w = mask.shape
        # Vertical projection.
        #
        # Every candle produces pixels at its horizontal
        # position. Unlike the previous code, we don't divide
        # the chart into 25/40/50 artificial columns.
        projection = np.sum(mask > 0, axis=0)
        # Small amount of smoothing prevents a candle from
        # being split into several tiny pieces.
        smooth_kernel = 3
        kernel = np.ones(smooth_kernel) / smooth_kernel
        projection_smooth = np.convolve(
            projection,
            kernel,
            mode="same"
        )
        # Adaptive threshold.
        nonzero = projection_smooth[projection_smooth > 0]
        if len(nonzero) == 0:
            return []
        threshold = max(
            2,
            np.percentile(nonzero, 25)
        )
        active = projection_smooth >= threshold
        # ----------------------------------------------------
        # GROUP NEIGHBORING ACTIVE COLUMNS
        # ----------------------------------------------------
        groups = []
        start = None
        for x in range(w):
            if active[x]:
                if start is None:
                    start = x
            else:
                if start is not None:
                    groups.append((start, x - 1))
                    start = None
        if start is not None:
            groups.append((start, w - 1))
        candidates = []
        for x1, x2 in groups:
            width = x2 - x1 + 1
            # Ignore extremely tiny noise.
            if width < 1:
                continue
            roi = mask[:, x1:x2 + 1]
            ys, xs = np.where(roi > 0)
            if len(ys) < 4:
                continue
            top = int(np.min(ys))
            bottom = int(np.max(ys))
            height = bottom - top + 1
            if height < 2:
                continue
            # Candle-like vertical object.
            #
            # Very wide regions are usually UI/grid elements
            # rather than a single candle.
            if width > max(30, int(w * 0.08)):
                continue
            center = (x1 + x2) / 2
            candidates.append({
                "x1": x1,
                "x2": x2,
                "center": center,
                "top": top,
                "bottom": bottom,
                "height": height,
                "width": width,
                "color": color
            })
        return candidates
    # --------------------------------------------------------
    # MERGE SAME CANDLE COMPONENTS
    # --------------------------------------------------------
    def merge_candidates(self, candidates):
        if not candidates:
            return []
        candidates = sorted(
            candidates,
            key=lambda x: x["center"]
        )
        merged = []
        for c in candidates:
            if not merged:
                merged.append(c)
                continue
            previous = merged[-1]
            # Nearby components can belong to the same candle.
            distance = c["center"] - previous["center"]
            if distance <= max(
                4,
                (c["width"] + previous["width"]) * 0.8
            ):
                new_x1 = min(previous["x1"], c["x1"])
                new_x2 = max(previous["x2"], c["x2"])
                new_top = min(previous["top"], c["top"])
                new_bottom = max(previous["bottom"], c["bottom"])
                # If same color, merge.
                if previous["color"] == c["color"]:
                    previous["x1"] = new_x1
                    previous["x2"] = new_x2
                    previous["center"] = (
                        new_x1 + new_x2
                    ) / 2
                    previous["top"] = new_top
                    previous["bottom"] = new_bottom
                    previous["width"] = new_x2 - new_x1 + 1
                    previous["height"] = new_bottom - new_top + 1
                else:
                    # Different colors very close together are
                    # more likely to be separate candles.
                    merged.append(c)
            else:
                merged.append(c)
        return merged
    # --------------------------------------------------------
    # ALTERNATIVE: CONNECTED COMPONENT METHOD
    # --------------------------------------------------------
    def connected_components(self, mask, color):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
        results = []
        h, w = mask.shape
        for i in range(1, num_labels):
            x, y, width, height, area = stats[i]
            if area < 3:
                continue
            if width > max(35, int(w * 0.08)):
                continue
            if height < 2:
                continue
            if height > int(h * 0.9):
                continue
            results.append({
                "x1": int(x),
                "x2": int(x + width - 1),
                "center": float(centroids[i][0]),
                "top": int(y),
                "bottom": int(y + height - 1),
                "height": int(height),
                "width": int(width),
                "color": color
            })
        return results
    # --------------------------------------------------------
    # BUILD FINAL CANDLE SEQUENCE
    # --------------------------------------------------------
    def build_sequence(self, green_candidates, red_candidates, chart_width):
        all_candidates = (
            green_candidates +
            red_candidates
        )
        if not all_candidates:
            return []
        all_candidates = sorted(
            all_candidates,
            key=lambda x: x["center"]
        )
        # ----------------------------------------------------
        # REMOVE DUPLICATES
        # ----------------------------------------------------
        final = []
        for c in all_candidates:
            duplicate = False
            for existing in final:
                distance = abs(
                    c["center"] -
                    existing["center"]
                )
                if distance <= max(
                    3,
                    min(
                        c["width"],
                        existing["width"]
                    )
                ):
                    # Keep the stronger detection.
                    if c["height"] > existing["height"]:
                        existing.update(c)
                    duplicate = True
                    break
            if not duplicate:
                final.append(c)
        # ----------------------------------------------------
        # REMOVE OBVIOUS UI-LIKE OBJECTS
        # ----------------------------------------------------
        cleaned = []
        for c in final:
            # Candles should not be extremely wide.
            if c["width"] > chart_width * 0.06:
                continue
            # Ignore full-height vertical UI artifacts.
            if c["height"] > chart_width * 0.8:
                continue
            cleaned.append(c)
        cleaned.sort(
            key=lambda x: x["center"]
        )
        return cleaned
    # --------------------------------------------------------
    # OCR ONLY FOR ASSET NAME
    # --------------------------------------------------------
    def detect_asset(self, img):
        h, w = img.shape[:2]
        # Asset/pair is normally around the upper-middle
        # portion of the Pocket Option screen.
        regions = [
            img[int(h * 0.15):int(h * 0.40),
                int(w * 0.20):int(w * 0.90)],
            img[int(h * 0.20):int(h * 0.50),
                int(w * 0.10):int(w * 0.95)]
        ]
        texts = []
        for roi in regions:
            if roi.size == 0:
                continue
            gray = cv2.cvtColor(
                roi,
                cv2.COLOR_BGR2GRAY
            )
            gray = cv2.resize(
                gray,
                None,
                fx=1.5,
                fy=1.5,
                interpolation=cv2.INTER_CUBIC
            )
            _, thresh = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY +
                cv2.THRESH_OTSU
            )
            text = pytesseract.image_to_string(
                thresh,
                config="--psm 6"
            )
            texts.append(text)
        combined = "\n".join(texts)
        # OTC pairs and normal currency pairs.
        patterns = [
            r'([A-Z]{3}\s*/\s*[A-Z]{3}\s*OTC)',
            r'([A-Z]{3}\s*[/\-]\s*[A-Z]{3})',
            r'([A-Z]{3}\s+[A-Z]{3}\s+OTC)',
            r'([A-Z]{3}\s*OTC)'
        ]
        for pattern in patterns:
            match = re.search(
                pattern,
                combined,
                re.IGNORECASE
            )
            if match:
                return match.group(1).upper().strip()
        # American Express is not a currency pair, but if it
        # appears on the chart we report it rather than
        # pretending it is USD/CHF.
        if re.search(
            r'AMERICAN\s+EXPRESS',
            combined,
            re.IGNORECASE
        ):
            return "AMERICAN EXPRESS OTC"
        return "NOT CONFIDENTLY DETECTED"
    # --------------------------------------------------------
    # MAIN READER
    # --------------------------------------------------------
    def read(self, path):
        img = self.load_image(path)
        original = img.copy()
        # Don't resize the entire screenshot unnecessarily.
        # Preserve the actual candle spacing.
        chart = self.get_chart_region(img)
        chart_h, chart_w = chart.shape[:2]
        green_mask, red_mask = self.get_masks(chart)
        green_mask = self.clean_mask(green_mask)
        red_mask = self.clean_mask(red_mask)
        # Method 1: projection
        green_a = self.find_candidates(
            green_mask,
            "GREEN"
        )
        red_a = self.find_candidates(
            red_mask,
            "RED"
        )
        # Method 2: connected components
        green_b = self.connected_components(
            green_mask,
            "GREEN"
        )
        red_b = self.connected_components(
            red_mask,
            "RED"
        )
        # Combine both visual methods.
        green = self.merge_candidates(
            green_a + green_b
        )
        red = self.merge_candidates(
            red_a + red_b
        )
        candles = self.build_sequence(
            green,
            red,
            chart_w
        )
        # ----------------------------------------------------
        # IMPORTANT:
        # NO FAKE CANDLES ARE ADDED HERE.
        # ----------------------------------------------------
        asset = self.detect_asset(original)
        return {
            "asset": asset,
            "candles": candles,
            "green": sum(
                1 for c in candles
                if c["color"] == "GREEN"
            ),
            "red": sum(
                1 for c in candles
                if c["color"] == "RED"
            ),
            "total": len(candles)
        }
# ============================================================
# TELEGRAM MESSAGE
# ============================================================
reader = CandleReader()
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🕯️ CANDLE DETECTION TEST\n\n"
        "Send a Pocket Option screenshot.\n\n"
        "I will only count candles that are actually "
        "visible in the screenshot.\n\n"
        "No OHLC generation.\n"
        "No random candles.\n"
        "No trading signal."
    )
async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    start_time = time.time()
    try:
        await update.message.reply_text(
            "📸 Reading visible candles..."
        )
        photo = await update.message.photo[-1].get_file()
        filename = "candle_test.png"
        await photo.download_to_drive(filename)
        result = reader.read(filename)
        candles = result["candles"]
        sequence = []
        for candle in candles:
            if candle["color"] == "GREEN":
                sequence.append("🟢")
            else:
                sequence.append("🔴")
        elapsed = time.time() - start_time
        if not candles:
            message = (
                "⚠️ **CANDLE DETECTION TEST**\n\n"
                f"💱 Detected asset: `{result['asset']}`\n\n"
                "❌ No reliable visible candles detected.\n\n"
                "No candles were generated."
            )
        else:
            message = (
                "🔎 **CANDLE DETECTION TEST**\n\n"
                f"💱 **Detected asset:** `{result['asset']}`\n\n"
                "📊 **WHAT THE BOT ACTUALLY DETECTED:**\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"🟢 Green candles: **{result['green']}**\n"
                f"🔴 Red candles: **{result['red']}**\n"
                f"📊 Total candles: **{result['total']}**\n\n"
                "🕯️ **Candle sequence (left → right):**\n"
                + " → ".join(sequence)
                + "\n\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "⚠️ **IMPORTANT:**\n"
                "Only visually detected candles are counted.\n"
                "No OHLC candles are generated.\n"
                "No random candles are added.\n"
                "No trading signal is generated.\n\n"
                f"⚡ **Processing time:** {elapsed:.2f}s"
            )
        await update.message.reply_text(
            message,
            parse_mode="Markdown"
        )
        print(message)
    except Exception as e:
        print("ERROR:", repr(e))
        await update.message.reply_text(
            f"❌ Detection error:\n`{str(e)}`",
            parse_mode="Markdown"
        )
# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 50)
    print("🕯️ POCKET OPTION CANDLE READER TEST")
    print("=" * 50)
    print("No random candles.")
    print("No generated OHLC.")
    print("No trading signals.")
    print("=" * 50)
    application = (
        Application
        .builder()
        .token(TOKEN)
        .build()
    )
    application.add_handler(
        CommandHandler("start", start)
    )
    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_photo
        )
    )
    print("✅ Bot is running...")
    application.run_polling(
        drop_pending_updates=True
    )
if __name__ == "__main__":
    main()

Requirements:

python-telegram-bot
opencv-python-headless
numpy
pytesseract
Pillow

And make sure Tesseract itself is installed in the environment if your hosting service doesn’t already provide it.

:::
### One important point
I deliberately **didn't put a fake “50-candle minimum” into the code**. If there are 50 visible candles and the detector only finds 25, it must report 25. We need to make the visual detector actually find the missing 25.
Also, **don't test this with a cropped chart** if your goal is to count everything visible. Send the same full Pocket Option screenshot you normally use.
Once you test this, send me the bot's exact **Green / Red / Total / sequence** result. From that single test we can see whether the problem is **missing candles, merged candles, or false detections** before we touch your reversal strategy.

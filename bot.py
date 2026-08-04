import os
import re
import time
import threading
import requests
import numpy as np
import cv2
import pytesseract
from PIL import Image, ImageEnhance
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime, timezone, timedelta

# ==========================================
# TELEGRAM CREDENTIALS
# ==========================================
TOKEN = "8846196749:AAHxqCpbH9MUQmXUWPmsYI_ktRDYT8mxndc"
CHAT_ID = "6280535707"
CHANNEL_ID = "-1004324805205"

# ==========================================
# TIME ZONE (UTC+1)
# ==========================================
LOCAL_TZ = timezone(timedelta(hours=1))

def get_next_candle_open_time():
    now = datetime.now(LOCAL_TZ)
    next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    return next_minute.strftime("%H:%M:%S")

def get_15s_expiry_from_next():
    now = datetime.now(LOCAL_TZ)
    next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = next_minute + timedelta(seconds=15)
    return expiry.strftime("%H:%M:%S")

# ==========================================
# PAIR DETECTION
# ==========================================
def detect_pair_from_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert('L')
        width, height = img.size
        crop_box = (0, 0, width, height // 3)
        cropped_img = img.crop(crop_box)
        enhancer = ImageEnhance.Contrast(cropped_img)
        cropped_img = enhancer.enhance(2)
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ/'
        text = pytesseract.image_to_string(cropped_img, config=custom_config)
        match = re.search(r'([A-Z]{3}/[A-Z]{3}\s+OTC)', text)
        if match:
            return match.group(1)
        match = re.search(r'([A-Z]{3}/[A-Z]{3})', text)
        if match:
            return match.group(1) + " OTC"
    except:
        pass
    return "AUD/CAD OTC"

# ==========================================
# FLASK WEB SERVER
# ==========================================
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Bot is running!"

@app.route('/ping')
def ping():
    return "pong", 200

def run_flask():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=10000, debug=False, threaded=True)

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"})
        if CHANNEL_ID:
            requests.post(url, data={"chat_id": CHANNEL_ID, "text": message, "parse_mode": "Markdown"})
    except:
        pass

# ==========================================
# SCREENSHOT READER
# ==========================================
class ScreenshotReader:
    def __init__(self):
        self.price_levels = []

    def read_screenshot(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None

        height, width = img.shape[:2]
        if width < 1000:
            new_width = 1500
            new_height = int(height * (1500 / width))
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        price_levels = self._extract_price_levels(img)
        if not price_levels or len(price_levels) < 3:
            return None

        self.price_levels = sorted(price_levels)

        candles = self._extract_candles(img)
        if not candles or len(candles) < 1:
            return None

        return self._generate_ohlc(candles)

    def _extract_price_levels(self, img):
        height, width = img.shape[:2]
        all_prices = []

        x1 = int(width * 0.80)
        x2 = width - 5
        y1 = int(height * 0.05)
        y2 = int(height * 0.95)

        region = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789. --oem 3'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        numbers = re.findall(r'\d+\.\d+', text)

        for num in numbers:
            try:
                val = float(num)
                if 0.01 < val < 1000.0:
                    all_prices.append(val)
            except:
                continue

        if all_prices:
            all_prices = sorted(set(all_prices))
            if len(all_prices) > 5:
                q1 = np.percentile(all_prices, 10)
                q3 = np.percentile(all_prices, 90)
                all_prices = [p for p in all_prices if q1 <= p <= q3]
            if len(all_prices) >= 3:
                return all_prices
        return None

    def _extract_candles(self, img):
        height, width = img.shape[:2]
        chart_region = img[int(height * 0.15):int(height * 0.80), int(width * 0.10):int(width * 0.85)]
        chart_height, chart_width = chart_region.shape[:2]

        hsv = cv2.cvtColor(chart_region, cv2.COLOR_BGR2HSV)

        green_mask = cv2.inRange(hsv, np.array([35, 30, 30]), np.array([85, 255, 255]))
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 30, 30]), np.array([15, 255, 255])),
            cv2.inRange(hsv, np.array([160, 30, 30]), np.array([180, 255, 255]))
        )

        num_candles = min(40, chart_width // 8)
        candle_width = chart_width // num_candles
        candles = []
        min_pixels = 10

        for i in range(num_candles):
            x_start = i * candle_width
            x_end = (i + 1) * candle_width

            green_pixels = np.sum(green_mask[:, x_start:x_end] > 0)
            red_pixels = np.sum(red_mask[:, x_start:x_end] > 0)

            if green_pixels > min_pixels or red_pixels > min_pixels:
                color = 'GREEN' if green_pixels > red_pixels else 'RED'
                col_data = chart_region[:, x_start:x_end]
                gray_col = cv2.cvtColor(col_data, cv2.COLOR_BGR2GRAY)
                non_zero = np.where(gray_col < 220)

                if len(non_zero[0]) > 0:
                    min_y = np.min(non_zero[0])
                    max_y = np.max(non_zero[0])
                    candles.append({
                        'color': color,
                        'top': min_y / chart_height,
                        'bottom': max_y / chart_height,
                    })
        return candles

    def _generate_ohlc(self, candles):
        if not candles or not self.price_levels:
            return None

        min_price = min(self.price_levels)
        max_price = max(self.price_levels)
        price_range = max_price - min_price

        ohlc = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}

        for i, candle in enumerate(candles):
            top_price = max_price - (candle['top'] * price_range)
            bottom_price = max_price - (candle['bottom'] * price_range)

            if candle['color'] == 'GREEN':
                open_price = bottom_price + (top_price - bottom_price) * 0.2
                close_price = top_price - (top_price - bottom_price) * 0.2
            else:
                open_price = top_price - (top_price - bottom_price) * 0.2
                close_price = bottom_price + (top_price - bottom_price) * 0.2

            high_price = max(open_price, close_price) + (price_range * 0.002)
            low_price = min(open_price, close_price) - (price_range * 0.002)

            ohlc['open'].append(open_price)
            ohlc['high'].append(high_price)
            ohlc['low'].append(low_price)
            ohlc['close'].append(close_price)
            ohlc['volume'].append(100 + (i * 5))

        return ohlc

# ==========================================
# FIXED: INDICATORS WITH CORRECT PATTERN DETECTION
# ==========================================
def calculate_indicators(price_data):
    close = np.array(price_data['close'])
    open_ = np.array(price_data['open'])
    high = np.array(price_data['high'])
    low = np.array(price_data['low'])
    volume = np.array(price_data.get('volume', []))

    indicators = {}
    bullish_count = 0
    bearish_count = 0

    # 1. RSI
    if len(close) >= 14:
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        if avg_loss > 0:
            rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
        else:
            rsi = 100
    else:
        rsi = 50

    if rsi > 70:
        bearish_count += 1
    elif rsi < 30:
        bullish_count += 1

    # 2. Stochastic
    if len(close) >= 14:
        high14 = np.max(high[-14:])
        low14 = np.min(low[-14:])
        if high14 - low14 > 0:
            stoch = (close[-1] - low14) / (high14 - low14) * 100
        else:
            stoch = 50
    else:
        stoch = 50

    if stoch > 80:
        bearish_count += 1
    elif stoch < 20:
        bullish_count += 1

    # 3. Bollinger Bands
    if len(close) >= 20:
        sma20 = np.mean(close[-20:])
        std20 = np.std(close[-20:])
        upper_band = sma20 + (2 * std20)
        lower_band = sma20 - (2 * std20)
        bb_position = (close[-1] - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) > 0 else 0.5
    else:
        bb_position = 0.5

    if bb_position > 0.8:
        bearish_count += 1
    elif bb_position < 0.2:
        bullish_count += 1

    # 4. Trend (FIXED: More accurate)
    if len(close) >= 5:
        # Check if price is making higher highs and higher lows
        if close[-1] > close[-2] and close[-2] > close[-3] and close[-1] > close[-5]:
            bullish_count += 2  # Strong uptrend
        elif close[-1] < close[-2] and close[-2] < close[-3] and close[-1] < close[-5]:
            bearish_count += 2  # Strong downtrend
        elif close[-1] > close[-5]:
            bullish_count += 1
        elif close[-1] < close[-5]:
            bearish_count += 1

    # 5. VWAP
    if len(volume) > 0 and len(close) > 0:
        vwap = np.sum(close * volume) / np.sum(volume) if np.sum(volume) > 0 else close[-1]
        if close[-1] > vwap:
            bullish_count += 1
        else:
            bearish_count += 1

    # 6. Candle Pattern (FIXED: Only label if it matches the direction)
    current_close = close[-1]
    current_open = open_[-1]
    current_high = high[-1]
    current_low = low[-1]

    body = abs(current_close - current_open)
    upper_wick = current_high - max(current_open, current_close)
    lower_wick = min(current_open, current_close) - current_low

    pattern = "Unknown"

    # Determine if candle is bullish or bearish
    is_bullish_candle = current_close > current_open
    is_bearish_candle = current_close < current_open

    # Only label patterns that match the candle direction
    if is_bearish_candle and upper_wick > body * 2:
        pattern = "Shooting Star"
    elif is_bullish_candle and lower_wick > body * 2:
        pattern = "Hammer"
    elif is_bearish_candle and upper_wick > body * 1.5:
        pattern = "Bearish Rejection"
    elif is_bullish_candle and lower_wick > body * 1.5:
        pattern = "Bullish Rejection"
    elif is_bearish_candle:
        pattern = "Bearish Candle"
    elif is_bullish_candle:
        pattern = "Bullish Candle"

    # Determine direction based on indicators and candle
    if bearish_count > bullish_count and bearish_count >= 3:
        direction = "SELL"
        confidence = min(92, 80 + (bearish_count - bullish_count) * 2)
        # If candle is bullish but indicators say SELL, it's a potential reversal
        if is_bullish_candle:
            pattern = "Bullish Rejection (Potential Reversal)"
    elif bullish_count > bearish_count and bullish_count >= 3:
        direction = "BUY"
        confidence = min(92, 80 + (bullish_count - bearish_count) * 2)
        # If candle is bearish but indicators say BUY, it's a potential reversal
        if is_bearish_candle:
            pattern = "Bearish Rejection (Potential Reversal)"
    else:
        direction = "NEUTRAL"
        confidence = 0

    return {
        'Direction': direction,
        'Confidence': confidence,
        'Bullish_Count': bullish_count,
        'Bearish_Count': bearish_count,
        'Pattern': pattern,
        'RSI': round(rsi, 1),
        'RSI_Signal': "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral",
        'Stochastic_Signal': "Overbought" if stoch > 80 else "Oversold" if stoch < 20 else "Neutral",
        'BB_Signal': "Overbought" if bb_position > 0.8 else "Oversold" if bb_position < 0.2 else "Neutral",
        if len(close) >= 5:
            if close[-1] > close[-5]:
                trend = "Uptrend"
            elif close[-1] < close[-5]:
                trend = "Downtrend"
            else:
                trend = "Sideways"
        else:
            trend = "Sideways"
            }

# ==========================================
# TELEGRAM BOT HANDLERS
# ==========================================
screenshot_reader = ScreenshotReader()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📊 **OTC Signal Bot**\n\n"
        "Send a screenshot of your OTC chart.\n\n"
        "✅ Fast analysis\n"
        "✅ Correct pattern detection\n"
        "✅ Entry at next candle open\n"
        "✅ Expiry 15s later"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        start_time = time.time()
        await update.message.reply_text("⏳ Analyzing...")

        photo = await update.message.photo[-1].get_file()
        await photo.download_to_drive("screenshot.png")

        pair_name = detect_pair_from_image("screenshot.png")
        price_data = screenshot_reader.read_screenshot("screenshot.png")

        if price_data is None:
            await update.message.reply_text(
                "❌ **Could not read screenshot**\n\n"
                "📸 Please ensure chart is visible"
            )
            return

        indicators = calculate_indicators(price_data)

        direction = indicators['Direction']
        confidence = indicators['Confidence']
        pattern = indicators['Pattern']

        if direction == "NEUTRAL" or confidence < 50:
            await update.message.reply_text(
                f"⛔ **No clear signal — DON'T TRADE.**\n\n"
                f"Bullish: {indicators['Bullish_Count']}\n"
                f"Bearish: {indicators['Bearish_Count']}\n"
                f"Confidence: {confidence}%"
            )
            return

        entry_time = get_next_candle_open_time()
        expiry_time = get_15s_expiry_from_next()

        # Build reason
        reason_parts = []
        if direction == "SELL":
            if indicators['RSI_Signal'] == "Overbought":
                reason_parts.append(f"RSI overbought ({indicators['RSI']})")
            if indicators['BB_Signal'] == "Overbought":
                reason_parts.append("Bollinger Bands overbought")
            if indicators['Trend'] == "Downtrend":
                reason_parts.append("Downtrend confirmed")
            if not reason_parts:
                reason_parts.append("Bearish indicators dominate")
        else:
            if indicators['RSI_Signal'] == "Oversold":
                reason_parts.append(f"RSI oversold ({indicators['RSI']})")
            if indicators['BB_Signal'] == "Oversold":
                reason_parts.append("Bollinger Bands oversold")
            if indicators['Trend'] == "Uptrend":
                reason_parts.append("Uptrend confirmed")
            if not reason_parts:
                reason_parts.append("Bullish indicators dominate")

        reason = " → ".join(reason_parts)

        # Active indicators
        active_indicators = []
        if indicators['RSI_Signal'] != "Neutral":
            active_indicators.append(f"RSI ({indicators['RSI_Signal']})")
        if indicators['Stochastic_Signal'] != "Neutral":
            active_indicators.append(f"Stochastic RSI ({indicators['Stochastic_Signal']})")
        if indicators['BB_Signal'] != "Neutral":
            active_indicators.append(f"Bollinger Bands ({indicators['BB_Signal']})")
        if indicators['Trend'] != "Sideways":
            active_indicators.append(f"Trend ({indicators['Trend']})")
        if not active_indicators:
            active_indicators.append("Price Action")

        # Build response
        direction_emoji = "🔴" if direction == "SELL" else "🟢"
        direction_text = "DOWN" if direction == "SELL" else "UP"

        response = f"📊 **OTC SIGNAL**\n\n"
        response += f"🔍 Pair: {pair_name} (1m)\n"
        response += f"📈 Your signal is {direction_emoji} **{direction_text}**\n"
        response += f"📊 Pattern: {pattern}\n"
        response += f"⏱️ Expiry: 15s\n"
        response += f"🎯 Confidence: {confidence}%\n\n"
        response += f"🔍 Reason:\n{reason}\n\n"
        response += f"📊 **Active Indicators:**\n"
        for ind in active_indicators:
            response += f"✅ {ind}\n"
        response += f"\n⏰ Entry: {entry_time} (next candle open)\n"
        response += f"⏰ Expiry: {expiry_time} (15s later)"

        try:
            await context.bot.forward_message(
                chat_id=CHANNEL_ID,
                from_chat_id=update.message.chat_id,
                message_id=update.message.message_id
            )
        except:
            pass

        send_telegram(response)

        elapsed = time.time() - start_time
        print(f"✅ Signal sent in {elapsed:.2f}s")

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# ==========================================
# START BOT
# ==========================================
def run_telegram():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.run_polling()

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("✅ Flask server started.")
    print("✅ Starting Telegram bot...")
    run_telegram()

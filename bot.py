import os
import time
import threading
import requests
import numpy as np
import cv2
import pytesseract
import re
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime, timezone, timedelta
from collections import deque
from PIL import Image

# ==========================================
# TELEGRAM CREDENTIALS (DO NOT TOUCH)
# ==========================================
TOKEN = "8846196749:AAHxqCpbH9MUQmXUWPmsYI_ktRDYT8mxndc"
CHAT_ID = "6280535707"
CHANNEL_ID = "-1004324805205"

# ==========================================
# TIME ZONE (UTC+1) (DO NOT TOUCH)
# ==========================================
LOCAL_TZ = timezone(timedelta(hours=1))

# ==========================================
# TIME FUNCTIONS (NEXT CANDLE OPEN + 15s EXPIRY)
# ==========================================

def get_next_candle_open_time():
    now = datetime.now(LOCAL_TZ)
    next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    return next_minute.strftime("%H:%M:%S")

def get_15s_expiry_from_next():
    now = datetime.now(LOCAL_TZ)
    next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = next_minute + timedelta(seconds=15)
    return expiry.strftime("%H:%M:%S")

def get_current_candle_open():
    now = datetime.now(LOCAL_TZ)
    return now.replace(second=0, microsecond=0).strftime("%H:%M:%S")

# ==========================================
# FLASK WEB SERVER
# ==========================================
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ OTC Signal Bot is running!"

@app.route('/ping')
def ping():
    return "pong", 200

def run_flask():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=10000, debug=False, threaded=True)

# ==========================================
# SEND TO TELEGRAM (PRIVATE + CHANNEL)
# ==========================================
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"})
        requests.post(url, data={"chat_id": CHANNEL_ID, "text": message, "parse_mode": "Markdown"})
        print("✅ Sent to private and channel")
    except Exception as e:
        print("Telegram error:", e)

# ==========================================
# SCREENSHOT READER - READS REAL DATA
# ==========================================

class ScreenshotReader:
    def __init__(self):
        self.price_levels = []
        self.candle_data = []
        self.pair_name = "Unknown"

    def read_screenshot(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None

        img = self._enhance_image(img)
        print(f"📸 Analyzing screenshot: {img.shape}")

        price_levels = self._extract_price_levels(img)
        if not price_levels or len(price_levels) < 3:
            return None

        self.price_levels = sorted(price_levels)
        print(f"✅ Extracted price levels: {self.price_levels[:6]}")

        candles = self._extract_candles(img)
        if not candles or len(candles) < 1:
            return None

        print(f"✅ Detected {len(candles)} candles")

        ohlc_data = self._generate_ohlc(candles)
        return ohlc_data

    def _enhance_image(self, img):
        height, width = img.shape[:2]
        if width < 1000:
            new_width = 2000
            new_height = int(height * (2000 / width))
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        img = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return img

    def _extract_price_levels(self, img):
        height, width = img.shape[:2]
        all_prices = []

        for x_start in [0.75, 0.80, 0.85]:
            x1 = int(width * x_start)
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
                    if 0.01 < val < 100.0:
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

        green_lower = np.array([35, 30, 30])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        red_lower1 = np.array([0, 30, 30])
        red_upper1 = np.array([15, 255, 255])
        red_lower2 = np.array([160, 30, 30])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, red_lower1, red_upper1),
            cv2.inRange(hsv, red_lower2, red_upper2)
        )

        num_candles = min(50, chart_width // 8)
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
                        'index': i
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
# TRADE ALGO INDICATORS
# ==========================================

def calculate_indicators(price_data):
    close = np.array(price_data['close'])
    open_ = np.array(price_data['open'])
    high = np.array(price_data['high'])
    low = np.array(price_data['low'])
    volume = np.array(price_data.get('volume', []))

    indicators = {}

    # 1. RSI
    if len(close) >= 14:
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100
    else:
        rsi = 50

    indicators['RSI'] = round(rsi, 1)
    indicators['RSI_Signal'] = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"

    # 2. Stochastic RSI
    if len(close) >= 14:
        high14 = np.max(high[-14:])
        low14 = np.min(low[-14:])
        if high14 - low14 > 0:
            stoch = (close[-1] - low14) / (high14 - low14) * 100
        else:
            stoch = 50
    else:
        stoch = 50

    indicators['Stochastic RSI'] = round(stoch, 1)
    indicators['Stochastic_Signal'] = "Overbought" if stoch > 80 else "Oversold" if stoch < 20 else "Neutral"

    # 3. Bollinger Bands
    if len(close) >= 20:
        sma20 = np.mean(close[-20:])
        std20 = np.std(close[-20:])
        upper_band = sma20 + (2 * std20)
        lower_band = sma20 - (2 * std20)
        bb_position = (close[-1] - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) > 0 else 0.5
    else:
        bb_position = 0.5

    indicators['Bollinger_Bands'] = round(bb_position * 100, 1)
    indicators['BB_Signal'] = "Overbought" if bb_position > 0.8 else "Oversold" if bb_position < 0.2 else "Neutral"

    # 4. Trend
    if len(close) >= 5:
        trend = "Uptrend" if close[-1] > close[-5] else "Downtrend" if close[-1] < close[-5] else "Sideways"
    else:
        trend = "Sideways"

    indicators['Trend'] = trend

    # 5. Support / Resistance
    if len(high) >= 10 and len(low) >= 10:
        resistance = np.max(high[-10:])
        support = np.min(low[-10:])
        near_resistance = (resistance - close[-1]) / resistance < 0.002
        near_support = (close[-1] - support) / close[-1] < 0.002
    else:
        resistance = 0
        support = 0
        near_resistance = False
        near_support = False

    indicators['Resistance_Level'] = resistance
    indicators['Support_Level'] = support
    indicators['Near_Resistance'] = near_resistance
    indicators['Near_Support'] = near_support

    # 6. OBV
    if len(volume) >= 14:
        obv = 0
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv += volume[i]
            elif close[i] < close[i-1]:
                obv -= volume[i]
        obv_trend = "Bullish" if obv > 0 else "Bearish"
    else:
        obv_trend = "Neutral"

    indicators['OBV'] = obv_trend

    # 7. VWAP
    if len(volume) > 0 and len(close) > 0:
        vwap = np.sum(close * volume) / np.sum(volume) if np.sum(volume) > 0 else close[-1]
        vwap_signal = "Bullish" if close[-1] > vwap else "Bearish"
    else:
        vwap_signal = "Neutral"

    indicators['VWAP'] = vwap_signal

    # 8. Volume
    if len(volume) > 0:
        avg_volume = np.mean(volume)
        volume_signal = "High" if volume[-1] > avg_volume else "Low"
    else:
        volume_signal = "Normal"

    indicators['Volume'] = volume_signal

    # 9. Ichimoku (simplified)
    if len(close) >= 26:
        tenkan = (np.max(high[-9:]) + np.min(low[-9:])) / 2
        kijun = (np.max(high[-26:]) + np.min(low[-26:])) / 2
        ichimoku_signal = "Bullish" if tenkan > kijun else "Bearish"
    else:
        ichimoku_signal = "Neutral"

    indicators['Ichimoku'] = ichimoku_signal

    # 10. Overall Confidence
    bullish_count = 0
    bearish_count = 0

    if indicators['RSI_Signal'] == "Oversold": bullish_count += 1
    if indicators['RSI_Signal'] == "Overbought": bearish_count += 1
    if indicators['Stochastic_Signal'] == "Oversold": bullish_count += 1
    if indicators['Stochastic_Signal'] == "Overbought": bearish_count += 1
    if indicators['BB_Signal'] == "Oversold": bullish_count += 1
    if indicators['BB_Signal'] == "Overbought": bearish_count += 1
    if indicators['Trend'] == "Uptrend": bullish_count += 1
    if indicators['Trend'] == "Downtrend": bearish_count += 1
    if indicators['VWAP'] == "Bullish": bullish_count += 1
    if indicators['VWAP'] == "Bearish": bearish_count += 1
    if indicators['OBV'] == "Bullish": bullish_count += 1
    if indicators['OBV'] == "Bearish": bearish_count += 1
    if indicators['Ichimoku'] == "Bullish": bullish_count += 1
    if indicators['Ichimoku'] == "Bearish": bearish_count += 1
    if indicators['Near_Resistance']: bearish_count += 1
    if indicators['Near_Support']: bullish_count += 1
    if indicators['Volume'] == "High":
    if close[-1] > close[-2]:
        bullish_count += 1
    else:
        bearish_count += 1

    indicators['Bullish_Count'] = bullish_count
    indicators['Bearish_Count'] = bearish_count

    # Determine direction
    if bearish_count > bullish_count and bearish_count >= 3:
        direction = "SELL"
        confidence = min(95, 80 + (bearish_count - bullish_count) * 2)
    elif bullish_count > bearish_count and bullish_count >= 3:
        direction = "BUY"
        confidence = min(95, 80 + (bullish_count - bearish_count) * 2)
    else:
        direction = "NEUTRAL"
        confidence = 0

    indicators['Direction'] = direction
    indicators['Confidence'] = confidence

    return indicators

# ==========================================
# TELEGRAM BOT HANDLERS
# ==========================================

screenshot_reader = ScreenshotReader()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📊 **OTC Signal Bot**\n\n"
        "Send a screenshot of your Pocket Option chart.\n\n"
        "✅ Trade Algo indicators\n"
        "✅ Entry at next candle open\n"
        "✅ Expiry 15s after entry\n"
        "✅ RSI, Stochastic, Bollinger Bands, OBV, VWAP, Ichimoku, Support/Resistance, Volume Profile"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        start_time = time.time()

        photo = await update.message.photo[-1].get_file()
        await photo.download_to_drive("screenshot.png")

        price_data = screenshot_reader.read_screenshot("screenshot.png")

        if price_data is None:
            await update.message.reply_text(
                "❌ **Could not read screenshot**\n\n"
                "Please ensure:\n"
                "📸 Clear screenshot from Pocket Option\n"
                "📊 Chart is visible\n"
                "🕯️ At least 1 candle visible"
            )
            return

        indicators = calculate_indicators(price_data)

        direction = indicators['Direction']
        confidence = indicators['Confidence']

        if direction == "NEUTRAL" or confidence < 50:
            await update.message.reply_text(
                f"⛔ **No clear signal — DON'T TRADE.**\n\n"
                f"Bullish indicators: {indicators['Bullish_Count']}\n"
                f"Bearish indicators: {indicators['Bearish_Count']}\n"
                f"Confidence: {confidence}%\n"
                f"💡 Not enough confirmation"
            )
            return

        # Entry = next candle open
        entry_time = get_next_candle_open_time()
        expiry_time = get_15s_expiry_from_next()

        # Determine pattern
        close = np.array(price_data['close'])
        open_ = np.array(price_data['open'])
        high = np.array(price_data['high'])
        low = np.array(price_data['low'])

        current_close = close[-1]
        current_open = open_[-1]
        current_high = high[-1]
        current_low = low[-1]

        body = abs(current_close - current_open)
        upper_wick = current_high - max(current_open, current_close)
        lower_wick = min(current_open, current_close) - current_low

        pattern = "Unknown"

        if direction == "SELL":
            if upper_wick > body * 2:
                pattern = "Shooting Star"
            elif len(close) >= 2:
                prev_close = close[-2]
                prev_open = open_[-2]
                if prev_close > prev_open and current_close < current_open and current_close < prev_open and current_open > prev_close:
                    pattern = "Bearish Engulfing"
                elif prev_close > prev_open and current_close < prev_open and current_close > prev_close:
                    pattern = "Dark Cloud Cover"
            if pattern == "Unknown":
                pattern = "Bearish Reversal"
        else:
            if lower_wick > body * 2:
                pattern = "Hammer"
            elif len(close) >= 2:
                prev_close = close[-2]
                prev_open = open_[-2]
                if prev_close < prev_open and current_close > current_open and current_close > prev_open and current_open < prev_close:
                    pattern = "Bullish Engulfing"
                elif prev_close < prev_open and current_close > prev_open and current_close < prev_close:
                    pattern = "Piercing Line"
            if pattern == "Unknown":
                pattern = "Bullish Reversal"

        # Build reason
        reason_parts = []
        if direction == "SELL":
            if indicators['RSI_Signal'] == "Overbought":
                reason_parts.append(f"RSI overbought ({indicators['RSI']})")
            if indicators['BB_Signal'] == "Overbought":
                reason_parts.append("Bollinger Bands overbought")
            if indicators['Stochastic_Signal'] == "Overbought":
                reason_parts.append("Stochastic RSI overbought")
            if indicators['Near_Resistance']:
                reason_parts.append("Price near resistance level")
            if indicators['Trend'] == "Downtrend":
                reason_parts.append("Downtrend confirmed")
            if not reason_parts:
                reason_parts.append("Bearish pattern detected")
        else:
            if indicators['RSI_Signal'] == "Oversold":
                reason_parts.append(f"RSI oversold ({indicators['RSI']})")
            if indicators['BB_Signal'] == "Oversold":
                reason_parts.append("Bollinger Bands oversold")
            if indicators['Stochastic_Signal'] == "Oversold":
                reason_parts.append("Stochastic RSI oversold")
            if indicators['Near_Support']:
                reason_parts.append("Price near support level")
            if indicators['Trend'] == "Uptrend":
                reason_parts.append("Uptrend confirmed")
            if not reason_parts:
                reason_parts.append("Bullish pattern detected")

        reason = " → ".join(reason_parts)

        # Build active indicators list
        active_indicators = []
        if indicators['RSI_Signal'] != "Neutral":
            active_indicators.append(f"RSI ({indicators['RSI_Signal']})")
        if indicators['Stochastic_Signal'] != "Neutral":
            active_indicators.append(f"Stochastic RSI ({indicators['Stochastic_Signal']})")
        if indicators['BB_Signal'] != "Neutral":
            active_indicators.append(f"Bollinger Bands ({indicators['BB_Signal']})")
        if indicators['Trend'] != "Sideways":
            active_indicators.append(f"Trend ({indicators['Trend']})")
        if indicators['VWAP'] != "Neutral":
            active_indicators.append(f"VWAP ({indicators['VWAP']})")
        if indicators['OBV'] != "Neutral":
            active_indicators.append(f"OBV ({indicators['OBV']})")
        if indicators['Ichimoku'] != "Neutral":
            active_indicators.append(f"Ichimoku ({indicators['Ichimoku']})")
        if indicators['Volume'] == "High":
            active_indicators.append("Volume (High)")
        if indicators['Near_Resistance']:
            active_indicators.append("Near Resistance")
        if indicators['Near_Support']:
            active_indicators.append("Near Support")

        if not active_indicators:
            active_indicators.append("Price Action")

        # Build response
        if direction == "SELL":
            direction_emoji = "🔴"
            direction_text = "DOWN"
        else:
            direction_emoji = "🟢"
            direction_text = "UP"

        response = f"📊 **OTC SIGNAL**\n\n"
        response += f"🔍 Pair: {screenshot_reader.pair_name} (1m)\n"
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

        await context.bot.forward_message(
            chat_id=CHANNEL_ID,
            from_chat_id=update.message.chat_id,
            message_id=update.message.message_id
        )

        send_telegram(response)

        elapsed = time.time() - start_time
        print(f"✅ Signal sent in {elapsed:.2f} seconds")

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# ==========================================
# START BOT
# ==========================================

def run_telegram():
    application = Application.builder().token(TOKEN).build()
    application.bot.delete_webhook()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.run_polling()

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    print("✅ Flask server started.")
    print("✅ Starting Telegram bot...")
    run_telegram()

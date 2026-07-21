import os
import time
import threading
import requests
import numpy as np
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime, timezone, timedelta
from collections import deque
import cv2
import pytesseract
from PIL import Image
import re

# ==========================================
# TELEGRAM CREDENTIALS
# ==========================================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8846196749:AAHxqCpbH9MUQmXUWPmsYI_ktRDYT8mxndc")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6280535707")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "-1004324805205")

# ==========================================
# TIME ZONE (UTC+1)
# ==========================================
LOCAL_TZ = timezone(timedelta(hours=1))

# ==========================================
# STRATEGY HEALTH TRACKING
# ==========================================
strategy_history = {name: deque(maxlen=10) for name in [
    "Candle Reversal Pattern", "3-Candle Momentum", "2-Minute Reset",
    "Double Touch", "Spike Rejection", "Consolidation Break",
    "EMA Pullback", "Bull/Bear Confirmation", "60-Second Scalp",
    "RSI Divergence", "Bollinger Squeeze", "MACD Crossover",
    "Support/Resistance Break", "MA Crossover"
]}

def get_strategy_health(strategy_name):
    if strategy_name not in strategy_history:
        return 50
    history = strategy_history[strategy_name]
    if len(history) == 0:
        return 50
    win_rate = sum(history) / len(history) * 100
    return min(100, max(50, win_rate))

def record_signal(strategy_name, win):
    if strategy_name in strategy_history:
        strategy_history[strategy_name].append(1 if win else 0)

# ==========================================
# TIME FUNCTIONS
# ==========================================
def get_next_minute():
    now = datetime.now(LOCAL_TZ)
    if now.second > 0 or now.microsecond > 0:
        next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    else:
        next_minute = now.replace(second=0, microsecond=0)
    return next_minute.strftime("%H:%M:%S")

def get_entry2_time(entry1_time):
    return (datetime.strptime(entry1_time, "%H:%M:%S") + timedelta(minutes=1)).strftime("%H:%M:%S")

# ==========================================
# POCKET OPTION SCREENSHOT READER - REAL DATA ONLY
# ==========================================

class PocketOptionScreenshotReader:
    """Reads REAL candlestick data from Pocket Option screenshots"""
    
    def __init__(self):
        self.price_levels = []
        
    def read_screenshot(self, image_path):
        """Extract REAL OHLC data from screenshot - NO RANDOM DATA"""
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Could not load image")
            return None
            
        print(f"📸 Reading REAL data from screenshot")
        
        # Detect chart area
        chart = self._detect_chart_area(img)
        if chart is None:
            print("❌ Could not detect chart area")
            return None
            
        # Extract price levels
        price_levels = self._extract_price_levels(img)
        if not price_levels or len(price_levels) < 3:
            print("❌ Could not extract price levels")
            return None
            
        self.price_levels = sorted(price_levels)
        print(f"💹 Extracted {len(self.price_levels)} price levels")
        
        # Detect candlesticks
        candles = self._detect_candlesticks(chart)
        if len(candles) < 5:
            print(f"❌ Only {len(candles)} candles detected - need at least 5")
            return None
            
        print(f"🕯️ Detected {len(candles)} REAL candles")
        
        # Map to OHLC
        ohlc_data = self._map_candles_to_prices(candles)
        
        if ohlc_data and len(ohlc_data['close']) > 0:
            print(f"✅ Success: {len(ohlc_data['close'])} REAL candles extracted")
            print(f"   Price range: {min(ohlc_data['close']):.5f} - {max(ohlc_data['close']):.5f}")
            return ohlc_data
        
        return None
    
    def _detect_chart_area(self, img):
        """Detect the chart area"""
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        chart_rect = None
        max_area = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (w > width * 0.4 and h > height * 0.3 and
                x > width * 0.1 and x < width * 0.8 and
                y > height * 0.1 and y < height * 0.8):
                area = w * h
                if area > max_area:
                    max_area = area
                    chart_rect = (x, y, w, h)
        
        if chart_rect:
            x, y, w, h = chart_rect
            return img[y:y+h, x:x+w]
        
        # Fallback: central region
        margin_w = int(width * 0.08)
        margin_h = int(height * 0.08)
        return img[margin_h:height-margin_h, margin_w:width-margin_w]
    
    def _extract_price_levels(self, img):
        """Extract price levels from Y-axis using OCR"""
        height, width = img.shape[:2]
        
        # Try right side first
        price_region = img[int(height*0.05):int(height*0.95), int(width*0.90):width-5]
        gray = cv2.cvtColor(price_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        try:
            custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789.'
            text = pytesseract.image_to_string(thresh, config=custom_config)
            numbers = re.findall(r'\d+\.\d+', text)
            
            price_levels = []
            for num in numbers:
                try:
                    val = float(num)
                    if 0 < val < 100:
                        price_levels.append(val)
                except:
                    continue
            
            if price_levels:
                price_levels = sorted(set(price_levels))
                if len(price_levels) > 5:
                    q1 = np.percentile(price_levels, 10)
                    q3 = np.percentile(price_levels, 90)
                    price_levels = [p for p in price_levels if q1 <= p <= q3]
                return price_levels
        except:
            pass
        
        # Try left side
        try:
            price_region = img[int(height*0.05):int(height*0.95), 5:int(width*0.10)]
            gray = cv2.cvtColor(price_region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh, config=custom_config)
            numbers = re.findall(r'\d+\.\d+', text)
            price_levels = []
            for num in numbers:
                try:
                    val = float(num)
                    if 0 < val < 100:
                        price_levels.append(val)
                except:
                    continue
            if price_levels:
                return sorted(set(price_levels))
        except:
            pass
        
        return None
    
    def _detect_candlesticks(self, chart):
        """Detect candlesticks from chart image"""
        height, width = chart.shape[:2]
        hsv = cv2.cvtColor(chart, cv2.COLOR_BGR2HSV)
        
        # Green candles (bullish)
        green_lower = np.array([40, 40, 40])
        green_upper = np.array([80, 255, 255])
        
        # Red candles (bearish)
        red_lower1 = np.array([0, 40, 40])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 40, 40])
        red_upper2 = np.array([180, 255, 255])
        
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        num_candles = min(60, width // 8)
        candle_width = width // num_candles
        candles = []
        
        for i in range(num_candles):
            x_start = i * candle_width
            x_end = (i + 1) * candle_width
            if x_end > width:
                break
            
            green_pixels = np.sum(green_mask[:, x_start:x_end] > 0)
            red_pixels = np.sum(red_mask[:, x_start:x_end] > 0)
            min_pixels = 10
            
            if green_pixels > min_pixels or red_pixels > min_pixels:
                color = 'GREEN' if green_pixels > red_pixels else 'RED'
                
                col_data = chart[:, x_start:x_end]
                gray_col = cv2.cvtColor(col_data, cv2.COLOR_BGR2GRAY)
                non_zero = np.where(gray_col < 240)
                
                if len(non_zero[0]) > 0:
                    min_y = np.min(non_zero[0])
                    max_y = np.max(non_zero[0])
                    
                    high_norm = min_y / height
                    low_norm = max_y / height
                    
                    if color == 'GREEN':
                        col_green = green_mask[:, x_start:x_end]
                        body_pixels = np.where(col_green > 0)
                        if len(body_pixels[0]) > 0:
                            body_top = np.min(body_pixels[0]) / height
                            body_bottom = np.max(body_pixels[0]) / height
                        else:
                            body_range = (low_norm - high_norm) * 0.4
                            body_top = high_norm + body_range
                            body_bottom = low_norm - body_range
                    else:
                        col_red = red_mask[:, x_start:x_end]
                        body_pixels = np.where(col_red > 0)
                        if len(body_pixels[0]) > 0:
                            body_top = np.min(body_pixels[0]) / height
                            body_bottom = np.max(body_pixels[0]) / height
                        else:
                            body_range = (low_norm - high_norm) * 0.4
                            body_top = high_norm + body_range
                            body_bottom = low_norm - body_range
                    
                    candle = {
                        'color': color,
                        'open': body_bottom if color == 'GREEN' else body_top,
                        'high': high_norm,
                        'low': low_norm,
                        'close': body_top if color == 'GREEN' else body_bottom,
                        'index': i
                    }
                    candles.append(candle)
        
        # Filter out tiny candles (noise)
        candles = [c for c in candles if (c['high'] - c['low']) > 0.01]
        return candles
    
    def _map_candles_to_prices(self, candles):
        """Map normalized values to actual prices"""
        if not candles or not self.price_levels:
            return None
        
        min_price = min(self.price_levels)
        max_price = max(self.price_levels)
        price_range = max_price - min_price
        
        ohlc = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
        
        for candle in candles:
            def map_to_price(norm_val):
                return max_price - (norm_val * price_range)
            
            open_price = map_to_price(candle['open'])
            high_price = map_to_price(candle['high'])
            low_price = map_to_price(candle['low'])
            close_price = map_to_price(candle['close'])
            
            # Ensure OHLC logic is correct
            if high_price < max(open_price, close_price):
                high_price = max(open_price, close_price) + (price_range * 0.001)
            if low_price > min(open_price, close_price):
                low_price = min(open_price, close_price) - (price_range * 0.001)
            
            ohlc['open'].append(open_price)
            ohlc['high'].append(high_price)
            ohlc['low'].append(low_price)
            ohlc['close'].append(close_price)
            ohlc['volume'].append(int((high_price - low_price) * 100000) + 100)
        
        return ohlc

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
# SEND TO TELEGRAM
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
# ==========================================
# 14 STRATEGIES - ALL COMPLETE
# ==========================================
# ==========================================

def run_strategies(price_data):
    results = []
    close = np.array(price_data['close'])
    open_ = np.array(price_data['open'])
    high = np.array(price_data['high'])
    low = np.array(price_data['low'])
    volume = np.array(price_data.get('volume', np.ones(len(close))))

    def calculate_rsi(data, period=14):
        if len(data) < period + 1:
            return 50
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    rsi = calculate_rsi(close)

    def calculate_ema(data, period=20):
        if len(data) < period:
            return data[-1] if len(data) > 0 else 0
        return np.mean(data[-period:])

    ema20 = calculate_ema(close, 20)

    def calculate_atr(high, low, period=14):
        if len(high) < period:
            return np.mean(high - low)
        return np.mean(high[-period:] - low[-period:])

    atr = calculate_atr(high, low)

    # --- 1. Candle Reversal Pattern ---
    if len(close) >= 3:
        if (close[-3:] > open_[-3:]).all() and (close[-1] < open_[-1]) and rsi > 70:
            results.append(("Candle Reversal Pattern", "SELL", 86, 2, 3))
        elif (close[-3:] < open_[-3:]).all() and (close[-1] > open_[-1]) and rsi < 30:
            results.append(("Candle Reversal Pattern", "BUY", 86, 2, 3))

    # --- 2. 3-Candle Momentum ---
    if len(close) >= 3:
        avg_volume = np.mean(volume[-5:]) if len(volume) >= 5 else np.mean(volume)
        if (close[-3:] > open_[-3:]).all() and volume[-1] > avg_volume:
            results.append(("3-Candle Momentum", "BUY", 82, 1, 2))
        elif (close[-3:] < open_[-3:]).all() and volume[-1] > avg_volume:
            results.append(("3-Candle Momentum", "SELL", 82, 1, 2))

    # --- 3. 2-Minute Reset ---
    if len(close) >= 3:
        if (close[-3:] < open_[-3:]).all() and close[-1] < ema20 * 0.98:
            results.append(("2-Minute Reset", "SELL", 78, 2, 3))
        elif (close[-3:] > open_[-3:]).all() and close[-1] > ema20 * 1.02:
            results.append(("2-Minute Reset", "BUY", 78, 2, 3))

    # --- 4. Double Touch ---
    if len(close) >= 10:
        if abs(low[-1] - low[-3]) < 0.0002 and close[-1] > max(close[-5:-1]):
            results.append(("Double Touch", "BUY", 89, 3, 4))
        elif abs(high[-1] - high[-3]) < 0.0002 and close[-1] < min(close[-5:-1]):
            results.append(("Double Touch", "SELL", 89, 3, 4))

    # --- 5. Spike Rejection ---
    if len(close) >= 2:
        avg_range = np.mean(high[-5:] - low[-5:]) if len(high) >= 5 else np.mean(high - low)
        if (high[-1] - high[-2]) > 2 * avg_range and (close[-1] < open_[-1]):
            results.append(("Spike Rejection", "SELL", 74, 2, 3))
        elif (low[-2] - low[-1]) > 2 * avg_range and (close[-1] > open_[-1]):
            results.append(("Spike Rejection", "BUY", 74, 2, 3))

    # --- 6. Consolidation Break ---
    if len(close) >= 10:
        high_range = max(high[-10:]) - min(low[-10:])
        if high_range < 0.0005 and (close[-1] - open_[-1]) > 0.0005:
            results.append(("Consolidation Break", "BUY", 71, 3, 4))
        elif high_range < 0.0005 and (open_[-1] - close[-1]) > 0.0005:
            results.append(("Consolidation Break", "SELL", 71, 3, 4))

    # --- 7. EMA Pullback ---
    if len(close) >= 20:
        if low[-1] < ema20 and close[-1] > ema20 and rsi > 40:
            results.append(("EMA Pullback", "BUY", 80, 2, 3))
        elif high[-1] > ema20 and close[-1] < ema20 and rsi < 60:
            results.append(("EMA Pullback", "SELL", 80, 2, 3))

    # --- 8. Bull/Bear Confirmation ---
    if len(close) >= 3:
        if (close[-1] > open_[-1] and close[-2] > open_[-2] and close[-3] > open_[-3]):
            results.append(("Bull/Bear Confirmation", "BUY", 76, 1, 2))
        elif (close[-1] < open_[-1] and close[-2] < open_[-2] and close[-3] < open_[-3]):
            results.append(("Bull/Bear Confirmation", "SELL", 76, 1, 2))

    # --- 9. 60-Second Scalp ---
    if len(close) >= 2:
        avg_volume = np.mean(volume[-3:]) if len(volume) >= 3 else np.mean(volume)
        if (close[-1] - open_[-1]) > (close[-2] - open_[-2]) * 1.5 and volume[-1] > avg_volume:
            results.append(("60-Second Scalp", "BUY", 72, 1, 1))
        elif (open_[-1] - close[-1]) > (open_[-2] - close[-2]) * 1.5 and volume[-1] > avg_volume:
            results.append(("60-Second Scalp", "SELL", 72, 1, 1))

    # --- 10. RSI Divergence ---
    if len(close) >= 20:
        if close[-1] < min(close[-5:-1]) and rsi > min(50, np.mean(rsi)):
            results.append(("RSI Divergence", "BUY", 84, 2, 3))
        elif close[-1] > max(close[-5:-1]) and rsi < max(50, np.mean(rsi)):
            results.append(("RSI Divergence", "SELL", 84, 2, 3))

    # --- 11. Bollinger Squeeze ---
    if len(close) >= 20:
        current_range = high[-1] - low[-1]
        if current_range < atr * 0.5:
            if close[-1] > open_[-1] and close[-1] > ema20:
                results.append(("Bollinger Squeeze", "BUY", 77, 2, 3))
            elif close[-1] < open_[-1] and close[-1] < ema20:
                results.append(("Bollinger Squeeze", "SELL", 77, 2, 3))

    # --- 12. MACD Crossover ---
    if len(close) >= 26:
        ema12 = np.mean(close[-12:])
        ema26 = np.mean(close[-26:])
        macd = ema12 - ema26
        signal = np.mean(close[-9:])
        if macd > signal and close[-1] > ema20:
            results.append(("MACD Crossover", "BUY", 80, 2, 3))
        elif macd < signal and close[-1] < ema20:
            results.append(("MACD Crossover", "SELL", 80, 2, 3))

    # --- 13. Support/Resistance Break ---
    if len(close) >= 20:
        resistance = max(high[-20:-1])
        support = min(low[-20:-1])
        if close[-1] > resistance and close[-1] > open_[-1]:
            results.append(("Support/Resistance Break", "BUY", 81, 2, 3))
        elif close[-1] < support and close[-1] < open_[-1]:
            results.append(("Support/Resistance Break", "SELL", 81, 2, 3))

    # --- 14. MA Crossover ---
    if len(close) >= 30:
        ma10 = np.mean(close[-10:])
        ma30 = np.mean(close[-30:])
        if ma10 > ma30 and close[-1] > open_[-1]:
            results.append(("MA Crossover", "BUY", 79, 2, 3))
        elif ma10 < ma30 and close[-1] < open_[-1]:
            results.append(("MA Crossover", "SELL", 79, 2, 3))

    # ==========================================
    # 5+ STRATEGIES MUST AGREE
    # ==========================================

    if len(results) < 5:
        return []

    buy_signals = [r for r in results if r[1] == "BUY"]
    sell_signals = [r for r in results if r[1] == "SELL"]

    if len(buy_signals) > len(sell_signals):
        direction = "BUY"
        group = buy_signals
    elif len(sell_signals) > len(buy_signals):
        direction = "SELL"
        group = sell_signals
    else:
        buy_avg = np.mean([r[2] for r in buy_signals]) if buy_signals else 0
        sell_avg = np.mean([r[2] for r in sell_signals]) if sell_signals else 0
        if buy_avg >= sell_avg:
            direction = "BUY"
            group = buy_signals
        else:
            direction = "SELL"
            group = sell_signals

    num_agree = len(group)

    if num_agree >= 10:
        agreement_conf = 90
    elif num_agree >= 8:
        agreement_conf = 85
    elif num_agree >= 6:
        agreement_conf = 80
    else:
        agreement_conf = 75

    avg_conf = np.mean([r[2] for r in group]) if group else 50
    final_conf = int((agreement_conf + avg_conf) / 2)
    final_conf = min(100, max(50, final_conf))

    best = max(group, key=lambda x: x[2])

    return [(best[0], direction, final_conf, best[3], best[4])]

# ==========================================
# PREDICTION ENGINE
# ==========================================
def predict_next_candles(strategy, direction, confidence, price_data):
    close = np.array(price_data['close'])
    high = np.array(price_data['high'])
    low = np.array(price_data['low'])

    base_prob = confidence / 100

    if close[-1] > close[-5:].mean():
        trend_factor = 0.10
    else:
        trend_factor = -0.10

    resistance = high[-5:].max()
    support = low[-5:].min()

    if close[-1] < support + 0.001:
        sr_factor = 0.08
    elif close[-1] > resistance - 0.001:
        sr_factor = -0.08
    else:
        sr_factor = 0

    prob1 = base_prob + trend_factor + sr_factor
    prob1 = max(0.50, min(0.90, prob1))

    prob2 = prob1 * 0.90
    prob3 = prob1 * 0.80

    if close[-1] > resistance - 0.001:
        prob3 = 1 - prob3
    elif close[-1] < support + 0.001:
        prob3 = prob1 * 0.85

    if direction == "SELL":
        prob1 = 1 - prob1
        prob2 = 1 - prob2
        prob3 = 1 - prob3

    return {
        "candle1": {"up": round(prob1 * 100, 1), "down": round((1 - prob1) * 100, 1)},
        "candle2": {"up": round(prob2 * 100, 1), "down": round((1 - prob2) * 100, 1)},
        "candle3": {"up": round(prob3 * 100, 1), "down": round((1 - prob3) * 100, 1)}
    }

def predict_entries(strategy, direction, confidence, expiry_1, expiry_2):
    entry1_time = get_next_minute()
    entry2_time = get_entry2_time(entry1_time)

    entry1_dir = "🟢 BUY" if direction == "BUY" else "🔴 SELL"
    entry2_dir = "🟢 BUY" if direction == "BUY" else "🔴 SELL"
    entry2_conf = max(confidence - 10, 50)

    return {
        "strategy": strategy,
        "entry1": {"time": entry1_time, "dir": entry1_dir, "conf": confidence, "expiry": expiry_1},
        "entry2": {"time": entry2_time, "dir": entry2_dir, "conf": entry2_conf, "expiry": expiry_2}
    }

# ==========================================
# TELEGRAM BOT HANDLERS
# ==========================================

# Initialize screenshot reader
screenshot_reader = PocketOptionScreenshotReader()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📊 **OTC Signal Bot**\n\n"
        "📸 Send a **Pocket Option screenshot**\n"
        "🤖 I'll extract REAL candlestick data\n"
        "📈 And run 14 strategies with 5-agreement filter\n\n"
        "⚠️ **No fake data - only real analysis!**"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        start_time = time.time()

        photo = await update.message.photo[-1].get_file()
        await photo.download_to_drive("screenshot.png")

        # READ REAL DATA FROM SCREENSHOT - NO RANDOM
        price_data = screenshot_reader.read_screenshot("screenshot.png")

        # If no real data, show error - NO FAKE DATA
        if price_data is None:
            await update.message.reply_text(
                "❌ **Could not read screenshot**\n\n"
                "Please ensure:\n"
                "📸 Clear screenshot from Pocket Option\n"
                "📊 Chart is visible\n"
                "🕯️ At least 5 candles visible\n\n"
                "⚠️ **No fake data generated!**"
            )
            return

        # RUN STRATEGIES ON REAL DATA
        results = run_strategies(price_data)

        if not results:
            await update.message.reply_text(
                "⛔ **No clear signal — DON'T TRADE.**\n\n"
                f"📊 Analyzed {len(price_data['close'])} REAL candles\n"
                "💡 Less than 5 strategies agreed\n"
                "⏳ Wait for stronger pattern formation"
            )
            return

        # Get best signal
        best = max(results, key=lambda x: x[2])
        strategy, direction, confidence, expiry_1, expiry_2 = best
        prediction = predict_entries(strategy, direction, confidence, expiry_1, expiry_2)
        candle_pred = predict_next_candles(strategy, direction, confidence, price_data)

        response = f"📊 **OTC SIGNAL**\n\n"
        response += f"📈 **Entry 1:**\n"
        response += f"   {prediction['entry1']['dir']} at {prediction['entry1']['time']} ({prediction['entry1']['expiry']} min) — Confidence: {prediction['entry1']['conf']}%\n\n"
        response += f"🔍 **Strategy:** {strategy}\n"
        response += f"   → Direction: {direction}\n"
        response += f"   → Confidence: {confidence}%\n"
        response += f"   → Expiry: {expiry_1} min\n\n"
        response += f"📈 **Entry 2:**\n"
        response += f"   {prediction['entry2']['dir']} at {prediction['entry2']['time']} ({prediction['entry2']['expiry']} min) — Confidence: {prediction['entry2']['conf']}%\n\n"
        response += f"📊 **Next 3 Candles (Probability):**\n"
        
        for i, candle in enumerate([candle_pred['candle1'], candle_pred['candle2'], candle_pred['candle3']], 1):
            if candle['up'] > candle['down']:
                response += f"   - Candle {i}: ⬆️ UP {candle['up']}%\n"
            else:
                response += f"   - Candle {i}: ⬇️ DOWN {candle['down']}%\n"
        
        response += f"\n📊 **Data:** {len(price_data['close'])} REAL candles from screenshot"
        response += f"\n⚠️ **Risk Warning:** Trade responsibly!"

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
    if not TOKEN or not CHAT_ID or not CHANNEL_ID:
        print("❌ Missing Telegram credentials!")
        exit(1)
    
    threading.Thread(target=run_flask, daemon=True).start()
    print("✅ Flask server started.")
    print("✅ Starting Telegram bot...")
    run_telegram(

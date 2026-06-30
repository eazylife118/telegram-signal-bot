import requests
import random
import time
import pandas as pd
import numpy as np
from flask import Flask
from threading import Thread

app = Flask(__name__)

# ==========================================
# CONFIGURATION – HARDCODED KEYS
# ==========================================
TWELVE_API_KEY = "90ab0986c80046bbb59e117779ffdd18"
BOT_TOKEN = "8612354100:AAFUTlaSiq19yycQWpO70J4d6DEbgF4Kicc"
CHAT_ID = "6280535707"

# ==========================================
# ALL OTC PAIRS TRACKED BY TWELVE DATA
# ==========================================
pairs = [
    "EURUSD-OTC",
    "GBPUSD-OTC",
    "USDJPY-OTC",
    "AUDUSD-OTC",
    "USDCAD-OTC",
    "USDCHF-OTC",
    "EURJPY-OTC",
    "GBPJPY-OTC",
    "AUDJPY-OTC",
    "CADJPY-OTC",
    "EURGBP-OTC",
    "EURAUD-OTC",
    "EURCAD-OTC",
    "EURCHF-OTC",
    "GBPAUD-OTC",
    "GBPCAD-OTC",
    "GBPCHF-OTC",
    "AUDCHF-OTC",
    "CADCHF-OTC",
    "NZDUSD-OTC",
    "EURTRY-OTC",
    "USDMXN-OTC",
    "USDCLP-OTC",
    "USDPKR-OTC",
    "USDIDR-OTC"
]

# ==========================================
# ROUND NUMBER LEVELS
# ==========================================
ROUND_LEVELS = {
    "EURUSD-OTC": 1.15600,
    "GBPUSD-OTC": 1.24800,
    "USDJPY-OTC": 162.800,
    "AUDUSD-OTC": 0.72100,
    "USDCAD-OTC": 1.39500,
    "USDCHF-OTC": 0.79400,
    "EURJPY-OTC": 183.800,
    "GBPJPY-OTC": 215.600,
    "AUDJPY-OTC": 108.400,
    "CADJPY-OTC": 120.800,
    "EURGBP-OTC": 0.85800,
    "EURAUD-OTC": 1.64000,
    "EURCAD-OTC": 1.50000,
    "EURCHF-OTC": 1.02459,
    "GBPAUD-OTC": 1.92135,
    "GBPCAD-OTC": 1.72000,
    "GBPCHF-OTC": 1.16000,
    "AUDCHF-OTC": 0.60884,
    "CADCHF-OTC": 0.58832,
    "NZDUSD-OTC": 0.59000,
    "EURTRY-OTC": 53.01600,
    "USDMXN-OTC": 17.7000,
    "USDCLP-OTC": 866.8000,
    "USDPKR-OTC": 277.400,
    "USDIDR-OTC": 16670.0
}

previous_prices = {}
rejection_count = {}
last_signal_time = {}

# ==========================================
# FETCH ALL PAIRS IN ONE BATCH REQUEST
# ==========================================
def get_all_prices():
    try:
        symbols = ",".join([p.replace("-OTC", "") for p in pairs])
        url = f"https://api.twelvedata.com/price?symbol={symbols}&apikey={TWELVE_API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()

        prices = {}
        if "price" in data:
            return {pairs[0]: float(data["price"])}
        else:
            for item in data:
                symbol = item.get("symbol")
                price = item.get("price")
                if symbol and price:
                    for pair in pairs:
                        if pair.replace("-OTC", "") == symbol:
                            prices[pair] = float(price)
                            break
            return prices
    except Exception as e:
        print(f"⚠️ Batch price error: {e}")
        return {}

# ==========================================
# GET OHLC DATA FOR TECHNICAL INDICATORS
# ==========================================
def get_ohlc(pair, interval="1min", outputsize=50):
    try:
        symbol = pair.replace("-OTC", "")
        url = (
            f"https://api.twelvedata.com/time_series"
            f"?symbol={symbol}"
            f"&interval={interval}"
            f"&outputsize={outputsize}"
            f"&apikey={TWELVE_API_KEY}"
        )
        response = requests.get(url).json()
        if "values" in response:
            closes = [float(candle["close"]) for candle in response["values"]]
            highs = [float(candle["high"]) for candle in response["values"]]
            lows = [float(candle["low"]) for candle in response["values"]]
            return closes, highs, lows
        return None, None, None
    except Exception as e:
        print(f"⚠️ OHLC error for {pair}: {e}")
        return None, None, None

# ==========================================
# STRATEGY 1: ROUND NUMBER REJECTION (LOWERED)
# ==========================================
def check_rejection(pair, current_price):
    global previous_prices, rejection_count, last_signal_time

    if pair not in previous_prices:
        previous_prices[pair] = current_price
        rejection_count[pair] = 0
        last_signal_time[pair] = 0
        return None

    if time.time() - last_signal_time.get(pair, 0) < 60:
        return None

    prev_price = previous_prices[pair]
    target = ROUND_LEVELS.get(pair)
    if target is None:
        return None

    if abs(current_price - target) > 0.0020:
        previous_prices[pair] = current_price
        return None

    if prev_price < target <= current_price:
        rejection_count[pair] += 1
        if rejection_count[pair] >= 1:
            rejection_count[pair] = 0
            previous_prices[pair] = current_price
            last_signal_time[pair] = time.time()
            return "SELL", f"Rejection at {target:.5f}"

    elif prev_price > target >= current_price:
        rejection_count[pair] += 1
        if rejection_count[pair] >= 1:
            rejection_count[pair] = 0
            previous_prices[pair] = current_price
            last_signal_time[pair] = time.time()
            return "BUY", f"Bounce from {target:.5f}"
    else:
        rejection_count[pair] = 0

    previous_prices[pair] = current_price
    return None

# ==========================================
# STRATEGY 2: BREAKOUT (LOWERED)
# ==========================================
def check_breakout(closes, highs, lows):
    if len(closes) < 10:
        return None, None
    
    resistance = max(highs[1:6])
    support = min(lows[1:6])
    current = closes[0]
    
    if current > resistance * 1.0002:
        return "BUY", f"Breakout above {resistance:.5f}"
    elif current < support * 0.9998:
        return "SELL", f"Breakdown below {support:.5f}"
    return None, None

# ==========================================
# STRATEGY 3: RSI DIVERGENCE (LOWERED)
# ==========================================
def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains = 0
    losses = 0
    for i in range(1, period + 1):
        diff = closes[i - 1] - closes[i]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def check_rsi_divergence(closes, highs, lows):
    if len(closes) < 15:
        return None, None
    
    rsi_values = []
    for i in range(10, len(closes)):
        rsi = calculate_rsi(closes[i-14:i+1])
        if rsi is not None:
            rsi_values.append(rsi)
    
    if len(rsi_values) < 10:
        return None, None
    
    price_low = min(closes[:6])
    price_low2 = min(closes[6:12])
    rsi_low = min(rsi_values[:6])
    rsi_low2 = min(rsi_values[6:12])
    
    if price_low2 < price_low and rsi_low2 > rsi_low:
        return "BUY", "RSI Bullish Divergence"
    
    price_high = max(closes[:6])
    price_high2 = max(closes[6:12])
    rsi_high = max(rsi_values[:6])
    rsi_high2 = max(rsi_values[6:12])
    
    if price_high2 > price_high and rsi_high2 < rsi_high:
        return "SELL", "RSI Bearish Divergence"
    
    return None, None

# ==========================================
# STRATEGY 4: MA CROSSOVER (LOWERED)
# ==========================================
def check_ma_crossover(closes):
    if len(closes) < 10:
        return None, None
    
    ma3 = sum(closes[:3]) / 3
    ma10 = sum(closes[:10]) / 10
    prev_ma3 = sum(closes[1:4]) / 3
    prev_ma10 = sum(closes[1:11]) / 10
    
    if prev_ma3 <= prev_ma10 and ma3 > ma10:
        return "BUY", f"MA3 crossed above MA10"
    elif prev_ma3 >= prev_ma10 and ma3 < ma10:
        return "SELL", f"MA3 crossed below MA10"
    return None, None

# ==========================================
# STRATEGY 5: BOLLINGER BANDS (LOWERED)
# ==========================================
def check_bollinger_breakout(closes):
    if len(closes) < 10:
        return None, None
    
    ma = sum(closes[:10]) / 10
    variance = sum([(x - ma) ** 2 for x in closes[:10]]) / 10
    std_dev = variance ** 0.5
    
    upper = ma + (1.5 * std_dev)
    lower = ma - (1.5 * std_dev)
    current = closes[0]
    
    if current > upper:
        return "BUY", f"Price above upper Bollinger Band ({upper:.5f})"
    elif current < lower:
        return "SELL", f"Price below lower Bollinger Band ({lower:.5f})"
    return None, None

# ==========================================
# STRATEGY 6: FIBONACCI (LOWERED)
# ==========================================
def check_fibonacci_retracement(closes, highs, lows):
    if len(closes) < 15:
        return None, None
    
    swing_high = max(closes[:10])
    swing_low = min(closes[:10])
    
    diff = swing_high - swing_low
    fib_618 = swing_high - (diff * 0.618)
    fib_500 = swing_high - (diff * 0.500)
    fib_382 = swing_high - (diff * 0.382)
    
    current = closes[0]
    
    if abs(current - fib_618) / current < 0.003:
        return "BUY" if current > closes[1] else "SELL", f"Price at 61.8% Fib ({fib_618:.5f})"
    elif abs(current - fib_500) / current < 0.003:
        return "BUY" if current > closes[1] else "SELL", f"Price at 50% Fib ({fib_500:.5f})"
    elif abs(current - fib_382) / current < 0.003:
        return "BUY" if current > closes[1] else "SELL", f"Price at 38.2% Fib ({fib_382:.5f})"
    
    return None, None

# ==========================================
# COMBINED SIGNAL GENERATOR
# ==========================================
def get_combined_signal(pair, current_price):
    closes, highs, lows = get_ohlc(pair)
    if closes is None:
        return None, None
    
    rejection = check_rejection(pair, current_price)
    if rejection is not None:
        direction, reason = rejection
        return direction, f"🔴 REJECTION: {reason}" if direction == "SELL" else f"🟢 REJECTION: {reason}"
    
    breakout_dir, breakout_reason = check_breakout(closes, highs, lows)
    if breakout_dir is not None:
        return breakout_dir, f"🚀 BREAKOUT: {breakout_reason}"
    
    rsi_dir, rsi_reason = check_rsi_divergence(closes, highs, lows)
    if rsi_dir is not None:
        return rsi_dir, f"📊 RSI: {rsi_reason}"
    
    ma_dir, ma_reason = check_ma_crossover(closes)
    if ma_dir is not None:
        return ma_dir, f"📈 MA: {ma_reason}"
    
    bb_dir, bb_reason = check_bollinger_breakout(closes)
    if bb_dir is not None:
        return bb_dir, f"📉 BB: {bb_reason}"
    
    fib_dir, fib_reason = check_fibonacci_retracement(closes, highs, lows)
    if fib_dir is not None:
        return fib_dir, f"🌀 FIB: {fib_reason}"
    
    return None, None

# ==========================================
# SEND SIGNAL
# ==========================================
def send_signal(pair, direction, reason):
    real_direction, strength = get_market_strength(pair)
    if real_direction is None or strength is None:
        print(f"⏭️ Skipping {pair} — strength unavailable")
        return

    expiry = random.choice(["1", "2", "3", "5"])
    now = time.time()
    signal_time = time.strftime('%H:%M', time.localtime(now))
    entry_time = time.strftime('%H:%M', time.localtime(now + 120))

    if direction == "BUY":
        dir_display = "🟢 BUY"
    else:
        dir_display = "🔴 SELL"

    message = f"""
🚨 SIGNAL ALERT

OTC Pair: {pair}
Direction: {dir_display}

⏰ Signal Time: {signal_time}
🎯 Entry Time: {entry_time}
Expiry: {expiry} Min

Strength: {strength}% 🔥
Strategy: {reason}
"""

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message})
        print(f"✅ {direction} signal sent for {pair} at {signal_time}")
    except Exception as e:
        print(f"❌ Send error: {e}")

# ==========================================
# GET MARKET STRENGTH
# ==========================================
def get_market_strength(pair):
    try:
        symbol = pair.replace("-OTC", "")
        url = (
            f"https://api.twelvedata.com/time_series"
            f"?symbol={symbol}"
            f"&interval=1min"
            f"&outputsize=20"
            f"&apikey={TWELVE_API_KEY}"
        )
        response = requests.get(url).json()
        closes = [float(candle["close"]) for candle in response["values"]]
        bullish = 0
        bearish = 0
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                bullish += 1
            else:
                bearish += 1
        total = bullish + bearish
        if total == 0:
            return "NEUTRAL", 0
        if bullish > bearish:
            return "BUY", int((bullish / total) * 100)
        else:
            return "SELL", int((bearish / total) * 100)
    except Exception as e:
        print(f"⚠️ Strength error for {pair}: {e}")
        return None, None

# ==========================================
# MAIN BOT LOOP
# ==========================================
def run_bot():
    CHECK_INTERVAL = 4
    print(f"🤖 Bot started. LOW THRESHOLD TEST MODE — Expect frequent signals.")

    while True:
        try:
            all_prices = get_all_prices()
            for pair, price in all_prices.items():
                direction, reason = get_combined_signal(pair, price)
                if direction is not None and reason is not None:
                    send_signal(pair, direction, reason)
                    time.sleep(2)
            
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print(f"❌ Main loop error: {e}")
            time.sleep(CHECK_INTERVAL)

# ==========================================
# FLASK KEEP-ALIVE FOR RENDER
# ==========================================
@app.route('/')
def home():
    return "✅ OTC Multi-Strategy Bot is running!"

@app.route('/ping')
def ping():
    return "pong", 200

# ==========================================
# START BOT IN BACKGROUND
# ==========================================
Thread(target=run_bot, daemon=True).start()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

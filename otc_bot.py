import requests
import random
import time
import pandas as pd
from flask import Flask
from threading import Thread

app = Flask(__name__)

# ===== CONFIGURATION =====
TWELVE_API_KEY = "90ab0986c80046bbb59e117779ffdd18"
BOT_TOKEN = "8612354100:AAFUTlaSiq19yycQWpO70J4d6DEbgF4Kicc"
CHAT_ID = "6280535707"

# ===== POCKET OPTIONS OTC PAIRS =====
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
    "CADCHF-OTC"
]

# ===== ROUND NUMBER LEVELS =====
ROUND_LEVELS = {
    "EURUSD-OTC": 1.10000,
    "GBPUSD-OTC": 1.30000,
    "USDJPY-OTC": 150.000,
    "AUDUSD-OTC": 0.67000,
    "USDCAD-OTC": 1.36000,
    "USDCHF-OTC": 0.89000,
    "EURJPY-OTC": 165.000,
    "GBPJPY-OTC": 195.000,
    "AUDJPY-OTC": 100.000,
    "CADJPY-OTC": 110.000,
    "EURGBP-OTC": 0.85000,
    "EURAUD-OTC": 1.64000,
    "EURCAD-OTC": 1.50000,
    "EURCHF-OTC": 0.98000,
    "GBPAUD-OTC": 1.94000,
    "GBPCAD-OTC": 1.72000,
    "GBPCHF-OTC": 1.16000,
    "AUDCHF-OTC": 0.60000,
    "CADCHF-OTC": 0.65000
}

previous_prices = {}
rejection_count = {}
last_signal_time = {}

# ===== FETCH ALL PAIRS IN ONE API CALL (BATCH) =====
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

# ===== GET REAL MARKET STRENGTH =====
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

# ===== ROUND NUMBER REJECTION LOGIC =====
def check_rejection(pair, current_price):
    global previous_prices, rejection_count, last_signal_time

    if pair not in previous_prices:
        previous_prices[pair] = current_price
        rejection_count[pair] = 0
        last_signal_time[pair] = 0
        return None

    # 2-MINUTE COOLDOWN
    if time.time() - last_signal_time.get(pair, 0) < 120:
        return None

    prev_price = previous_prices[pair]
    target = ROUND_LEVELS.get(pair)
    if target is None:
        return None

    if abs(current_price - target) > 0.0010:
        previous_prices[pair] = current_price
        return None

    if prev_price < target <= current_price:
        rejection_count[pair] += 1
        if rejection_count[pair] >= 2:
            rejection_count[pair] = 0
            previous_prices[pair] = current_price
            last_signal_time[pair] = time.time()
            return "SELL", target

    elif prev_price > target >= current_price:
        rejection_count[pair] += 1
        if rejection_count[pair] >= 2:
            rejection_count[pair] = 0
            previous_prices[pair] = current_price
            last_signal_time[pair] = time.time()
            return "BUY", target
    else:
        rejection_count[pair] = 0

    previous_prices[pair] = current_price
    return None

# ===== SEND SIGNAL WITH GREEN/RED COLOR =====
def send_signal(pair, direction=None, rejection_target=None):
    real_direction, strength = get_market_strength(pair)
    if real_direction is None or strength is None:
        print(f"⏭️ Skipping {pair} — strength unavailable")
        return

    if direction is None:
        direction = real_direction

    expiry = random.choice(["1", "2", "3", "5"])
    now = time.time()
    signal_time = time.strftime('%H:%M', time.localtime(now))
    entry_time = time.strftime('%H:%M', time.localtime(now + 120))

    # === COLOR CODING ===
    if direction == "BUY":
        dir_display = "🟢 BUY"
    else:
        dir_display = "🔴 SELL"

    if rejection_target is not None:
        message = f"""
📊 REJECTION SIGNAL

OTC Pair: {pair}
Direction: {dir_display}

⏰ Signal Time: {signal_time}
🎯 Entry Time: {entry_time}
Expiry: {expiry} Min

Strength: {strength}% 🔥
"""
    else:
        message = f"""
🚨 SIGNAL ALERT

OTC Pair: {pair}
Direction: {dir_display}

⏰ Signal Time: {signal_time}
🎯 Entry Time: {entry_time}
Expiry: {expiry} Min

Strength: {strength}% 🔥
"""

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message})
        print(f"✅ {direction} signal sent for {pair} at {signal_time}")
    except Exception as e:
        print(f"❌ Send error: {e}")

# ===== MAIN BOT LOOP =====
def run_bot():
    CHECK_INTERVAL = 3
    print(f"🤖 Bot started. Checking every {CHECK_INTERVAL} seconds. Instant signal on 2nd rejection.")
    
    while True:
        try:
            all_prices = get_all_prices()
            for pair, price in all_prices.items():
                signal = check_rejection(pair, price)
                if signal is not None:
                    direction, target = signal
                    send_signal(pair, direction, rejection_target=target)
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print(f"❌ Main loop error: {e}")
            time.sleep(CHECK_INTERVAL)

# ===== FLASK KEEP‑ALIVE =====
@app.route('/')
def home():
    return "✅ OTC Instant Rejection Bot is running!"

@app.route('/ping')
def ping():
    return "pong", 200

# ===== START =====
Thread(target=run_bot, daemon=True).start()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

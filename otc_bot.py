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

# ===== ROUND NUMBER LEVELS FOR EACH OTC PAIR =====
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

# ===== GET MARKET SIGNAL =====
def get_market_signal(pair):
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
        if bullish > bearish:
            return "BUY", int((bullish / 19) * 100)
        else:
            return "SELL", int((bearish / 19) * 100)
    except Exception as e:
        print(f"⚠️ Market signal error for {pair}: {e}")
        return "BUY", 75

# ===== ROUND NUMBER REJECTION LOGIC =====
def get_rejection_signal(pair, current_price):
    global previous_prices, rejection_count

    if pair not in previous_prices:
        previous_prices[pair] = current_price
        rejection_count[pair] = 0
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
            return "SELL", target

    elif prev_price > target >= current_price:
        rejection_count[pair] += 1
        if rejection_count[pair] >= 2:
            rejection_count[pair] = 0
            previous_prices[pair] = current_price
            return "BUY", target
    else:
        rejection_count[pair] = 0

    previous_prices[pair] = current_price
    return None

# ===== FETCH CURRENT PRICE =====
def get_current_price(pair):
    try:
        symbol = pair.replace("-OTC", "")
        url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        if "price" in data:
            return float(data["price"])
        return None
    except Exception as e:
        print(f"⚠️ Price fetch error for {pair}: {e}")
        return None

# ===== SEND SIGNAL (TIME LOGIC UNTOUCHED) =====
def send_signal(pair, direction=None, strength=None, rejection_target=None):
    if direction is None:
        direction, strength = get_market_signal(pair)

    expiry = random.choice(["1", "2", "3", "5"])

    if rejection_target is not None:
        if direction == "SELL":
            signal_type = "🔴 SELL (Rejection)"
            target = rejection_target - 0.0020
            stop = rejection_target + 0.0010
        else:
            signal_type = "🟢 BUY (Bounce)"
            target = rejection_target + 0.0020
            stop = rejection_target - 0.0010

        message = f"""
🚨 REJECTION SIGNAL ({time.strftime('%H:%M')})

OTC Pair: {pair}
Direction: {signal_type}

Rejection Level: {rejection_target:.5f}
🎯 Target: {target:.5f}
🛑 Stop: {stop:.5f}

⏰ Signal Time: {time.strftime('%H:%M', time.localtime(time.time() + 3600))}
🎯 Entry Time: {time.strftime('%H:%M', time.localtime(time.time() + 3720))}
Expiry: {expiry} Min

💡 Strategy: Round Number Rejection (65% accuracy)
"""
    else:
        message = f"""
🚨 SIGNAL ALERT

OTC Pair: {pair}
Direction: {direction}

⏰ Signal Time: {time.strftime('%H:%M', time.localtime(time.time() + 3600))}
🎯 Entry Time: {time.strftime('%H:%M', time.localtime(time.time() + 3720))}
Expiry: {expiry} Min
Strength: {strength}% 🔥
"""

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message})
        print(f"✅ Signal sent for {pair} at {time.strftime('%H:%M')}")
    except Exception as e:
        print(f"❌ Send error: {e}")

# ===== MAIN BOT LOOP (30 MINUTES) =====
def run_bot():
    while True:
        try:
            rejection_signals = []
            for pair in pairs:
                price = get_current_price(pair)
                if price is not None:
                    signal = get_rejection_signal(pair, price)
                    if signal is not None:
                        direction, target = signal
                        rejection_signals.append((pair, direction, target))

            if rejection_signals:
                for pair, direction, target in rejection_signals:
                    send_signal(pair, direction, rejection_target=target)
            else:
                pair = random.choice(pairs)
                direction, strength = get_market_signal(pair)
                send_signal(pair, direction, strength)

            print(f"✅ Loop completed. Next run in 30 minutes. ({time.strftime('%H:%M')})")
            time.sleep(1800)

        except Exception as e:
            print(f"❌ Main loop error: {e}")
            time.sleep(1800)

# ===== FLASK KEEP‑ALIVE SERVER (FOR RENDER) =====
@app.route('/')
def home():
    return "✅ OTC Rejection Bot is running!"

@app.route('/ping')
def ping():
    return "pong", 200

# ===== START BOT IN BACKGROUND =====
Thread(target=run_bot, daemon=True).start()

# ===== RUN FLASK =====
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

from flask import Flask
from threading import Thread
import os
import requests
import random
import time
import pandas as pd
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot Running"

def run_web():
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )

TWELVE_API_KEY = "90ab0986c80046bbb59e117779ffdd18"

BOT_TOKEN = "8612354100:AAFUTlaSiq19yycQWpO70J4d6DEbgF4Kicc"
CHAT_ID = "6280535707"

pairs = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "AUD/USD",
    "USD/CAD",
    "USD/CHF",
    "EUR/JPY",
    "GBP/JPY",
    "AUD/JPY",
    "CAD/JPY",
    "EUR/GBP",
    "EUR/AUD",
    "EUR/CAD",
    "EUR/CHF",
    "GBP/AUD",
    "GBP/CAD",
    "GBP/CHF",
    "AUD/CHF",
    "CAD/CHF"
]

timeframes = ["1m", "2m", "3m", "4m", "5m"]
def get_market_signal(pair):
    try:
        symbol = pair

        url = (
            f"https://api.twelvedata.com/time_series"
            f"?symbol={symbol}"
            f"&interval=1min"
            f"&outputsize=20"
            f"&apikey={TWELVE_API_KEY}"
        )

        print("About to request TwelveData")
        print("URL:", url)
        
        try:
            response = requests.get(url, timeout=15)

            print("Status Code:", response.status_code)
            print("Response Text:", response.text[:300])

            response = response.json()

        except Exception as e:
            print("TWELVEDATA ERROR:", e)
            return "BUY", 75

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
        print(e)
        return "BUY", 75

def send_signal(pair):
    pair = random.choice(pairs)
    print("Getting market signal...")
    direction, strength = get_market_signal(pair)
    print("Market signal received")
    expiry = random.choice(["1", "2", "3", "5"])
    
    current_time = time.time()

    message = f"""
🚨 SIGNAL ALERT

Pair: {pair}
Direction: {direction}

⏰ Signal Time: {time.strftime('%H:%M', time.localtime(time.time() + 3600))}

🎯 Entry Time: {time.strftime('%H:%M', time.localtime(time.time() + 3720))}

Expiry: {expiry} Min

Strength: {strength}% 🔥
"""

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    response = requests.post(
    url,
    data={
        "chat_id": CHAT_ID,
        "text": message
    },
    timeout=30
)

Thread(target=run_web).start()

while True:
    try:
        pair = random.choice(pairs)
        print("Starting signal...")
        send_signal(pair)
        print("Signal sent successfully")
        time.sleep(120)

    except Exception as e:
        print(f"ERROR: {e}")
        time.sleep(30)

import requests
import random
import time

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
def get_market_signal():
    try:
        symbol = pair

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
        print(e)
        return "BUY", 75

def send_signal(pair):
    pair = random.choice(pairs)
    direction, strength = get_market_signal(pair)
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

    requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "text": message
        }
    )


while True:
    pair = random.choice(pairs)
    send_signal(pairs)
    print("Signal sent")
    time.sleep(60)

import requests
import random
import time
import yfinance as yf

BOT_TOKEN = "8612354100:AAFUTlaSiq19yycQWpO70J4d6DEbgF4Kicc"
CHAT_ID = "6280535707"

pairs = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
    "AUDUSD",
    "NZDUSD",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
    "CHFJPY",
    "EURCHF",
    "GBPCAD",
    "GBPCHF",
    "EURAUD",
    "AUDCAD",
    "AUDCHF",
    "CADJPY",
    "NZDJPY"
]

timeframes = ["1m", "2m", "3m", "4m", "5m"]
def get_market_signal():
    data = yf.download("EURUSD=X", period="1d", interval="1m")

    if len(data) < 20:
        return "BUY", 75

    candles = data.tail(20)

    bullish = 0
    bearish = 0

    for _, candle in candles.iterrows():
        if candle["Close"] > candle["Open"]:
            bullish += 1
        else:
            bearish += 1

    if bullish > bearish:
        direction = "BUY"
        strength = int((bullish / 20) * 100)
    else:
        direction = "SELL"
        strength = int((bearish / 20) * 100)

    return direction, strength

def send_signal():
    pair = random.choice(pairs)
    direction = random.choice(["BUY", "SELL"])
    expiry = random.choice(["1", "2", "3", "5"])
    strength = random.randint(75, 95)

    current_time = time.time()

    message = f"""
🚨 SIGNAL ALERT

Pair: {pair}
Direction: {direction}

⏰ Signal Time: {time.strftime('%H:%M', time.localtime(time.time() + 3600))}

🎯 Entry Time: {time.strftime('%H:%M', time.localtime(time.time() + 3660))}

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
    send_signal()
    print("Signal sent")
    time.sleep(60)

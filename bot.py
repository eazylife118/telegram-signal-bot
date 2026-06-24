import requests
import random
import time
TWELVE_API_KEY = "90ab0986c80046bbb59e117779ffdd18"

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
    try:
        data = yf.download("EURUSD=X", period="1d", interval="1m")

        closes = data["Close"].tail(20)

        bullish = 0
        bearish = 0

        for i in range(1, len(closes)):
            if closes.iloc[i] > closes.iloc[i - 1]:
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

def send_signal():
    pair = random.choice(pairs)
    direction, strength = get_market_signal()
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
    send_signal()
    print("Signal sent")
    time.sleep(120)

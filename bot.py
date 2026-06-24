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
    return "BUY", 80

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

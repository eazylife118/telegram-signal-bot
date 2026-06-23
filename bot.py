import requests
import time
import random

BOT_TOKEN = "8612354100:AAFUTlaSiq19yycQWpO70J4d6DEbgF4Kicc"
CHAT_ID = "6280535707"

pairs = ["GBPUSD", "EURUSD", "USDJPY", "USDCAD"]

def send_signal():
    pair = random.choice(pairs)
    direction = random.choice(["BUY", "SELL"])

    message = f"""
📊 SIGNAL ALERT

Pair: {pair}

Direction: {direction}

Timeframe: 1m

⚠️ Test Signal
"""

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    requests.post(url, data={
        "chat_id": CHAT_ID,
        "text": message
    })

while True:
    send_signal()
    print("Signal sent")
    time.sleep(60)  # 1 minute

import requests
import time
import random

BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

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

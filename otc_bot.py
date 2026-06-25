import random
import time
import requests

BOT_TOKEN = "8612354100:AAFUTlaSiq19yycQWpO70J4d6DEbgF4Kicc"
CHAT_ID = "6280535707"

pairs = [
    "EUR/USD OTC",
    "GBP/USD OTC",
    "USD/JPY OTC",
    "EUR/JPY OTC",
    "AUD/USD OTC"
]

def generate_signal():
    direction = random.choice(["BUY", "SELL"])
    strength = random.randint(65, 90)

    return direction, strength

def send_signal():
    pair = random.choice(pairs)

    direction, strength = generate_signal()

    message = f"""
🚨 OTC SIGNAL
Pair: {pair}
Direction: {direction}
⏰ Signal Time: {time.strftime('%H:%M', time.localtime(time.time() + 3600))}
🎯 Entry Time: {time.strftime('%H:%M', time.localtime(time.time() + 3720))}
Strength: {strength}% 🔥
Expiry: 1 Min
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
    print("OTC signal sent")
    time.sleep(60)

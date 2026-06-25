from flask import Flask
from threading import Thread
import os
import random
import time
import requests
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot Running"

def run_web():
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )


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
    direction = random.choice(["🟢⬆️", "🔴⬇️"])
    strength = random.randint(65, 90)

    return direction, strength

def send_signal():
    pair = random.choice(pairs)

    direction, strength = generate_signal()

    message = f"""
📊 OTC
Pair: {pair}
Direction: {direction}
⏰ Signal: {time.strftime('%H:%M', time.localtime(time.time() + 3600))}
⚡ Entry : {time.strftime('%H:%M', time.localtime(time.time() + 3720))}
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
Thread(target=run_web).start()

while True:
    send_signal()
    print("OTC signal sent")
    time.sleep(60)
    
except Exception as e:
    print(f"ERROR: {e}")
    time.sleep(30)

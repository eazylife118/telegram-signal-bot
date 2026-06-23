import requests
import time
import random

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

def send_signal():
pair = random.choice(pairs)
direction = random.choice([“BUY”, “SELL”])
timeframe = random.choice(timeframes)

message = f"""

📊 SIGNAL ALERT

Pair: {pair}

Direction: {direction}

Timeframe: {timeframe}

⏰ Entry Time: {time.strftime(’%H:%M’)}

⚠️ Test Signal
“””

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
print(“Signal sent”)
time.sleep(60)


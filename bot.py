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
direction = random.choice([“BUY”,“SELL”])
expiry = random.choice([“1”,“2”,“3”,“5”])
strength = random.randint(75,95)

current_time = time.time()
message = f"""

🚨 SIGNAL ALERT

Pair: {pair}
Direction: {direction}

Entry Time: {time.strftime(’%H:%M’)}

Trade Time: {time.strftime(’%H:%M’, time.localtime(current_time + 60))}

Expiry: {expiry} Min

Strength: {strength}% 🔥
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

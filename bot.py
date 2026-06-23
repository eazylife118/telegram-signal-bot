import requests
import time
import random
BOT_TOKEN = "8612354100:AAFUTlaSiq19yycQWpO70J4d6DEbgF4Kicc"
CHAT_ID = "6280535707"

pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

directions = ["BUY", "SELL"]

times = ["15s", "30s", "1m", "2m", "3m", "5m", "15m"]

while True:

    pair = random.choice(pairs)

    direction = random.choice(directions)

    tf = random.choice(times)

    message = f"""📊 SIGNAL ALERT

Pair: {pair}

Direction: {direction}

Timeframe: {tf}

⚠️ Test Signal

"""

    requests.get(

        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",

        params={

            "chat_id": CHAT_ID,

            "text": message

        }

    )

    print("Signal sent")

    time.sleep(120)

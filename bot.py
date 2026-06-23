import requests
import time
BOT_TOKEN = "8612354100:AAEuBmuJGHvf9Lw05cC7VafVs0OMXtDIAGs"
CHAT_ID = "6280535707"

while True:
    requests.get(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        params={
            "chat_id": CHAT_ID,
            "text": "✅ Test message from Render"
        }
    )

    print("Message sent")
    time.sleep(60)

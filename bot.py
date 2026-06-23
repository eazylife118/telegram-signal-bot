import requests
import os
import time

BOT_TOKEN = os.getenv("8612354100:AAFUTlaSiq19yycQWpO70J4d6DEbgF4Kicc")
CHAT_ID = "6280535707"

def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    r = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "text": text
        }
    )

    print(r.text)

send_message("✅ Bot is working!")

while True:
    time.sleep(60)

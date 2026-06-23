import os
import requests
import time

BOT_TOKEN = os.getenv(“BOT_TOKEN”)
CHAT_ID = “YOUR_CHAT_ID”

def send_message(text):
url = f”https://api.telegram.org/bot{BOT_TOKEN}/sendMessage”
requests.post(url, data={
“chat_id”: CHAT_ID,
“text”: text
})

send_message(“✅ Signal Bot Started Successfully!”)

while True:
time.sleep(60)

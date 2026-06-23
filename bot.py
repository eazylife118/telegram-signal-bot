import os
import requests
import time

BOT_TOKEN = os.getenv("8612354100:AAFUTlaSiq19yycQWpO70J4d6DEbgF4Kicc")
CHAT_ID = "6280535707"

def send_message(text):
url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
requests.post(url, data={
"chat_id": CHAT_ID,
"text": text
})

send_message("✅ Signal Bot Started Successfully!")

while True:
time.sleep(60)

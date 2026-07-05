import requests

TOKEN = "8846196749:AAG9CP2fNqw4vSt0l4MAUQK3lc783VR0Hb0"
CHANNEL_ID = "-1004324805205"

def send_test_message():
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {
        "chat_id": CHANNEL_ID,
        "text": "✅ Test message from bot to channel!",
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            print("✅ Message sent successfully!")
        else:
            print("❌ Failed:", response.text)
    except Exception as e:
        print("❌ Error:", e)

send_test_message()

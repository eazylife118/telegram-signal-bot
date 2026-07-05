import requests

# ==========================================
# YOUR CREDENTIALS
# ==========================================
TOKEN = "8846196749:AAHbfxiFqW49b0Ap2yZ2Vt_qE1wlfQ4os6s"
CHAT_ID = "6280535707"
CHANNEL_ID = "-1004324805205"  # Your channel ID

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
            print("Response:", response.json())
        else:
            print("❌ Failed to send message.")
            print("Error:", response.text)
    except Exception as e:
        print("❌ Error:", e)

if __name__ == "__main__":
    send_test_message()

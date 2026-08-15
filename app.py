import os
import json
import time
import threading
import requests
import websocket
from flask import Flask
# ============================================================
# RENDER VARIABLES
# ============================================================
PO_UID = os.getenv("PO_UID", "").strip()
CI = os.getenv("CI", "").strip()
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()
WS_URL = "wss://api-msk.po.market"
# ============================================================
# FLASK
# ============================================================
app = Flask(__name__)
@app.route("/")
def home():
    return "PO live data test bot is running."
@app.route("/ping")
def ping():
    return "OK"
# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram variables are missing.")
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message
            },
            timeout=15
        )
        print("Telegram:", response.status_code, response.text)
        return response.ok
    except Exception as e:
        print("Telegram error:", e)
        return False
# ============================================================
# SIMPLE LIVE-DATA EXTRACTION
# ============================================================
signal_sent = False
previous_price = None
def find_pair(data):
    """
    Look for a pair/symbol name inside real messages received
    from the server.
    """
    if isinstance(data, dict):
        # Common possible names
        for key in (
            "symbol",
            "pair",
            "asset",
            "instrument",
            "asset_name",
            "active",
            "ticker"
        ):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        # Search nested objects
        for value in data.values():
            result = find_pair(value)
            if result:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_pair(item)
            if result:
                return result
    return None
def find_price(data):
    """
    Extract a REAL numeric price from the received message.
    """
    if isinstance(data, dict):
        for key in (
            "price",
            "rate",
            "quote",
            "close",
            "last",
            "value"
        ):
            value = data.get(key)
            try:
                if value is not None:
                    number = float(value)
                    if number > 0:
                        return number
            except (ValueError, TypeError):
                pass
        # Search nested data
        for value in data.values():
            result = find_price(value)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_price(item)
            if result is not None:
                return result
    return None
# ============================================================
# SIMPLE SIGNAL
# ============================================================
def process_live_data(raw_message):
    global signal_sent
    global previous_price
    if signal_sent:
        return
    try:
        data = json.loads(raw_message)
    except Exception:
        return
    pair = find_pair(data)
    price = find_price(data)
    # We require REAL received data.
    if not pair or price is None:
        return
    print("REAL DATA")
    print("Pair:", pair)
    print("Price:", price)
    # Need two real price observations before deciding direction.
    if previous_price is None:
        previous_price = price
        return
    if price > previous_price:
        direction = "BUY"
    elif price < previous_price:
        direction = "SELL"
    else:
        previous_price = price
        return
    message = (
        "🔔 SIGNAL\n\n"
        f"Pair: {pair}\n"
        f"Direction: {direction}\n"
        "Expiry: 1 Min"
    )
    if send_telegram(message):
        signal_sent = True
        print("ONE TEST SIGNAL SENT.")
    previous_price = price
# ============================================================
# WEBSOCKET
# ============================================================
def on_open(ws):
    print("✅ WebSocket connected.")
    # Send the available session values.
    # The server may require a different protocol/message format;
    # this test deliberately does not invent one.
    auth_message = {
        "uid": PO_UID,
        "ci": CI
    }
    try:
        ws.send(json.dumps(auth_message))
        print("Authentication data sent.")
    except Exception as e:
        print("Send error:", e)
def on_message(ws, message):
    print("📡 DATA RECEIVED:")
    print(message[:1000])
    process_live_data(message)
def on_error(ws, error):
    print("❌ WebSocket error:")
    print(error)
def on_close(ws, close_status_code, close_msg):
    print("🔌 Connection closed.")
    print("Code:", close_status_code)
    print("Message:", close_msg)
def websocket_worker():
    if not PO_UID:
        print("❌ PO_UID is missing.")
        return
    if not CI:
        print("❌ CI is missing.")
        return
    print("Starting PO live-data test...")
    print("Endpoint:", WS_URL)
    print("PO_UID: supplied")
    print("CI: supplied")
    while True:
        try:
            ws = websocket.WebSocketApp(
                WS_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever(
                ping_interval=20,
                ping_timeout=10
            )
        except Exception as e:
            print("Connection exception:", e)
        print("Retrying in 10 seconds...")
        time.sleep(10)
# ============================================================
# START
# ============================================================
if __name__ == "__main__":
    thread = threading.Thread(
        target=websocket_worker,
        daemon=True
    )
    thread.start()
    port = int(os.getenv("PORT", "10000"))
    app.run(
        host="0.0.0.0",
        port=port
    )

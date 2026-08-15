import websocket

WS_URL = "wss://api-msk.po.market"


def on_open(ws):
    print("✅ Connected to:", WS_URL)


def on_message(ws, message):
    print("📩 Received:")
    print(message)


def on_error(ws, error):
    print("❌ Error:", error)


def on_close(ws, close_status_code, close_msg):
    print("🔌 Connection closed")
    print("Code:", close_status_code)
    print("Message:", close_msg)


if __name__ == "__main__":
    print("Connecting...")
    
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws.run_forever()

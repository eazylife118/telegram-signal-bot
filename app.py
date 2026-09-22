import os
import json
import websocket
import requests

# ============================================================
# TELEGRAM SETTINGS — FROM RENDER ENVIRONMENT VARIABLES
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise RuntimeError("❌ BOT_TOKEN or CHAT_ID is missing from Render Environment Variables")


# ============================================================
# DERIV API
# ============================================================

DERIV_WS = "wss://api.derivws.com/trading/v1/options/ws/public"


def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    try:
        response = requests.post(url, data=payload, timeout=15)

        if response.ok:
            print("✅ Telegram message sent")
        else:
            print("❌ Telegram error:", response.text)

    except Exception as e:
        print("❌ Telegram connection error:", e)


def on_open(ws):
    print("✅ Connected to Deriv API")

    request = {
        "active_symbols": "brief",
        "req_id": 1
    }

    ws.send(json.dumps(request))

    print("🔎 Detecting active currency pairs...")


def on_message(ws, message):
    data = json.loads(message)

    if data.get("msg_type") == "active_symbols":

        forex_pairs = []

        for item in data.get("active_symbols", []):
            if item.get("market") == "forex":

                symbol = item.get("underlying_symbol")
                name = item.get("underlying_symbol_name")

                if symbol and name:
                    forex_pairs.append((symbol, name))

        # ====================================================
        # TERMINAL RESPONSE
        # ====================================================

        print("\n💱 ACTIVE FOREX PAIRS")
        print("=" * 35)

        for symbol, name in forex_pairs:
            print(f"{name} → {symbol}")

        print("=" * 35)
        print(f"Total Forex pairs detected: {len(forex_pairs)}")

        # ====================================================
        # TELEGRAM RESPONSE
        # ====================================================

        telegram_message = "💱 DERIV ACTIVE FOREX PAIRS\n\n"

        for symbol, name in forex_pairs:
            telegram_message += f"{name} → {symbol}\n"

        telegram_message += (
            f"\n━━━━━━━━━━━━━━━━━━\n"
            f"✅ Total Forex pairs detected: {len(forex_pairs)}"
        )

        send_telegram(telegram_message)

        ws.close()


def on_error(ws, error):
    print("❌ API Error:", error)


def on_close(ws, close_status_code, close_msg):
    print("🔌 Connection closed")


# ============================================================
# START
# ============================================================

ws = websocket.WebSocketApp(
    DERIV_WS,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
)

print("🔄 Connecting to Deriv...")

ws.run_forever()

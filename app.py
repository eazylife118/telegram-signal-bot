import os
import json
import requests
import websocket


# ============================================================
# RENDER ENVIRONMENT VARIABLES
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

DERIV_AUTH_TOKEN = os.getenv("DERIV_AUTH_TOKEN")
DERIV_ACCOUNT_ID = os.getenv("DERIV_ACCOUNT_ID")
DERIV_APP_ID = os.getenv("DERIV_APP_ID")


# ============================================================
# CHECK REQUIRED VARIABLES
# ============================================================

required = {
    "BOT_TOKEN": BOT_TOKEN,
    "CHAT_ID": CHAT_ID,
    "DERIV_AUTH_TOKEN": DERIV_AUTH_TOKEN,
    "DERIV_ACCOUNT_ID": DERIV_ACCOUNT_ID,
    "DERIV_APP_ID": DERIV_APP_ID,
}

missing = [name for name, value in required.items() if not value]

if missing:
    raise RuntimeError(
        "❌ Missing Render Environment Variable(s): "
        + ", ".join(missing)
    )


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    try:
        response = requests.post(
            url,
            data=payload,
            timeout=15
        )

        if response.ok:
            print("✅ Telegram message sent")
        else:
            print("❌ Telegram error")

    except Exception as e:
        print("❌ Telegram connection error:", e)


# ============================================================
# STEP 1 — GET PRIVATE WEBSOCKET URL
# ============================================================

def get_private_websocket_url():

    otp_url = (
        "https://api.derivws.com"
        f"/trading/v1/options/accounts/{DERIV_ACCOUNT_ID}/otp"
    )

    headers = {
        "Authorization": f"Bearer {DERIV_AUTH_TOKEN}",
        "Deriv-App-ID": DERIV_APP_ID
    }

    print("🔐 Requesting private Deriv WebSocket access...")

    try:
        response = requests.post(
            otp_url,
            headers=headers,
            timeout=15
        )

        if not response.ok:
            print("❌ Failed to obtain private WebSocket access")
            print("HTTP status:", response.status_code)

            try:
                error_data = response.json()

                for error in error_data.get("errors", []):
                    print(
                        "Deriv error:",
                        error.get("message", "Unknown error")
                    )

            except Exception:
                pass

            return None

        data = response.json()

        ws_url = data.get("data", {}).get("url")

        if not ws_url:
            print("❌ Deriv did not return a WebSocket URL")
            return None

        print("✅ Private WebSocket URL obtained")

        return ws_url

    except Exception as e:
        print("❌ OTP request error:", e)
        return None


# ============================================================
# STEP 2 — CONNECT TO PRIVATE WEBSOCKET
# ============================================================

def on_open(ws):

    print("✅ PRIVATE DERIV WEBSOCKET CONNECTED")

    # Read account balance only.
    # NO trade is placed.

    request = {
        "balance": 1,
        "req_id": 1
    }

    ws.send(json.dumps(request))

    print("🔎 Verifying authenticated account connection...")


def on_message(ws, message):

    try:
        data = json.loads(message)

    except Exception:
        print("❌ Invalid Deriv response")
        return


    # ========================================================
    # AUTHENTICATED BALANCE RESPONSE
    # ========================================================

    if data.get("msg_type") == "balance":

        balance_data = data.get("balance", {})

        currency = balance_data.get("currency")

        print("✅ PRIVATE DERIV ACCOUNT VERIFIED")
        print("💰 Account currency:", currency)

        telegram_message = (
            "🔐 PRIVATE DERIV API CONNECTED\n\n"
            "✅ Authentication successful\n"
            "✅ Private WebSocket connected\n"
            "✅ Deriv account verified\n\n"
            "🚫 No trade was placed"
        )

        send_telegram(telegram_message)

        ws.close()


    # ========================================================
    # DERIV API ERROR
    # ========================================================

    elif data.get("error"):

        error = data.get("error", {})

        message_text = error.get(
            "message",
            "Unknown Deriv API error"
        )

        print("❌ Deriv API error:", message_text)

        send_telegram(
            "❌ PRIVATE DERIV API CONNECTION FAILED\n\n"
            f"Reason: {message_text}"
        )

        ws.close()


# ============================================================
# ERROR
# ============================================================

def on_error(ws, error):

    print("❌ WebSocket error:", error)


# ============================================================
# CLOSE
# ============================================================

def on_close(ws, close_status_code, close_msg):

    print("🔌 Private Deriv WebSocket closed")


# ============================================================
# START
# ============================================================

print("🚀 Starting private Deriv API test...")

private_ws_url = get_private_websocket_url()

if not private_ws_url:

    send_telegram(
        "❌ PRIVATE DERIV API CONNECTION FAILED\n\n"
        "Could not obtain authenticated WebSocket access."
    )

    raise SystemExit


ws = websocket.WebSocketApp(
    private_ws_url,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
)


print("🔄 Connecting to private Deriv account...")

ws.run_forever()

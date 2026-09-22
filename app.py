import os
import json
import time
import threading
import requests
import websocket
import telebot

from flask import Flask


# ============================================================
# CONFIGURATION
# ============================================================

DERIV_AUTH_TOKEN = os.getenv("DERIV_AUTH_TOKEN")
DERIV_APP_ID = os.getenv("DERIV_APP_ID")

BOT_TOKEN = os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ============================================================
# VALIDATE ENVIRONMENT
# ============================================================

required = {
    "DERIV_AUTH_TOKEN": DERIV_AUTH_TOKEN,
    "DERIV_APP_ID": DERIV_APP_ID,
    "BOT_TOKEN": BOT_TOKEN,
    "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
}

missing = [
    name for name, value in required.items()
    if not value
]

if missing:
    raise RuntimeError(
        "❌ Missing Render Environment Variable(s): "
        + ", ".join(missing)
    )


# ============================================================
# FLASK KEEP-ALIVE
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return (
        "✅ Deriv Volatility Index connection test "
        "is running!"
    )


@app.route("/ping")
def ping():
    return "OK"


# ============================================================
# TELEGRAM
# ============================================================

bot = telebot.TeleBot(BOT_TOKEN)


def send_telegram(message):

    try:

        bot.send_message(
            TELEGRAM_CHAT_ID,
            message,
            disable_web_page_preview=True
        )

    except Exception as e:

        print("Telegram error:", e)


# ============================================================
# GET ALL OPTIONS ACCOUNTS
# ============================================================

def get_options_accounts():

    print("🔎 Getting Deriv Options accounts...")

    url = (
        "https://api.derivws.com"
        "/trading/v1/options/accounts"
    )

    headers = {
        "Authorization": (
            f"Bearer {DERIV_AUTH_TOKEN}"
        ),
        "Deriv-App-ID": DERIV_APP_ID,
        "Content-Type": "application/json",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    print(
        "Accounts HTTP status:",
        response.status_code
    )

    if response.status_code != 200:

        print("Accounts response:")
        print(response.text)

        raise RuntimeError(
            "❌ Could not retrieve Deriv Options "
            f"accounts: {response.status_code} "
            f"{response.text}"
        )

    result = response.json()

    print("Accounts response received.")

    data = result.get("data")

    if data is None:

        print(
            json.dumps(
                result,
                indent=2
            )
        )

        raise RuntimeError(
            "❌ Deriv returned no account data."
        )

    # The API may return either a list or a single
    # account object depending on the response.
    if isinstance(data, dict):

        accounts = [data]

    elif isinstance(data, list):

        accounts = data

    else:

        raise RuntimeError(
            "❌ Unexpected account response format."
        )

    if not accounts:

        raise RuntimeError(
            "❌ No Options trading accounts were "
            "returned for this PAT."
        )

    print(
        f"✅ Found {len(accounts)} Options account(s)."
    )

    return accounts


# ============================================================
# SELECT ACCOUNT
# ============================================================

def select_account(accounts):

    print("\nAvailable Options accounts:")

    for account in accounts:

        account_id = account.get(
            "account_id",
            "UNKNOWN"
        )

        account_type = account.get(
            "account_type",
            "UNKNOWN"
        )

        currency = account.get(
            "currency",
            "UNKNOWN"
        )

        status = account.get(
            "status",
            "UNKNOWN"
        )

        print(
            f"  {account_id} | "
            f"{account_type} | "
            f"{currency} | "
            f"{status}"
        )

    # Prefer DEMO for this connection test.
    demo_accounts = [
        account
        for account in accounts
        if str(
            account.get(
                "account_type",
                ""
            )
        ).lower() == "demo"
    ]

    if demo_accounts:

        selected = demo_accounts[0]

    else:

        selected = accounts[0]

    account_id = selected.get("account_id")

    if not account_id:

        raise RuntimeError(
            "❌ Selected account has no account_id."
        )

    return selected


# ============================================================
# GET AUTHENTICATED WEBSOCKET URL
# ============================================================

def get_authenticated_websocket_url(account_id):

    print(
        f"🔐 Requesting WebSocket OTP for "
        f"{account_id}..."
    )

    url = (
        "https://api.derivws.com"
        f"/trading/v1/options/accounts/"
        f"{account_id}/otp"
    )

    headers = {
        "Authorization": (
            f"Bearer {DERIV_AUTH_TOKEN}"
        ),
        "Deriv-App-ID": DERIV_APP_ID,
        "Content-Type": "application/json",
    }

    response = requests.post(
        url,
        headers=headers,
        timeout=20
    )

    print(
        "OTP HTTP status:",
        response.status_code
    )

    if response.status_code != 200:

        print("OTP response:")
        print(response.text)

        raise RuntimeError(
            "❌ OTP request failed: "
            f"{response.status_code} "
            f"{response.text}"
        )

    result = response.json()

    # Current Deriv response format:
    # {
    #   "data": {
    #       "url": "wss://..."
    #   }
    # }

    data = result.get("data", {})

    websocket_url = data.get("url")

    if not websocket_url:

        print(
            "Full OTP response:"
        )

        print(
            json.dumps(
                result,
                indent=2
            )
        )

        raise RuntimeError(
            "❌ Deriv did not return a WebSocket URL."
        )

    return websocket_url


# ============================================================
# DERIV CONNECTION TEST
# ============================================================

class DerivConnectionTest:

    def __init__(self):

        self.ws = None

        self.tick_count = 0

        self.volatility_symbols = []

        self.subscribed_symbols = set()

        self.contract_results = {}

        self.first_tick_sent = False


    # --------------------------------------------------------
    # SEND
    # --------------------------------------------------------

    def send(self, payload):

        if (
            self.ws
            and self.ws.sock
            and self.ws.sock.connected
        ):

            self.ws.send(
                json.dumps(payload)
            )


    # --------------------------------------------------------
    # OPEN
    # --------------------------------------------------------

    def on_open(self, ws):

        print(
            "🟢 Authenticated WebSocket connected."
        )

        send_telegram(
            "🟢 DERIV CONNECTION SUCCESS\n\n"
            "Authenticated WebSocket connected.\n"
            "⏳ Requesting Volatility Index data..."
        )

        # Request active symbols.
        self.send({
            "active_symbols": "brief",
            "req_id": 1
        })


    # --------------------------------------------------------
    # MESSAGE
    # --------------------------------------------------------

    def on_message(self, ws, message):

        try:

            data = json.loads(message)

        except Exception:

            print(
                "⚠️ Received non-JSON message."
            )

            return


        # ----------------------------------------------------
        # API ERROR
        # ----------------------------------------------------

        if "error" in data:

            error = data.get(
                "error",
                {}
            )

            print(
                "❌ Deriv API error:"
            )

            print(
                json.dumps(
                    error,
                    indent=2
                )
            )

            send_telegram(
                "❌ DERIV API ERROR\n\n"
                + json.dumps(
                    error,
                    indent=2
                )[:3500]
            )

            return


        msg_type = data.get(
            "msg_type"
        )


        # ----------------------------------------------------
        # ACTIVE SYMBOLS
        # ----------------------------------------------------

        if msg_type == "active_symbols":

            symbols = data.get(
                "active_symbols",
                []
            )

            print(
                f"📊 Active symbols received: "
                f"{len(symbols)}"
            )

            volatility = []

            for item in symbols:

                symbol = item.get(
                    "underlying_symbol",
                    ""
                )

                name = item.get(
                    "underlying_symbol_name",
                    symbol
                )

                market = item.get(
                    "market",
                    ""
                )

                symbol_type = item.get(
                    "underlying_symbol_type",
                    ""
                )

                text = (
                    f"{symbol} "
                    f"{name} "
                    f"{market} "
                    f"{symbol_type}"
                ).lower()

                # Detect Volatility/Synthetic markets.
                if (
                    "volatility" in text
                    or "volatility index" in text
                    or "synthetic" in text
                    or "1hz" in symbol.lower()
                    or symbol.lower().endswith("v")
                ):

                    volatility.append({
                        "symbol": symbol,
                        "name": name
                    })


            # Remove duplicates.
            unique = {}

            for item in volatility:

                unique[
                    item["symbol"]
                ] = item


            self.volatility_symbols = list(
                unique.values()
            )


            print(
                f"📈 Volatility candidates: "
                f"{len(self.volatility_symbols)}"
            )


            if not self.volatility_symbols:

                send_telegram(
                    "⚠️ CONNECTED\n\n"
                    "No Volatility Index symbols "
                    "were identified."
                )

                return


            lines = []

            for item in self.volatility_symbols:

                lines.append(
                    f"• {item['name']} "
                    f"({item['symbol']})"
                )


            send_telegram(
                "📊 VOLATILITY INDICES FOUND\n\n"
                + "\n".join(lines[:50])
                + "\n\n"
                "⏳ Requesting live tick data..."
            )


            # Subscribe directly to live ticks.
            #
            # We deliberately do NOT place trades.
            #
            for index, item in enumerate(
                self.volatility_symbols[:50]
            ):

                symbol = item["symbol"]

                if symbol in self.subscribed_symbols:
                    continue

                self.send({
                    "ticks": symbol,
                    "subscribe": 1,
                    "req_id": 5000 + index
                })

                self.subscribed_symbols.add(
                    symbol
                )

            return


        # ----------------------------------------------------
        # TICK
        # ----------------------------------------------------

        if msg_type == "tick":

            tick = data.get(
                "tick",
                {}
            )

            symbol = tick.get(
                "symbol",
                "UNKNOWN"
            )

            quote = tick.get(
                "quote"
            )

            epoch = tick.get(
                "epoch"
            )

            self.tick_count += 1


            print(
                f"📈 TICK "
                f"{self.tick_count}: "
                f"{symbol} = {quote} "
                f"@ {epoch}"
            )


            # Send Telegram confirmation only once.
            if not self.first_tick_sent:

                self.first_tick_sent = True

                send_telegram(
                    "🟢 LIVE VOLATILITY TICK RECEIVED\n\n"
                    f"Index: {symbol}\n"
                    f"Price: {quote}\n"
                    f"Epoch: {epoch}\n\n"
                    "✅ Account discovery works.\n"
                    "✅ Authenticated connection works.\n"
                    "✅ Live Volatility data works.\n"
                    "🚫 NO TRADE WAS PLACED."
                )

            return


        # ----------------------------------------------------
        # AUTHORIZE
        # ----------------------------------------------------

        if msg_type == "authorize":

            print(
                "✅ Authorization message received."
            )

            return


        # ----------------------------------------------------
        # BALANCE
        # ----------------------------------------------------

        if msg_type == "balance":

            balance_data = data.get(
                "balance",
                {}
            )

            print(
                "💰 Balance:",
                balance_data
            )

            return


    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    def on_error(self, ws, error):

        print(
            "❌ WebSocket error:",
            error
        )

        send_telegram(
            "❌ DERIV WEBSOCKET ERROR\n\n"
            f"{error}"
        )


    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def on_close(
        self,
        ws,
        close_status_code,
        close_msg
    ):

        print(
            "🔴 WebSocket closed:",
            close_status_code,
            close_msg
        )


    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    def start(self):

        while True:

            try:

                print(
                    "\n===================================="
                )

                print(
                    "DERIV AUTOMATIC ACCOUNT CONNECTION"
                )

                print(
                    "===================================="
                )


                # --------------------------------------------
                # STEP 1: GET ACCOUNTS AUTOMATICALLY
                # --------------------------------------------

                accounts = get_options_accounts()


                # --------------------------------------------
                # STEP 2: SELECT ACCOUNT
                # --------------------------------------------

                account = select_account(
                    accounts
                )

                account_id = account.get(
                    "account_id"
                )

                account_type = account.get(
                    "account_type",
                    "unknown"
                )

                currency = account.get(
                    "currency",
                    "unknown"
                )

                balance = account.get(
                    "balance",
                    "unknown"
                )


                print(
                    "\n===================================="
                )

                print(
                    "SELECTED OPTIONS ACCOUNT"
                )

                print(
                    "===================================="
                )

                print(
                    f"Account ID: {account_id}"
                )

                print(
                    f"Type: {account_type}"
                )

                print(
                    f"Currency: {currency}"
                )

                print(
                    f"Balance: {balance}"
                )


                send_telegram(
                    "🔐 DERIV OPTIONS ACCOUNT FOUND\n\n"
                    f"Account: {account_id}\n"
                    f"Type: {account_type}\n"
                    f"Currency: {currency}\n"
                    f"Balance: {balance}\n\n"
                    "⏳ Connecting to live market data..."
                )


                # --------------------------------------------
                # STEP 3: GET OTP / WEBSOCKET URL
                # --------------------------------------------

                websocket_url = (
                    get_authenticated_websocket_url(
                        account_id
                    )
                )


                print(
                    "✅ Authenticated WebSocket URL obtained."
                )


                # --------------------------------------------
                # STEP 4: CONNECT
                # --------------------------------------------

                self.ws = websocket.WebSocketApp(
                    websocket_url,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                )


                self.ws.run_forever(
                    ping_interval=25,
                    ping_timeout=10
                )


            except Exception as e:

                print(
                    "❌ Connection cycle failed:"
                )

                print(e)


                send_telegram(
                    "🔴 DERIV CONNECTION FAILED\n\n"
                    f"{e}"
                )


            print(
                "⏳ Reconnecting in 10 seconds..."
            )

            time.sleep(10)


# ============================================================
# START DERIV TEST
# ============================================================

deriv_test = DerivConnectionTest()


threading.Thread(
    target=deriv_test.start,
    daemon=True
).start()


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )

import os
import asyncio
import threading
import logging
import telebot

from flask import Flask
from pocket_option import PocketOptionClient
from pocket_option.constants import Regions
from pocket_option.contrib.default_init import default_init
from pocket_option.models import Asset, AuthorizationData


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# RENDER / FLASK
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "Pocket Option Telegram Bot is running."


@app.route("/ping")
def ping():
    return "OK"


# ============================================================
# TELEGRAM
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing from Render Environment Variables.")

bot = telebot.TeleBot(BOT_TOKEN)


# ============================================================
# POCKET OPTION CREDENTIALS
# ============================================================

PO_SESSION = os.getenv("PO_SESSION")
PO_UID = os.getenv("PO_UID")

if not PO_SESSION:
    raise RuntimeError("PO_SESSION is missing from Render Environment Variables.")

if not PO_UID:
    raise RuntimeError("PO_UID is missing from Render Environment Variables.")


# ============================================================
# POCKET OPTION SETTINGS
# ============================================================

# This is ONLY used to subscribe to an OTC asset and verify
# that the Pocket Option connection is alive.
#
# NO TRADE WILL BE OPENED.
ASSET = Asset.AUDCAD_otc

CANDLE_PERIOD = 60

IS_DEMO = 1


# ============================================================
# CONNECTION STATUS
# ============================================================

pocket_client = None

pocket_loop = None

pocket_connected = False

pocket_connecting = False

pocket_error = None


# ============================================================
# POCKET OPTION CONNECTION
# ============================================================

async def connect_pocket_option():

    global pocket_client
    global pocket_connected
    global pocket_connecting
    global pocket_error

    pocket_connecting = True
    pocket_connected = False
    pocket_error = None

    try:

        logger.info("========================================")
        logger.info("POCKET OPTION CONNECTION TEST")
        logger.info("========================================")

        logger.info("Creating Pocket Option client...")

        client = PocketOptionClient(logger=True)

        pocket_client = client

        logger.info("Loading authorization...")

        authorization = AuthorizationData.model_validate(
            {
                "session": PO_SESSION,
                "isDemo": IS_DEMO,
                "uid": int(PO_UID),
                "platform": 2,
                "isFastHistory": True,
                "isOptimized": True,
            }
        )

        logger.info("Initializing OTC asset subscription...")

        default_init(
            client,
            authorization=authorization,
            sub_assets=[ASSET],
            sub_period=CANDLE_PERIOD,
        )

        logger.info("Connecting to Pocket Option...")

        await client.connect(Regions.DEMO)

        logger.info("Waiting for Pocket Option authorization...")

        # Wait for the SDK to confirm authorization.
        await asyncio.wait_for(
            client.authorized_event.wait(),
            timeout=30,
        )

        pocket_connected = True
        pocket_connecting = False
        pocket_error = None

        logger.info("========================================")
        logger.info("POCKET OPTION CONNECTED SUCCESSFULLY")
        logger.info("========================================")

        logger.info(
            "OTC asset subscription: %s",
            ASSET,
        )

        logger.info("Automatic trading: OFF")

        # Keep the connection alive.
        while True:
            await asyncio.sleep(30)

    except asyncio.TimeoutError:

        pocket_connected = False
        pocket_connecting = False
        pocket_error = "Pocket Option authorization timed out."

        logger.error(
            "Pocket Option authorization timed out."
        )

    except Exception as e:

        pocket_connected = False
        pocket_connecting = False
        pocket_error = str(e)

        logger.exception(
            "Pocket Option connection failed."
        )


# ============================================================
# START POCKET OPTION IN BACKGROUND
# ============================================================

def pocket_thread():

    global pocket_loop

    pocket_loop = asyncio.new_event_loop()

    asyncio.set_event_loop(
        pocket_loop
    )

    try:

        pocket_loop.run_until_complete(
            connect_pocket_option()
        )

    except Exception:

        logger.exception(
            "Pocket Option background thread stopped."
        )

    finally:

        try:
            pocket_loop.close()
        except Exception:
            pass


# ============================================================
# TELEGRAM /START
# ============================================================

@bot.message_handler(commands=["start"])
def start_command(message):

    logger.info(
        "Telegram /start received from chat %s",
        message.chat.id,
    )

    # --------------------------------------------------------
    # TELEGRAM IS DEFINITELY CONNECTED IF WE RECEIVED /START
    # --------------------------------------------------------

    if pocket_connected:

        text = (
            "🟢 BOT CONNECTION TEST\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "📱 Telegram: ✅ CONNECTED\n"
            "🟢 Pocket Option: ✅ CONNECTED\n"
            "💱 OTC market: ✅ CONNECTED\n"
            "🤖 Automatic trading: 🔴 OFF\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "✅ Connection test successful.\n\n"
            "Pocket Option is connected and ready "
            "for the next testing stage."
        )

    elif pocket_connecting:

        text = (
            "🟡 BOT CONNECTION TEST\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "📱 Telegram: ✅ CONNECTED\n"
            "🟡 Pocket Option: ⏳ CONNECTING\n"
            "💱 OTC market: ⏳ WAITING\n"
            "🤖 Automatic trading: 🔴 OFF\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "Please send /start again in a few seconds."
        )

    else:

        error_text = pocket_error or "Unknown connection error."

        text = (
            "🔴 BOT CONNECTION TEST\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "📱 Telegram: ✅ CONNECTED\n"
            "🔴 Pocket Option: ❌ NOT CONNECTED\n"
            "💱 OTC market: ❌ NOT CONNECTED\n"
            "🤖 Automatic trading: 🔴 OFF\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "❌ Pocket Option connection failed.\n\n"
            f"Reason:\n{error_text}"
        )

    bot.reply_to(
        message,
        text,
    )


# ============================================================
# SIMPLE STATUS COMMAND
# ============================================================

@bot.message_handler(commands=["status"])
def status_command(message):

    if pocket_connected:

        text = (
            "🟢 STATUS\n\n"
            "Telegram: ✅ Connected\n"
            "Pocket Option: ✅ Connected\n"
            "OTC connection: ✅ Active\n"
            "Automatic trading: 🔴 OFF"
        )

    elif pocket_connecting:

        text = (
            "🟡 STATUS\n\n"
            "Telegram: ✅ Connected\n"
            "Pocket Option: ⏳ Connecting...\n"
            "Automatic trading: 🔴 OFF"
        )

    else:

        text = (
            "🔴 STATUS\n\n"
            "Telegram: ✅ Connected\n"
            "Pocket Option: ❌ Not connected\n"
            "Automatic trading: 🔴 OFF\n\n"
            f"Error: {pocket_error or 'Unknown'}"
        )

    bot.reply_to(
        message,
        text,
    )


# ============================================================
# START EVERYTHING
# ============================================================

if __name__ == "__main__":

    print("========================================")
    print("POCKET OPTION TELEGRAM BOT")
    print("========================================")
    print("BOT_TOKEN: loaded from Render")
    print("PO_SESSION: loaded from Render")
    print("PO_UID: loaded from Render")
    print("Credentials are NOT stored in app.py")
    print("Automatic trading: OFF")
    print("Screenshot analysis: OFF")
    print("Signal generation: OFF")
    print("========================================")

    # --------------------------------------------------------
    # START POCKET OPTION CONNECTION
    # --------------------------------------------------------

    threading.Thread(
        target=pocket_thread,
        daemon=True,
    ).start()

    # --------------------------------------------------------
    # START TELEGRAM POLLING
    # --------------------------------------------------------

    logger.info(
        "Starting Telegram polling..."
    )

    threading.Thread(
        target=lambda: bot.infinity_polling(
            timeout=30,
            long_polling_timeout=30,
            skip_pending=True,
        ),
        daemon=True,
    ).start()

    # --------------------------------------------------------
    # START FLASK
    # --------------------------------------------------------

    port = int(
        os.getenv(
            "PORT",
            "10000",
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )

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
# FLASK / RENDER
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "Telegram + Pocket Option connection test is running."


@app.route("/ping")
def ping():
    return "OK"


# ============================================================
# RENDER ENVIRONMENT VARIABLES
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
PO_SESSION = os.getenv("PO_SESSION")
PO_UID = os.getenv("PO_UID")


if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing.")

if not PO_SESSION:
    raise RuntimeError("PO_SESSION is missing.")

if not PO_UID:
    raise RuntimeError("PO_UID is missing.")


# ============================================================
# TELEGRAM
# ============================================================

bot = telebot.TeleBot(BOT_TOKEN)


# ============================================================
# POCKET OPTION STATUS
# ============================================================

po_status = "CONNECTING"
po_error = None

po_client = None


# ============================================================
# POCKET OPTION CONNECTION
# ============================================================

async def pocket_option_connection():

    global po_status
    global po_error
    global po_client

    try:

        logger.info("Starting Pocket Option connection...")

        po_status = "CONNECTING"

        client = PocketOptionClient(logger=True)

        po_client = client

        authorization = AuthorizationData.model_validate(
            {
                "session": PO_SESSION,
                "isDemo": 1,
                "uid": int(PO_UID),
                "platform": 2,
                "isFastHistory": True,
                "isOptimized": True,
            }
        )

        logger.info("Initializing OTC subscription...")

        default_init(
            client,
            authorization=authorization,
            sub_assets=[Asset.AUDCAD_otc],
            sub_period=60,
        )

        logger.info("Connecting to Pocket Option WebSocket...")

        await client.connect(Regions.DEMO)

        logger.info("Waiting for authorization...")

        await asyncio.wait_for(
            client.authorized_event.wait(),
            timeout=20,
        )

        po_status = "CONNECTED"
        po_error = None

        logger.info(
            "========================================"
        )
        logger.info(
            "POCKET OPTION CONNECTED"
        )
        logger.info(
            "========================================"
        )

        # Keep connection alive.
        while True:
            await asyncio.sleep(30)

    except asyncio.TimeoutError:

        po_status = "FAILED"

        po_error = (
            "Pocket Option authorization timed out."
        )

        logger.error(
            "Pocket Option authorization timed out."
        )

    except Exception as e:

        po_status = "FAILED"

        po_error = str(e)

        logger.exception(
            "Pocket Option connection failed."
        )


# ============================================================
# POCKET OPTION BACKGROUND THREAD
# ============================================================

def run_pocket_connection():

    loop = asyncio.new_event_loop()

    asyncio.set_event_loop(loop)

    try:

        loop.run_until_complete(
            pocket_option_connection()
        )

    except Exception as e:

        logger.exception(
            "Pocket Option thread stopped: %s",
            e,
        )

    finally:

        loop.close()


# ============================================================
# TELEGRAM /START
# ============================================================

@bot.message_handler(commands=["start"])
def start_command(message):

    logger.info(
        "Received /start from Telegram chat %s",
        message.chat.id,
    )

    # Telegram test first.
    bot.reply_to(
        message,
        "📱 Telegram: ✅ CONNECTED\n\n"
        "🔎 Checking Pocket Option connection..."
    )

    # Give Pocket Option a moment, but DON'T block Telegram.
    if po_status == "CONNECTED":

        bot.send_message(
            message.chat.id,
            "🟢 POCKET OPTION: CONNECTED\n\n"
            "📱 Telegram: ✅ Connected\n"
            "🟢 Pocket Option: ✅ Connected\n"
            "💱 OTC connection: ✅ Active\n"
            "🤖 Automatic trading: 🔴 OFF\n\n"
            "Connection test successful."
        )

    elif po_status == "CONNECTING":

        bot.send_message(
            message.chat.id,
            "🟡 POCKET OPTION: STILL CONNECTING\n\n"
            "Telegram is working correctly.\n"
            "Pocket Option is still attempting authorization.\n\n"
            "Send /status again in a few seconds."
        )

    else:

        bot.send_message(
            message.chat.id,
            "🔴 POCKET OPTION: NOT CONNECTED\n\n"
            "📱 Telegram: ✅ Connected\n"
            "🔴 Pocket Option: ❌ Failed\n"
            "🤖 Automatic trading: 🔴 OFF\n\n"
            f"Reason:\n{po_error or 'Unknown error'}"
        )


# ============================================================
# TELEGRAM /STATUS
# ============================================================

@bot.message_handler(commands=["status"])
def status_command(message):

    if po_status == "CONNECTED":

        text = (
            "🟢 CONNECTION STATUS\n\n"
            "📱 Telegram: ✅ CONNECTED\n"
            "🟢 Pocket Option: ✅ CONNECTED\n"
            "💱 OTC: ✅ ACTIVE\n"
            "🤖 Automatic trading: 🔴 OFF"
        )

    elif po_status == "CONNECTING":

        text = (
            "🟡 CONNECTION STATUS\n\n"
            "📱 Telegram: ✅ CONNECTED\n"
            "🟡 Pocket Option: ⏳ CONNECTING\n"
            "💱 OTC: ⏳ WAITING\n"
            "🤖 Automatic trading: 🔴 OFF"
        )

    else:

        text = (
            "🔴 CONNECTION STATUS\n\n"
            "📱 Telegram: ✅ CONNECTED\n"
            "🔴 Pocket Option: ❌ FAILED\n"
            "🤖 Automatic trading: 🔴 OFF\n\n"
            f"Error:\n{po_error or 'Unknown error'}"
        )

    bot.reply_to(
        message,
        text,
    )


# ============================================================
# TELEGRAM ERROR HANDLER
# ============================================================

@bot.message_handler(
    func=lambda message: True
)
def unknown_message(message):

    bot.reply_to(
        message,
        "🟢 Telegram is connected.\n\n"
        "Send /start to run the connection test."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("========================================")
    print("POCKET OPTION TELEGRAM CONNECTION TEST")
    print("========================================")
    print("BOT_TOKEN: loaded from Render")
    print("PO_SESSION: loaded from Render")
    print("PO_UID: loaded from Render")
    print("Automatic trading: OFF")
    print("Screenshot analysis: OFF")
    print("Signals: OFF")
    print("========================================")

    # Start Pocket Option separately.
    threading.Thread(
        target=run_pocket_connection,
        daemon=True,
    ).start()

    # Start Telegram separately.
    threading.Thread(
        target=lambda: bot.infinity_polling(
            timeout=30,
            long_polling_timeout=30,
            skip_pending=True,
        ),
        daemon=True,
    ).start()

    # Start Render web server.
    port = int(
        os.getenv("PORT", "10000")
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )

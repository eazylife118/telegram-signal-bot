import os
import asyncio
import logging
from flask import Flask
import telebot

from pocket_option import PocketOptionClient
from pocket_option.constants import Regions
from pocket_option.contrib.default_init import default_init
from pocket_option.models import Asset, AuthorizationData

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
# SETTINGS
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

client = PocketOptionClient(logger=True)

ASSET = Asset.EURUSD_otc
CANDLE_PERIOD = 60
IS_DEMO = 1

# ============================================================
# POCKET OPTION CONNECTION
# ============================================================

default_init(
    client,
    authorization=AuthorizationData.model_validate(
        {
            "session": PO_SESSION,
            "isDemo": IS_DEMO,
            "uid": int(PO_UID),
            "platform": 2,
            "isFastHistory": True,
            "isOptimized": True,
        }
    ),
    sub_assets=[ASSET],
    sub_period=CANDLE_PERIOD,
)

# ============================================================
# TELEGRAM TEST
# ============================================================

@bot.message_handler(commands=["start"])
def start_command(message):
    bot.reply_to(
        message,
        "✅ Bot is online.\n\n"
        "Pocket Option connection is being initialized.\n"
        "No automatic trades are enabled."
    )

# ============================================================
# MAIN ASYNC LOOP
# ============================================================

async def pocket_option_loop():

    try:
        logger.info("Connecting to Pocket Option...")

        await client.connect(Regions.DEMO)

        await client.authorized_event.wait()

        logger.info("✅ Pocket Option authorized.")

        # Send confirmation through Telegram
        # Replace this with your own Telegram chat ID later
        logger.info("Pocket Option connection is ready.")

        while True:
            try:
                # Connection heartbeat
                await asyncio.sleep(10)

            except Exception:
                logger.exception("Pocket Option loop error")
                await asyncio.sleep(10)

    except Exception:
        logger.exception("❌ Pocket Option connection failed")

    finally:
        try:
            await client.disconnect()
        except Exception:
            pass

# ============================================================
# RUN POCKET OPTION LOOP IN BACKGROUND
# ============================================================

def start_async_loop():
    asyncio.run(pocket_option_loop())

# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    import threading

    threading.Thread(
        target=start_async_loop,
        daemon=True
    ).start()

    print("========================================")
    print("POCKET OPTION TELEGRAM BOT")
    print("========================================")
    print("BOT_TOKEN: loaded from Render")
    print("PO_SESSION: loaded from Render")
    print("PO_UID: loaded from Render")
    print("Credentials are NOT stored in app.py")
    print("Automatic trading: OFF")
    print("========================================")

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )

import os
import re
import time
import cv2
import pytesseract
import numpy as np
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ============================================================
# TELEGRAM SETTINGS
# ============================================================
TOKEN = "8937673241:AAGvyTA-G12xfwMlhif3Nh4_2Ag8OStq3tU"

# ============================================================
# FLASK SERVER (LIGHTWEIGHT)
# ============================================================
app = Flask(__name__)

@app.route("/")
def home():
    return "Screenshot Pair Detection Bot is running!"

@app.route("/ping")
def ping():
    return "pong", 200

def run_flask():
    app.run(host="0.0.0.0", port=10000, debug=False, threaded=True)

# ============================================================
# FAST CURRENCY PAIR DETECTION
# ============================================================
CURRENCY_CODES = [
    "USD", "EUR", "GBP", "JPY", "AUD",
    "CAD", "CHF", "NZD", "SGD", "HKD",
    "CNY", "TRY", "ZAR", "MXN", "BRL"
]

# Pre-compile regex pattern for speed
PAIR_PATTERN = re.compile(
    r"\b(" + "|".join(CURRENCY_CODES) + r")\s*[/\\\-_:]?\s*(" + "|".join(CURRENCY_CODES) + r")\b"
)

def detect_pair_fast(image_path):
    """ULTRA FAST pair detection - optimized for speed"""
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Quick resize (faster than full processing)
    height, width = img.shape[:2]
    if width > 1000:
        scale = 1000 / width
        new_width = 1000
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Convert to grayscale (fast)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Single threshold pass (fastest)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR - single pass
    try:
        text = pytesseract.image_to_string(thresh, config="--psm 6 --oem 3")
    except:
        return None
    
    # Quick cleanup
    text = text.upper()
    
    # Quick replacements
    replacements = {"USO": "USD", "EURO": "EUR", "EUP": "EUR", "6BP": "GBP", "G8P": "GBP", "JPV": "JPY"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Find pair
    matches = PAIR_PATTERN.findall(text)
    for base, quote in matches:
        if base != quote:
            return f"{base}/{quote}"
    
    return None

# ============================================================
# TELEGRAM HANDLERS
# ============================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📸 **Screenshot Pair Detection**\n\n"
        "Send a Pocket Option screenshot.\n"
        "I'll detect the currency pair **instantly**.\n\n"
        "⚡ **Extremely fast** processing!"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start_time = time.time()
    
    try:
        # Download photo (fast)
        photo = await update.message.photo[-1].get_file()
        file_path = "screenshot.png"
        await photo.download_to_drive(file_path)
        
        # Detect pair (fast)
        pair = detect_pair_fast(file_path)
        
        elapsed = time.time() - start_time
        
        if pair:
            response = (
                f"✅ **PAIR DETECTED!**\n\n"
                f"💱 **Pair:** `{pair}`\n"
                f"⚡ **Time:** `{elapsed:.2f}s`\n\n"
                "📸 Screenshot processed successfully.\n"
                "🔎 Currency pair detected by OCR."
            )
        else:
            response = (
                f"❌ **PAIR NOT DETECTED**\n\n"
                f"⚡ **Time:** `{elapsed:.2f}s`\n\n"
                "📸 Screenshot received.\n"
                "🔎 No currency pair found.\n\n"
                "💡 Try a clearer screenshot with the pair visible."
            )
        
        await update.message.reply_text(response)
        
        # Clean up
        try:
            os.remove(file_path)
        except:
            pass
            
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# ============================================================
# START BOT
# ============================================================
def run_bot():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.run_polling(drop_pending_updates=True)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import threading
    
    print("========================================")
    print("⚡ SCREENSHOT PAIR DETECTION (ULTRA FAST)")
    print("========================================")
    
    threading.Thread(target=run_flask, daemon=True).start()
    print("✅ Flask server started.")
    print("✅ Starting Telegram bot...")
    run_bot()

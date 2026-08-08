import cv2
import numpy as np
import telebot

# 1. Telegram Configuration
TELEGRAM_TOKEN = "8937673241:AAGvyTA-G12xfwMlhif3Nh4_2Ag8OStq3tU"
CHAT_ID = "6280535707"

bot = telebot.TeleBot(TELEGRAM_TOKEN)

def count_candles(image_path):
    # Load image and convert to HSV for color detection
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Define color ranges (Adjust these thresholds based on your specific chart background)
    # Green ranges 
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Red ranges 
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # 3. Create masks
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    # 4. Count function
    def get_candle_count(mask):
        # Find contours of the detected color blobs
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_candles = 0
        for cnt in contours:
            # Filter out tiny specks to only count proper candle bodies
            if cv2.contourArea(cnt) > 100: 
                valid_candles += 1
        return valid_candles

    green_count = get_candle_count(mask_green)
    red_count = get_candle_count(mask_red)
    
    return green_count, red_count

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        # Download the screenshot sent to Telegram
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        image_path = 'chart_screenshot.png'
        with open(image_path, 'wb') as new_file:
            new_file.write(downloaded_file)
            
        # Count candles
        green, red = count_candles(image_path)
        
        # Send result
        reply_text = f"📊 *Candlestick Analysis*\n🟢 Green Candles: {green}\n🔴 Red Candles: {red}\nTotal: {green + red}"
        bot.reply_to(message, reply_text, parse_mode='Markdown')
        
    except Exception as e:
        bot.reply_to(message, f"❌ An error occurred: {str(e)}")

# Start Bot
print("Bot is listening...")
bot.polling()

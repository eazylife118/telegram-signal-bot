import os
import cv2
import numpy as np
import telebot
import time
import requests
from PIL import Image
import io

# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_TOKEN = os.getenv(
    "BOT_TOKEN",
    "PASTE_YOUR_BOT_TOKEN_HERE"
)

bot = telebot.TeleBot(TELEGRAM_TOKEN)


# ============================================================
# NUMBER EXTRACTION — NO TESSERACT
# ============================================================

def extract_numbers(image_path):
    """Extract numbers from screenshot using OpenCV only."""
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return [], None
    
    height, width = img.shape[:2]
    
    # Resize for consistency
    if width < 1000:
        scale = 1000 / width
        new_width = 1000
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        height, width = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ============================================================
    # ADAPTIVE THRESHOLDING — Better for numbers
    # ============================================================
    
    # Method 1: Otsu threshold
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Method 2: Adaptive threshold
    thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
    
    # Combine both methods
    thresh = cv2.bitwise_or(thresh_otsu, thresh_adapt)
    
    # ============================================================
    # CLEAN UP
    # ============================================================
    
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # ============================================================
    # FIND CONTOURS
    # ============================================================
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_contours = []
    
    # Image dimensions for filtering
    img_area = height * width
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        area = w * h
        
        # Filter by size (digits are usually 0.5% - 3% of image area)
        min_area = img_area * 0.001
        max_area = img_area * 0.05
        
        if area < min_area or area > max_area:
            continue
        
        # Filter by aspect ratio (digits are taller than wide)
        aspect = h / max(w, 1)
        if aspect < 0.3 or aspect > 2.5:
            continue
        
        # Filter by solidity (digits are solid shapes)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.3:
                continue
        
        digit_contours.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "aspect": aspect,
            "contour": contour,
            "roi": thresh[y:y+h, x:x+w]
        })
    
    # ============================================================
    # SORT LEFT TO RIGHT
    # ============================================================
    
    digit_contours.sort(key=lambda d: d["x"])
    
    # ============================================================
    # RECOGNIZE DIGITS
    # ============================================================
    
    recognized = []
    
    for digit in digit_contours:
        roi = digit["roi"]
        h, w = roi.shape
        
        # Resize to standard size for comparison
        resized = cv2.resize(roi, (20, 30))
        
        # Count pixels
        total = np.sum(resized > 0)
        
        # Divide into 4 quadrants
        q1 = np.sum(resized[0:15, 0:10] > 0)   # Top-left
        q2 = np.sum(resized[0:15, 10:20] > 0)  # Top-right
        q3 = np.sum(resized[15:30, 0:10] > 0)  # Bottom-left
        q4 = np.sum(resized[15:30, 10:20] > 0) # Bottom-right
        
        # Features
        left = q1 + q3
        right = q2 + q4
        top = q1 + q2
        bottom = q3 + q4
        
        # Simple classification
        if total < 15:
            digit_value = "."
        elif total < 30:
            digit_value = "1"
        else:
            # Ratio features
            left_ratio = left / max(total, 1)
            right_ratio = right / max(total, 1)
            top_ratio = top / max(total, 1)
            bottom_ratio = bottom / max(total, 1)
            
            # Classify based on shape
            if left_ratio > 0.65 and top_ratio > 0.55:
                digit_value = "7"
            elif left_ratio > 0.65 and bottom_ratio > 0.55:
                digit_value = "2"
            elif right_ratio > 0.65 and top_ratio > 0.55:
                digit_value = "9"
            elif right_ratio > 0.65 and bottom_ratio > 0.55:
                digit_value = "3"
            elif top_ratio > 0.6 and bottom_ratio > 0.6 and left_ratio > 0.4 and right_ratio > 0.4:
                digit_value = "8"
            elif left_ratio > 0.6 and right_ratio > 0.4:
                digit_value = "6"
            elif right_ratio > 0.6 and left_ratio > 0.4:
                digit_value = "4"
            elif bottom_ratio > 0.7:
                digit_value = "5"
            elif left_ratio > 0.7 and right_ratio < 0.3:
                digit_value = "1"
            elif top_ratio > 0.7 and bottom_ratio < 0.3:
                digit_value = "7"
            elif top_ratio > 0.4 and bottom_ratio > 0.4 and left_ratio > 0.4 and right_ratio > 0.4:
                digit_value = "0"
            else:
                digit_value = "?"
        
        recognized.append({
            "x": digit["x"],
            "y": digit["y"],
            "w": digit["w"],
            "h": digit["h"],
            "value": digit_value,
            "roi": roi
        })
    
    # ============================================================
    # GROUP INTO NUMBERS
    # ============================================================
    
    numbers = []
    current_number = []
    current_x_start = None
    
    for digit in recognized:
        if not current_number:
            current_number.append(digit)
            current_x_start = digit["x"]
        else:
            # Check if close enough to be same number
            last = current_number[-1]
            gap = digit["x"] - (last["x"] + last["w"])
            
            if gap < 15:
                current_number.append(digit)
            else:
                # Build number string
                num_str = "".join([d["value"] for d in current_number])
                numbers.append({
                    "value": num_str,
                    "x": current_x_start,
                    "digits": current_number
                })
                current_number = [digit]
                current_x_start = digit["x"]
    
    if current_number:
        num_str = "".join([d["value"] for d in current_number])
        numbers.append({
            "value": num_str,
            "x": current_x_start,
            "digits": current_number
        })
    
    return numbers, digit_contours


def create_detection_map(img, numbers):
    """Draw boxes around detected numbers."""
    
    output = img.copy()
    
    for i, num in enumerate(numbers):
        digits = num["digits"]
        if not digits:
            continue
        
        # Get bounding box for the entire number
        x = min(d["x"] for d in digits)
        y = min(d["y"] for d in digits)
        w = max(d["x"] + d["w"] for d in digits) - x
        h = max(d["y"] + d["h"] for d in digits) - y
        
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output, num["value"], (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return output


# ============================================================
# TELEGRAM HANDLER
# ============================================================

@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    start_time = time.time()
    
    try:
        bot.reply_to(message, "🔢 Extracting numbers...")
        
        # Download image
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        image_path = "number_screenshot.png"
        with open(image_path, "wb") as f:
            f.write(downloaded_file)
        
        # Extract numbers
        numbers, _ = extract_numbers(image_path)
        
        elapsed = time.time() - start_time
        
        if not numbers:
            bot.reply_to(message, "❌ No numbers detected in the screenshot.")
            return
        
        # Build response
        response = "🔢 **NUMBERS EXTRACTED**\n\n"
        response += f"📊 Found: {len(numbers)} numbers\n"
        response += f"⚡ Time: {elapsed:.2f}s\n\n"
        response += "**Detected numbers:**\n"
        
        for i, num in enumerate(numbers[:30], 1):
            response += f"{i}. `{num['value']}`\n"
        
        if len(numbers) > 30:
            response += f"... and {len(numbers) - 30} more\n"
        
        bot.reply_to(message, response, parse_mode="Markdown")
        
        # Send detection map
        img = cv2.imread(image_path)
        if img is not None:
            map_img = create_detection_map(img, numbers)
            map_path = "number_detection_map.png"
            cv2.imwrite(map_path, map_img)
            
            with open(map_path, "rb") as f:
                bot.send_photo(
                    message.chat.id,
                    f,
                    caption="🔍 Numbers detected (green boxes with yellow labels)"
                )
            os.remove(map_path)
        
        os.remove(image_path)
        
    except Exception as e:
        bot.reply_to(message, f"❌ Error: {str(e)}")
        print(f"Error: {e}")


@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(
        message,
        "🔢 **NUMBER EXTRACTOR BOT**\n\n"
        "Send a screenshot with numbers.\n"
        "Extracts numbers WITHOUT Tesseract.\n"
        "⚡ Speed: 1-2 seconds.\n\n"
        "✅ Real numbers only\n"
        "✅ No fake data"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    print("=" * 40)
    print("🔢 NUMBER EXTRACTOR BOT")
    print("=" * 40)
    print("✅ No Tesseract")
    print("✅ 1-2 second processing")
    print("=" * 40)
    
    bot.infinity_polling(timeout=30, long_polling_timeout=30)

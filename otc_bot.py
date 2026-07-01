import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ==========================================
# TELEGRAM CREDENTIALS
# ==========================================
TELEGRAM_TOKEN = "8608138546:AAEetCz5xKlQlIRc0eZ3gVzvs046dPb86UI"
TELEGRAM_CHAT_ID = "6280535707"

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
        print(f"✅ Sent: {message}")
    except Exception as e:
        print("Telegram error:", e)

# ==========================================
# STARTUP MESSAGE
# ==========================================
send_telegram("✅ Bot is LIVE — scanning for real OTC price movements...")

# ==========================================
# BROWSER SETUP
# ==========================================
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

driver.get("https://pocketoption.com/en/trading")
time.sleep(10)

send_telegram("✅ Connected to Pocket Option — watching for live price changes...")

# ==========================================
# REAL SIGNAL LOOP (PRICE CHANGE TRIGGER)
# ==========================================
last_prices = {}

while True:
    try:
        assets = driver.find_elements(By.CLASS_NAME, "asset-item")
        for asset in assets:
            name = asset.get_attribute("data-name")
            if name and "OTC" in name:
                try:
                    price_el = asset.find_element(By.CLASS_NAME, "price")
                    price = float(price_el.text.replace(",", ""))
                    
                    if name not in last_prices or last_prices[name] != price:
                        last_prices[name] = price
                        send_telegram(f"📈 LIVE OTC SIGNAL: {name} - Price: {price}")
                except:
                    pass
        time.sleep(3)
    except Exception as e:
        print("Error:", e)
        time.sleep(5)

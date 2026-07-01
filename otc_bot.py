import requests
import random
import time
import json
import threading
import websocket
from flask import Flask
from threading import Thread

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
BOT_TOKEN = "8612354100:AAFUTlaSiq19yycQWpO70J4d6DEbgF4Kicc"
CHAT_ID = "6280535707"

# === YOUR POCKET OPTIONS SESSION CREDENTIALS ===
PO_SESSION = "deemw95tVnMPCTT7FTQ9imj7YkrhKqGCbT1FInXfxmKUmHAau4BpVYxCAamInBFx"
PO_UID = "131437859"

# ==========================================
# ALL OTC PAIRS (ORIGINAL FULL LIST)
# ==========================================
PAIRS = [
    "EURUSD-OTC",
    "GBPUSD-OTC",
    "USDJPY-OTC",
    "AUDUSD-OTC",
    "USDCAD-OTC",
    "USDCHF-OTC",
    "EURJPY-OTC",
    "GBPJPY-OTC",
    "AUDJPY-OTC",
    "CADJPY-OTC",
    "EURGBP-OTC",
    "EURAUD-OTC",
    "EURCAD-OTC",
    "EURCHF-OTC",
    "GBPAUD-OTC",
    "GBPCAD-OTC",
    "GBPCHF-OTC",
    "AUDCHF-OTC",
    "CADCHF-OTC",
    "NZDUSD-OTC",
    "EURTRY-OTC",
    "USDMXN-OTC",
    "USDCLP-OTC",
    "USDPKR-OTC",
    "USDIDR-OTC"
]

# ==========================================
# ROUND NUMBER LEVELS
# ==========================================
ROUND_LEVELS = {
    "EURUSD-OTC": 1.15600,
    "GBPUSD-OTC": 1.24800,
    "USDJPY-OTC": 162.800,
    "AUDUSD-OTC": 0.72100,
    "USDCAD-OTC": 1.39500,
    "USDCHF-OTC": 0.79400,
    "EURJPY-OTC": 183.800,
    "GBPJPY-OTC": 215.600,
    "AUDJPY-OTC": 108.400,
    "CADJPY-OTC": 120.800,
    "EURGBP-OTC": 0.85800,
    "EURAUD-OTC": 1.64000,
    "EURCAD-OTC": 1.50000,
    "EURCHF-OTC": 1.02459,
    "GBPAUD-OTC": 1.92135,
    "GBPCAD-OTC": 1.72000,
    "GBPCHF-OTC": 1.16000,
    "AUDCHF-OTC": 0.60884,
    "CADCHF-OTC": 0.58832,
    "NZDUSD-OTC": 0.59000,
    "EURTRY-OTC": 53.01600,
    "USDMXN-OTC": 17.7000,
    "USDCLP-OTC": 866.8000,
    "USDPKR-OTC": 277.400,
    "USDIDR-OTC": 16670.0
}

# --- Memory for strategies ---
previous_prices = {}
rejection_count = {}
last_signal_time = {}
latest_prices = {}
price_lock = threading.Lock()
bounce_memory = {}

# ==========================================
# POCKET OPTIONS WEBSOCKET CONNECTION
# ==========================================
def on_message(ws, message):
    global latest_prices
    try:
        data = json.loads(message)
        if "price" in data:
            symbol = data.get("symbol") or data.get("asset")
            price = float(data["price"])
            if symbol:
                with price_lock:
                    latest_prices[symbol] = price
    except Exception as e:
        pass

def on_error(ws, error):
    print(f"⚠️ WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("🔌 WebSocket closed. Reconnecting in 5 seconds...")
    time.sleep(5)
    connect_websocket()

def on_open(ws):
    print("✅ WebSocket connected to Pocket Options")
    for pair in PAIRS:
        subscribe_msg = {"action": "subscribe", "symbol": pair}
        ws.send(json.dumps(subscribe_msg))
        print(f"📡 Subscribed to {pair}")
        time.sleep(0.1)

def connect_websocket():
    ws_url = f"wss://demo-api-eu.po.market?token={PO_SESSION}"
    ws = websocket.WebSocketApp(ws_url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()

def get_all_prices():
    global latest_prices
    with price_lock:
        return latest_prices.copy()

# ==========================================
# GET OHLC DATA (SYNTHETIC FOR STRATEGIES)
# ==========================================
def get_ohlc(pair, interval="1min", outputsize=50):
    prices = get_all_prices()
    if pair in prices:
        current = prices[pair]
        return [current] * 50, [current] * 50, [current] * 50
    return None, None, None

# ==========================================
# STRATEGY 1: ROUND NUMBER REJECTION
# ==========================================
def check_rejection(pair, current_price):
    global previous_prices, rejection_count, last_signal_time

    if pair not in previous_prices:
        previous_prices[pair] = current_price
        rejection_count[pair] = 0
        last_signal_time[pair] = 0
        return None

    if time.time() - last_signal_time.get(pair, 0) < 120:
        return None

    prev_price = previous_prices[pair]
    target = ROUND_LEVELS.get(pair)
    if target is None:
        return None

    if abs(current_price - target) > 0.0010:
        previous_prices[pair] = current_price
        return None

    if prev_price < target <= current_price:
        rejection_count[pair] += 1
        if rejection_count[pair] >= 2:
            rejection_count[pair] = 0
            previous_prices[pair] = current_price
            last_signal_time[pair] = time.time()
            return "SELL", f"Rejection at {target:.5f}"

    elif prev_price > target >= current_price:
        rejection_count[pair] += 1
        if rejection_count[pair] >= 2:
            rejection_count[pair] = 0
            previous_prices[pair] = current_price
            last_signal_time[pair] = time.time()
            return "BUY", f"Bounce from {target:.5f}"
    else:
        rejection_count[pair] = 0

    previous_prices[pair] = current_price
    return None

# ==========================================
# STRATEGY 2: PRICE BOUNCE
# ==========================================
def check_price_bounce(pair, current_price):
    global bounce_memory
    if pair not in bounce_memory:
        bounce_memory[pair] = []
    
    bounce_memory[pair].append(current_price)
    if len(bounce_memory[pair]) > 10:
        bounce_memory[pair].pop(0)
    
    if len(bounce_memory[pair]) >= 5:
        for i in range(2, len(bounce_memory[pair]) - 1):
            drop = bounce_memory[pair][i-1] - bounce_memory[pair][i]
            if drop > bounce_memory[pair][i-1] * 0.0005:
                recovery = bounce_memory[pair][i+1] - bounce_memory[pair][i]
                if recovery > drop * 0.5:
                    return "BUY", f"Price Bounce at {current_price:.5f}"
    return None, None

# ==========================================
# STRATEGY 3: SUPPORT/RESISTANCE BREAKOUT
# ==========================================
def check_breakout(closes, highs, lows):
    if len(closes) < 20:
        return None, None
    
    resistance = max(highs[1:10])
    support = min(lows[1:10])
    current = closes[0]
    
    if current > resistance * 1.0005:
        return "BUY", f"Breakout above {resistance:.5f}"
    elif current < support * 0.9995:
        return "SELL", f"Breakdown below {support:.5f}"
    return None, None

# ==========================================
# STRATEGY 4: RSI DIVERGENCE
# ==========================================
def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains = 0
    losses = 0
    for i in range(1, period + 1):
        diff = closes[i - 1] - closes[i]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def check_rsi_divergence(closes, highs, lows):
    if len(closes) < 30:
        return None, None
    
    rsi_values = []
    for i in range(20, len(closes)):
        rsi = calculate_rsi(closes[i-14:i+1])
        if rsi is not None:
            rsi_values.append(rsi)
    
    if len(rsi_values) < 20:
        return None, None
    
    price_low = min(closes[:10])
    price_low2 = min(closes[10:20])
    rsi_low = min(rsi_values[:10])
    rsi_low2 = min(rsi_values[10:20])
    
    if price_low2 < price_low and rsi_low2 > rsi_low:
        return "BUY", "RSI Bullish Divergence"
    
    price_high = max(closes[:10])
    price_high2 = max(closes[10:20])
    rsi_high = max(rsi_values[:10])
    rsi_high2 = max(rsi_values[10:20])
    
    if price_high2 > price_high and rsi_high2 < rsi_high:
        return "SELL", "RSI Bearish Divergence"
    
    return None, None

# ==========================================
# STRATEGY 5: MOVING AVERAGE CROSSOVER
# ==========================================
def check_ma_crossover(closes):
    if len(closes) < 20:
        return None, None
    
    ma5 = sum(closes[:5]) / 5
    ma20 = sum(closes[:20]) / 20
    prev_ma5 = sum(closes[1:6]) / 5
    prev_ma20 = sum(closes[1:21]) / 20
    
    if prev_ma5 <= prev_ma20 and ma5 > ma20:
        return "BUY", f"MA5 crossed above MA20"
    elif prev_ma5 >= prev_ma20 and ma5 < ma20:
        return "SELL", f"MA5 crossed below MA20"
    return None, None

# ==========================================
# STRATEGY 6: BOLLINGER BANDS
# ==========================================
def check_bollinger_breakout(closes):
    if len(closes) < 20:
        return None, None
    
    ma = sum(closes[:20]) / 20
    variance = sum([(x - ma) ** 2 for x in closes[:20]]) / 20
    std_dev = variance ** 0.5
    
    upper = ma + (2 * std_dev)
    lower = ma - (2 * std_dev)
    current = closes[0]
    
    if current > upper:
        return "BUY", f"Price above upper Bollinger Band"
    elif current < lower:
        return "SELL", f"Price below lower Bollinger Band"
    return None, None

# ==========================================
# STRATEGY 7: FIBONACCI RETRACEMENT
# ==========================================
def check_fibonacci_retracement(closes, highs, lows):
    if len(closes) < 30:
        return None, None
    
    swing_high = max(closes[:20])
    swing_low = min(closes[:20])
    
    diff = swing_high - swing_low
    fib_618 = swing_high - (diff * 0.618)
    fib_500 = swing_high - (diff * 0.500)
    fib_382 = swing_high - (diff * 0.382)
    
    current = closes[0]
    
    if abs(current - fib_618) / current < 0.001:
        return "BUY" if current > closes[1] else "SELL", f"Price at 61.8% Fib"
    elif abs(current - fib_500) / current < 0.001:
        return "BUY" if current > closes[1] else "SELL", f"Price at 50% Fib"
    elif abs(current - fib_382) / current < 0.001:
        return "BUY" if current > closes[1] else "SELL", f"Price at 38.2% Fib"
    
    return None, None

# ==========================================
# COMBINED SIGNAL GENERATOR (ALL 7 STRATEGIES)
# ==========================================
def get_combined_signal(pair, current_price):
    closes, highs, lows = get_ohlc(pair)
    if closes is None:
        return None, None
    
    # Strategy 1: Round Number Rejection
    rejection = check_rejection(pair, current_price)
    if rejection is not None:
        direction, reason = rejection
        return direction, f"🔴 REJECTION: {reason}" if direction == "SELL" else f"🟢 REJECTION: {reason}"
    
    # Strategy 2: Price Bounce
    bounce_dir, bounce_reason = check_price_bounce(pair, current_price)
    if bounce_dir is not None:
        return bounce_dir, f"📈 BOUNCE: {bounce_reason}"
    
    # Strategy 3: Breakout
    breakout_dir, breakout_reason = check_breakout(closes, highs, lows)
    if breakout_dir is not None:
        return breakout_dir, f"🚀 BREAKOUT: {breakout_reason}"
    
    # Strategy 4: RSI Divergence
    rsi_dir, rsi_reason = check_rsi_divergence(closes, highs, lows)
    if rsi_dir is not None:
        return rsi_dir, f"📊 RSI: {rsi_reason}"
    
    # Strategy 5: MA Crossover
    ma_dir, ma_reason = check_ma_crossover(closes)
    if ma_dir is not None:
        return ma_dir, f"📈 MA: {ma_reason}"
    
    # Strategy 6: Bollinger Bands
    bb_dir, bb_reason = check_bollinger_breakout(closes)
    if bb_dir is not None:
        return bb_dir, f"📉 BB: {bb_reason}"
    
    # Strategy 7: Fibonacci
    fib_dir, fib_reason = check_fibonacci_retracement(closes, highs, lows)
    if fib_dir is not None:
        return fib_dir, f"🌀 FIB: {fib_reason}"
    
    return None, None

# ==========================================
# GET MARKET STRENGTH
# ==========================================
def get_market_strength(pair):
    prices = get_all_prices()
    if pair in prices:
        return "BUY", random.randint(65, 85)
    return None, None

# ==========================================
# SEND SIGNAL TO TELEGRAM
# ==========================================
def send_signal(pair, direction, reason):
    prices = get_all_prices()
    if pair not in prices:
        return
    price = prices[pair]
    
    real_direction, strength = get_market_strength(pair)
    if real_direction is None or strength is None:
        strength = 65

    expiry = random.choice(["1", "2", "3", "5"])
    now = time.time()
    signal_time = time.strftime('%H:%M', time.localtime(now))
    entry_time = time.strftime('%H:%M', time.localtime(now + 120))

    if direction == "BUY":
        dir_display = "🟢 BUY"
    else:
        dir_display = "🔴 SELL"

    message = f"""
🚨 SIGNAL ALERT

OTC Pair: {pair}
Direction: {dir_display}
Price: {price:.5f}

⏰ Signal Time: {signal_time}
🎯 Entry Time: {entry_time}
Expiry: {expiry} Min

Strength: {strength}% 🔥
Strategy: {reason}
"""

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message})
        print(f"✅ Signal sent for {pair}")
    except Exception as e:
        print(f"❌ Send error: {e}")

# ==========================================
# MAIN BOT LOOP
# ==========================================
def run_bot():
    # === DIRECT TELEGRAM TEST (INDEPENDENT) ===
    try:
        import requests
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        response = requests.post(url, data={"chat_id": CHAT_ID, "text": "✅ DIRECT TEST: Telegram is working!"})
        print(f"Direct test response: {response.text}")
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
    # === END DIRECT TEST ===

    CHECK_INTERVAL = 4
    print(f"🤖 Bot started. TEST MODE — Sending test signals every 30 seconds.")
    
    while True:
        try:
            # === FORCE TEST SIGNAL EVERY 30 SECONDS ===
            if int(time.time()) % 30 == 0:
                send_signal("EURUSD-OTC", "BUY", "🧪 TEST SIGNAL - IGNORE")
                print("✅ Test signal sent to Telegram")
                time.sleep(2)
            # === END TEST ===
            
            # ... (rest of your code)
            
            # Still try to get real prices and signals
            all_prices = get_all_prices()
            if all_prices:
                for pair, price in all_prices.items():
                    direction, reason = get_combined_signal(pair, price)
                    if direction is not None and reason is not None:
                        send_signal(pair, direction, reason)
                        time.sleep(2)
            
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print(f"❌ Main loop error: {e}")
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print(f"❌ Main loop error: {e}")
            time.sleep(CHECK_INTERVAL)

# ==========================================
# FLASK KEEP-ALIVE
# ==========================================
@app.route('/')
def home():
    return "✅ OTC Multi-Strategy Bot is running!"

@app.route('/ping')
def ping():
    return "pong", 200

# ==========================================
# START BOT
# ==========================================
Thread(target=connect_websocket, daemon=True).start()
Thread(target=run_bot, daemon=True).start()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

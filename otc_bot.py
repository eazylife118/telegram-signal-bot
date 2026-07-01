import websocket
import json
import requests
import time
import threading

# ========== YOUR CREDENTIALS (PASTE HERE) ==========
TELEGRAM_TOKEN = "8608138546:AAEetCz5xKlQlIRc0eZ3gVzvs046dPb86UI"        
TELEGRAM_CHAT_ID = "6280535707"       

# Pocket Option credentials (already have these)
PO_SESSION = "deemw95tVnMPCTT7FTQ9imj7YkrhKqGCbT1FInXfxmKUmHAau4BpVYxCAamInBFx"
PO_UID = "131437859"

# ========== OTC PAIRS TO SCAN ==========
OTC_PAIRS = [
    "EURUSD-OTC",
    "GBPUSD-OTC",
    "USDJPY-OTC",
    "AUDUSD-OTC",
    "USDCAD-OTC",
    "NZDUSD-OTC",
    "EURJPY-OTC",
    "GBPJPY-OTC"
]

# ========== TELEGRAM SENDER ==========
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Telegram error:", e)

# ========== STRATEGY ENGINE (10 STRATEGIES) ==========
def check_strategies(pair, candle1, candle2, candle3, candle4, candle5, ema_20):
    signals = []

    # Strategy 1: Candle Reversal Pattern (2 min)
    if (candle3['close'] > candle3['open'] and
        candle2['close'] > candle2['open'] and
        (candle1['high'] - candle1['close']) > (candle1['close'] - candle1['low'])):
        signals.append(f"📈 BUY {pair} - Reversal Pattern (2 min)")

    # Strategy 2: 3-Candle Momentum (1 min)
    if (candle1['close'] > candle1['open'] and
        candle2['close'] > candle2['open'] and
        candle3['close'] > candle3['open']):
        signals.append(f"📈 BUY {pair} - 3-Candle Momentum (1 min)")

    # Strategy 3: 2-Minute Reset (2 min)
    if (candle1['close'] < candle1['open'] and
        candle2['close'] < candle2['open'] and
        candle3['close'] < candle3['open']):
        signals.append(f"📉 SELL {pair} - 2-Minute Reset (2 min)")

    # Strategy 4: Double Touch (3 min)
    if abs(candle1['low'] - candle3['low']) < 0.0002:
        signals.append(f"📈 BUY {pair} - Double Touch (3 min)")

    # Strategy 5: Spike Rejection (1–2 min)
    if (candle1['high'] - candle2['high']) > 0.001 and (candle1['close'] < candle1['open']):
        signals.append(f"📉 SELL {pair} - Spike Rejection (1-2 min)")

    # Strategy 6: Consolidation Break (2–3 min)
    high_range = max(candle1['high'], candle2['high'], candle3['high'])
    low_range = min(candle1['low'], candle2['low'], candle3['low'])
    if (high_range - low_range) < 0.0005:
        signals.append(f"⏳ BREAKOUT PENDING {pair} - Consolidation (2-3 min)")

    # Strategy 7: EMA Pullback (2 min)
    if candle1['low'] < ema_20 and candle1['close'] > ema_20:
        signals.append(f"📈 BUY {pair} - EMA Pullback (2 min)")

    # Strategy 8: Bull/Bear Confirmation (1 min)
    if candle1['close'] > candle1['open'] and candle2['close'] > candle2['open']:
        signals.append(f"📈 BUY {pair} - Bull Confirmation (1 min)")

    # Strategy 9: 60-Second Scalp (1 min)
    if (candle1['close'] - candle1['open']) > (candle2['close'] - candle2['open']) * 1.5:
        signals.append(f"📈 BUY {pair} - Scalp (1 min)")

    return signals

# ========== POCKET OPTION WEBSOCKET ==========
class PocketOptionWebSocket:
    def __init__(self):
        self.url = "wss://ws.pocketoption.com/websocket"
        self.ws = None
        self.candles = {}  # Store candles for each pair
        self.ema_20 = {}   # Store EMA for each pair
        self.session = PO_SESSION
        self.uid = PO_UID
        self.connected = False
        self.last_signal_time = 0
        self.signal_cooldown = 60  # Seconds between signals

    def generate_auth_payload(self):
        payload = {
            "event": "auth",
            "session": self.session,
            "uid": self.uid
        }
        return json.dumps(payload)

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            
            # Check if it's a candle update
            if 'candle' in data and 'pair' in data:
                pair = data['pair']
                candle = {
                    'open': data['candle']['open'],
                    'high': data['candle']['high'],
                    'low': data['candle']['low'],
                    'close': data['candle']['close']
                }
                
                # Initialize storage for this pair
                if pair not in self.candles:
                    self.candles[pair] = []
                    self.ema_20[pair] = 0
                
                # Store last 5 candles
                self.candles[pair].append(candle)
                if len(self.candles[pair]) > 5:
                    self.candles[pair].pop(0)
                
                # Calculate EMA 20 (simplified)
                if len(self.candles[pair]) >= 5:
                    self.ema_20[pair] = sum([c['close'] for c in self.candles[pair]]) / len(self.candles[pair])
                    
                    # Run strategies for this pair
                    if len(self.candles[pair]) >= 5:
                        signals = check_strategies(
                            pair,
                            self.candles[pair][-1],  # candle 1 (most recent)
                            self.candles[pair][-2],  # candle 2
                            self.candles[pair][-3],  # candle 3
                            self.candles[pair][-4],  # candle 4
                            self.candles[pair][-5],  # candle 5
                            self.ema_20[pair]
                        )
                        
                        # Send ONLY THE FIRST SIGNAL (one per cycle)
                        if signals:
                            signal = signals[0]  # Take only the first one
                            current_time = time.time()
                            
                            # Cooldown check
                            if current_time - self.last_signal_time >= self.signal_cooldown:
                                print(f"🔔 {signal}")
                                send_telegram(f"🔔 {signal}")
                                self.last_signal_time = current_time
            
            # Handle connection confirmation
            elif 'status' in data and data['status'] == 'connected':
                print("✅ Connected and authenticated")
                self.connected = True
                
        except Exception as e:
            print("Error processing message:", e)

    def on_error(self, ws, error):
        print("WebSocket error:", error)
        self.connected = False

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket closed - Reconnecting in 5 seconds...")
        self.connected = False
        time.sleep(5)
        self.connect()

    def on_open(self, ws):
        print("✅ WebSocket connected")
        # Send authentication
        auth_payload = self.generate_auth_payload()
        ws.send(auth_payload)
        print("🔐 Authentication sent...")
        
        # Subscribe to ALL OTC pairs
        for pair in OTC_PAIRS:
            subscribe_msg = {
                "event": "subscribe",
                "pair": pair
            }
            ws.send(json.dumps(subscribe_msg))
            print(f"📡 Subscribed to {pair}")
            time.sleep(0.1)  # Small delay to avoid flooding

    def connect(self):
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        print("🔄 Connecting to Pocket Option WebSocket...")
        self.ws.run_forever()

# ========== START BOT ==========
if __name__ == "__main__":
    print("=" * 50)
    print("🤖 OTC SIGNAL BOT STARTING...")
    print("=" * 50)
    print(f"📱 Telegram Bot: {TELEGRAM_TOKEN[:10]}...")
    print(f"👤 Pocket Option UID: {PO_UID}")
    print(f"📊 Scanning {len(OTC_PAIRS)} OTC pairs")
    print("⏱️  One signal every 60 seconds max")
    print("=" * 50)
    
    bot = PocketOptionWebSocket()
    bot.connect()

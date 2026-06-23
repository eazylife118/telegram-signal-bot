# Telegram Signal Bot

Simple Telegram bot that sends test trading signals every 1 minute.

## Features

- Sends Telegram alerts automatically
- Random BUY/SELL signals
- Runs continuously
- Easy to upgrade with real market analysis

## Setup

1. Create a Telegram bot using BotFather.
2. Get your Bot Token.
3. Get your Chat ID.
4. Open `bot.py`.
5. Replace:

```python
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"
```

with your real values.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python bot.py
```

## Example Alert

📊 SIGNAL ALERT

Pair: GBPUSD

Direction: BUY

Timeframe: 1m

⚠️ Test Signal

## Deployment

Can be deployed on:
- Render
- GitHub Actions
- Replit
- Railway

## Future Upgrades

- Real market scanner
- RSI signals
- Trend detection
- Multiple currency pairs
- Signal statistics

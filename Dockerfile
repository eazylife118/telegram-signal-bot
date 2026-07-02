# Use the official Python image (slim-buster is smaller)
FROM python:3.10-slim-buster

# Install Tesseract OCR (system-level) — minimal dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements_screenshot.txt .
RUN pip install --no-cache-dir -r requirements_screenshot.txt

# Copy the rest of the application
COPY . .

# Command to run the bot
CMD ["python", "bot.py"]

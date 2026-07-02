# Use the official Python image
FROM python:3.10-slim

# Install Tesseract OCR (system-level)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
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

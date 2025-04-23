import os
from flask import Flask
from threading import Thread
import asyncio
import logging
# Import the main function that now reads token from env
from binary_bot import main as bot_main

# Configure logging (StreamHandler only)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Ensure only StreamHandler
)
logger = logging.getLogger(__name__)


# Create Flask app
app = Flask(__name__)

# Define route for Render health checks
@app.route('/')
def home():
    # Basic check - ideally should check if bot thread is alive too
    return "Binary Trading Bot web service is running!"

# Function to run the bot's main async function
def run_bot():
    try:
        logger.info("Starting bot main function in background thread...")
        # Ensure any required setup happens before this if needed
        asyncio.run(bot_main())
        # asyncio.run() will block until main() completes/errors
        logger.info("Bot main function finished or stopped.")
    except Exception as e:
        # Log crashes within the bot's main function/loop
        logger.error(f"Exception in bot thread: {e}", exc_info=True) # Add exc_info for traceback

# Get port from environment variable provided by Render
port = int(os.environ.get("PORT", 8080)) # Default added just in case

if __name__ == "__main__":
    # Start the bot in a separate thread
    logger.info("Creating bot thread...")
    bot_thread = Thread(target=run_bot, name="TelegramBotThread")
    bot_thread.daemon = True # Allows main Flask thread to exit even if bot thread hangs (use carefully)
    bot_thread.start()
    logger.info("Bot thread started.")

    # Run the Flask app (Render expects this for a Web Service)
    # Use 'waitress' or 'gunicorn' in production instead of Flask's development server
    logger.info(f"Starting Flask web server on host 0.0.0.0 port {port}")
    # For simplicity, using Flask's built-in server:
    app.run(host='0.0.0.0', port=port)
    # In production for Render, your start command in render.yaml might be:
    # gunicorn app:app --bind 0.0.0.0:$PORT

# --- End of app.py ---

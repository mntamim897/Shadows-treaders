import os
import logging
import time
import sys
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("watchdog.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def check_bot_running():
    """Check if the bot process is running and restart if needed"""
    try:
        # Import the bot module to check for errors
        import binary_bot
        logger.info("Bot module imported successfully")
        return True
    except Exception as e:
        logger.error(f"Error importing bot module: {e}")
        return False

def restart_bot():
    """Restart the bot process"""
    try:
        logger.info("Attempting to restart the bot...")
        # Import and run the main function from binary_bot
        from binary_bot import main
        import asyncio
        asyncio.run(main())
        logger.info("Bot restarted successfully")
        return True
    except Exception as e:
        logger.error(f"Error restarting bot: {e}")
        return False

def main_watchdog():
    """Main watchdog function to keep the bot running"""
    logger.info("Starting watchdog for binary trading bot...")
    
    while True:
        if not check_bot_running():
            logger.warning("Bot not running, attempting to restart...")
            restart_success = restart_bot()
            if not restart_success:
                logger.error("Failed to restart bot, will try again in 60 seconds")
        
        # Sleep for 60 seconds before checking again
        logger.info("Watchdog sleeping for 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    try:
        main_watchdog()
    except KeyboardInterrupt:
        logger.info("Watchdog terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Watchdog encountered an error: {e}")
        sys.exit(1)

import os
import logging
import asyncio
from datetime import datetime, timedelta
import pytz
import sys # Added for potentially exiting if token is missing
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from binary_data import BinaryDataFetcher
from binary_analysis import BinaryAnalyzer

# Load environment variables (for local testing)
load_dotenv()

# --- Read Token from Environment ---
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
ADMIN_CHAT_ID_STR = os.environ.get('ADMIN_CHAT_ID') # Read as string

# Configure logging (FileHandler removed)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler("binary_bot.log"), # Removed
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Check for Token ---
if not TELEGRAM_BOT_TOKEN:
    logger.critical("FATAL ERROR: TELEGRAM_BOT_TOKEN environment variable not set.")
    # Exit if the token is absolutely required to even start the application
    sys.exit("FATAL ERROR: Bot token not found in environment variables.")

# --- Process Admin Chat ID ---
admin_chat_id_int = None
if ADMIN_CHAT_ID_STR:
    try:
        admin_chat_id_int = int(ADMIN_CHAT_ID_STR)
    except ValueError:
        logger.warning(f"ADMIN_CHAT_ID ('{ADMIN_CHAT_ID_STR}') in environment is not a valid integer.")
else:
     # Set a default if not provided in env, or handle as needed
     logger.info("ADMIN_CHAT_ID not found in environment variables.")
     # admin_chat_id_int = 5679669495 # Example: Set a default if required


SIGNAL_TIMEFRAMES = ['1', '5'] # Kept as is, seems unused by env var

class BinarySignalBot:
    """
    Telegram bot for binary trading signals
    Optimized for 1-5 minute timeframes with asset and platform selection
    """

    def __init__(self):
        """Initialize the bot with necessary components"""
        self.data_fetcher = BinaryDataFetcher()
        self.analyzer = BinaryAnalyzer()
        self.authorized_users = set()
        # Use the integer admin_chat_id_int processed earlier
        if admin_chat_id_int:
            self.authorized_users.add(admin_chat_id_int)

        # Get supported assets and platforms
        self.asset_types = ["forex", "crypto", "commodities", "indices"]
        self.assets = self.data_fetcher.get_supported_assets()
        self.platforms = self.data_fetcher.get_supported_platforms()

        # User preferences storage
        self.user_preferences = {}  # {user_id: {'asset': asset, 'platform': platform, 'timeframe': timeframe}}

        # Bot settings
        self.signal_interval = 1  # Minutes between signal checks
        self.auto_signal = False  # Auto-send signals flag
        self.signal_threshold = 70  # Minimum signal strength (0-100)

        logger.info("BinarySignalBot initialized")

    # --- Rest of your BinarySignalBot class methods ---
    # (start_command, help_command, signal_command, settings_command, etc.)
    # No changes needed inside these methods unless they directly used the old hardcoded token
    # Ensure ADMIN_CHAT_ID logic uses the 'admin_chat_id_int' variable if needed elsewhere.

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command"""
        user_id = update.effective_user.id
        username = update.effective_user.username or update.effective_user.first_name

        logger.info(f"Start command from user {user_id} ({username})")

        # Authorize the user automatically
        self.authorized_users.add(user_id)

        # Initialize user preferences if not exists
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                'asset': 'EUR/USD',  # Default asset
                'platform': 'Quotex',  # Default platform
                'timeframe': '1min'  # Default timeframe
            }

        welcome_message = (
            f"👋 Welcome to the Binary Trading Signal Bot, {username}!\n\n"
            f"This bot provides binary trading signals with accuracy percentages.\n\n"
            f"Let's get started by selecting which asset you want to trade:"
        )

        # Create keyboard with asset type options
        keyboard = [
            [InlineKeyboardButton("Forex", callback_data="asset_type_forex")],
            [InlineKeyboardButton("Crypto", callback_data="asset_type_crypto")],
            [InlineKeyboardButton("Commodities", callback_data="asset_type_commodities")],
            [InlineKeyboardButton("Indices", callback_data="asset_type_indices")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(welcome_message, reply_markup=reply_markup)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /help command"""
        help_message = (
            "📚 *Binary Trading Signal Bot Help*\n\n"
            "*Commands:*\n"
            "/start - Start the bot and select asset to trade\n"
            "/help - Show this help message\n"
            "/signal - Get current trading signals\n"
            "/settings - Configure bot settings\n"
            "/platform - Select trading platform\n\n"

            "*Signal Information:*\n"
            "• Signals are optimized for 1-5 minute timeframes\n"
            "• Each signal includes entry price and expiry time\n"
            "• Signal accuracy percentage is provided\n"
            "• Supported asset types: Forex, Crypto, Commodities, Indices\n"
            "• Supported platforms: Quotex, IQ Option, Pocket Option, Binomo, Olymp Trade\n\n"

            "*Trading Tips:*\n"
            "• Always use proper risk management\n"
            "• Never risk more than 1-2% of your account per trade\n"
            "• Verify signals with your own analysis\n"
            "• Be aware of market news and events\n\n"

            "⚠️ *RISK DISCLAIMER:* Trading binary options carries significant risk. "
            "These signals are based on technical analysis and are not guaranteed to be profitable."
        )

        await update.message.reply_text(help_message, parse_mode='Markdown')

    async def signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /signal command"""
        user_id = update.effective_user.id

        if user_id not in self.authorized_users:
            await update.message.reply_text(
                "⚠️ You are not authorized to use this bot. Please use /start command first."
            )
            return

        # Check if user has preferences
        if user_id not in self.user_preferences:
            await update.message.reply_text(
                "Please use /start command first to select your trading preferences."
            )
            return

        await update.message.reply_text("🔍 Analyzing the market... Please wait.")

        # Get user preferences
        preferences = self.user_preferences[user_id]

        # Create keyboard with timeframe options
        keyboard = [
            [
                InlineKeyboardButton("1 min", callback_data="timeframe_1min"),
                InlineKeyboardButton("5 min", callback_data="timeframe_5min")
            ],
            [InlineKeyboardButton("Change Asset", callback_data="change_asset")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            f"Current selection: *{preferences['asset']}* on *{preferences['platform']}*\n\n"
            "Select timeframe for signals:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /settings command"""
        user_id = update.effective_user.id

        if user_id not in self.authorized_users:
            await update.message.reply_text(
                "⚠️ You are not authorized to use this bot. Please use /start command first."
            )
            return

        # Create settings keyboard
        keyboard = [
            [InlineKeyboardButton("Change Asset", callback_data="change_asset")],
            [InlineKeyboardButton("Change Platform", callback_data="change_platform")],
            [InlineKeyboardButton("Signal Threshold", callback_data="signal_threshold")],
            [
                InlineKeyboardButton("Auto Signal ON" if self.auto_signal else "Auto Signal OFF",
                                    callback_data="toggle_auto_signal")
            ],
            [InlineKeyboardButton("Back to Main Menu", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Get user preferences
        preferences = self.user_preferences.get(user_id, {
            'asset': 'EUR/USD',
            'platform': 'Quotex',
            'timeframe': '1min'
        })

        settings_message = (
            "⚙️ *Bot Settings*\n\n"
            f"*Current Asset:* {preferences.get('asset', 'Not set')}\n"
            f"*Trading Platform:* {preferences.get('platform', 'Not set')}\n"
            f"*Timeframe:* {preferences.get('timeframe', '1min')}\n"
            f"*Signal Threshold:* {self.signal_threshold}%\n"
            f"*Auto Signal:* {'Enabled' if self.auto_signal else 'Disabled'}\n"
            f"*Signal Interval:* {self.signal_interval} minute(s)\n\n"
            "Select an option to configure:"
        )

        await update.message.reply_text(settings_message, reply_markup=reply_markup, parse_mode='Markdown')

    async def platform_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /platform command"""
        user_id = update.effective_user.id

        if user_id not in self.authorized_users:
            await update.message.reply_text(
                "⚠️ You are not authorized to use this bot. Please use /start command first."
            )
            return

        await self.show_platform_selection(update.message.chat_id)

    async def show_platform_selection(self, chat_id):
        """Show platform selection keyboard"""
        # Create keyboard with platform options
        keyboard = []
        for platform in self.platforms:
            keyboard.append([InlineKeyboardButton(platform, callback_data=f"platform_{platform}")])

        keyboard.append([InlineKeyboardButton("Back to Settings", callback_data="settings")])
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Ensure self.application is set before calling this
        if not hasattr(self, 'application') or not self.application:
             logger.error("Application context not available for sending message.")
             return # Or handle appropriately

        await self.application.bot.send_message(
            chat_id=chat_id,
            text="Select your trading platform:",
            reply_markup=reply_markup
        )

    async def show_asset_type_selection(self, chat_id):
        """Show asset type selection keyboard"""
        # Create keyboard with asset type options
        keyboard = [
            [InlineKeyboardButton("Forex", callback_data="asset_type_forex")],
            [InlineKeyboardButton("Crypto", callback_data="asset_type_crypto")],
            [InlineKeyboardButton("Commodities", callback_data="asset_type_commodities")],
            [InlineKeyboardButton("Indices", callback_data="asset_type_indices")]
        ]
        keyboard.append([InlineKeyboardButton("Back to Settings", callback_data="settings")])
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Ensure self.application is set before calling this
        if not hasattr(self, 'application') or not self.application:
             logger.error("Application context not available for sending message.")
             return # Or handle appropriately

        await self.application.bot.send_message(
            chat_id=chat_id,
            text="Select asset type:",
            reply_markup=reply_markup
        )

    async def show_asset_selection(self, chat_id, asset_type):
        """Show asset selection keyboard for a specific asset type"""
        # Get assets for the selected type
        assets = self.data_fetcher.get_supported_assets(asset_type)

        # Create keyboard with asset options
        keyboard = []
        for asset in assets:
            keyboard.append([InlineKeyboardButton(asset, callback_data=f"asset_{asset}")])

        keyboard.append([InlineKeyboardButton("Back to Asset Types", callback_data="change_asset")])
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Ensure self.application is set before calling this
        if not hasattr(self, 'application') or not self.application:
             logger.error("Application context not available for sending message.")
             return # Or handle appropriately

        await self.application.bot.send_message(
            chat_id=chat_id,
            text=f"Select {asset_type} asset to trade:",
            reply_markup=reply_markup
        )

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()

        user_id = query.from_user.id

        # Make sure application context is available if needed later
        if not hasattr(self, 'application') or not self.application:
             # Attempt to get it from context if possible (depends on PTB version/setup)
             if context._application:
                 self.application = context._application
             else:
                  logger.error("Application context not available in button_callback.")
                  # Decide how to handle this - maybe reply with an error?
                  # await query.edit_message_text("Error: Bot context not fully initialized.")
                  # return

        # Initialize user preferences if not exists
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                'asset': 'EUR/USD',
                'platform': 'Quotex',
                'timeframe': '1min'
            }

        # Asset type selection
        if query.data.startswith("asset_type_"):
            asset_type = query.data.split("_")[2]
            await self.show_asset_selection(query.message.chat_id, asset_type)

        # Asset selection
        elif query.data.startswith("asset_"):
            asset = query.data.split("_", 1)[1] # Use split with maxsplit=1
            self.user_preferences[user_id]['asset'] = asset

            # After selecting asset, prompt for platform
            await query.edit_message_text(
                f"Asset selected: *{asset}*\n\nNow select your trading platform:",
                parse_mode='Markdown'
            )
            await self.show_platform_selection(query.message.chat_id)

        # Platform selection
        elif query.data.startswith("platform_"):
            platform = query.data.split("_", 1)[1] # Use split with maxsplit=1
            self.user_preferences[user_id]['platform'] = platform

            # After selecting platform, show confirmation and options
            keyboard = [
                [InlineKeyboardButton("Get Signal Now", callback_data="get_signals")],
                [InlineKeyboardButton("Settings", callback_data="settings")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                f"✅ Setup complete!\n\n"
                f"Asset: *{self.user_preferences[user_id]['asset']}*\n"
                f"Platform: *{platform}*\n\n"
                f"You can now get trading signals or adjust settings.",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )

        # Timeframe selection
        elif query.data.startswith("timeframe_"):
            timeframe = query.data.split("_")[1]
            self.user_preferences[user_id]['timeframe'] = timeframe

            # Generate and show signal
            await query.edit_message_text(f"Generating signal for {timeframe} timeframe... Please wait.")

            # Get user preferences
            preferences = self.user_preferences[user_id]
            asset = preferences['asset']
            platform = preferences['platform']

            # Get expiry minutes based on timeframe
            expiry_minutes = 1 if timeframe == "1min" else 5

            # Generate signal
            signal = await self.generate_signal(asset, timeframe, platform, expiry_minutes)

            if not signal:
                await query.edit_message_text(
                    "No significant signals found at this time. Try again later or adjust signal threshold in settings."
                )
                return

            # Format and send signal
            message = self.format_signal_message(signal)

            # Add back button
            keyboard = [[InlineKeyboardButton("Get Another Signal", callback_data="get_signals")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')

        # Other callbacks
        elif query.data == "get_signals":
            # Show timeframe selection
            keyboard = [
                [
                    InlineKeyboardButton("1 min", callback_data="timeframe_1min"),
                    InlineKeyboardButton("5 min", callback_data="timeframe_5min")
                ],
                [InlineKeyboardButton("Change Asset", callback_data="change_asset")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            preferences = self.user_preferences[user_id]

            await query.edit_message_text(
                f"Current selection: *{preferences['asset']}* on *{preferences['platform']}*\n\n"
                "Select timeframe for signals:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )

        elif query.data == "change_asset":
            await self.show_asset_type_selection(query.message.chat_id)

        elif query.data == "change_platform":
            await self.show_platform_selection(query.message.chat_id)

        elif query.data == "settings":
             # Call the settings command handler logic directly
             # Need to simulate an update object if the handler expects it
             # Or refactor settings display logic into a reusable method
            await self.display_settings(query.message.chat_id, user_id)


        elif query.data == "signal_threshold":
            # Create keyboard with threshold options
            keyboard = [
                [
                    InlineKeyboardButton("50%", callback_data="threshold_50"),
                    InlineKeyboardButton("60%", callback_data="threshold_60"),
                    InlineKeyboardButton("70%", callback_data="threshold_70")
                ],
                [
                    InlineKeyboardButton("80%", callback_data="threshold_80"),
                    InlineKeyboardButton("90%", callback_data="threshold_90"),
                    InlineKeyboardButton("95%", callback_data="threshold_95")
                ],
                [InlineKeyboardButton("Back to Settings", callback_data="settings")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                f"Select signal strength threshold (currently {self.signal_threshold}%):\n\n"
                "Higher threshold = fewer but stronger signals\n"
                "Lower threshold = more signals but potentially weaker",
                reply_markup=reply_markup
            )

        elif query.data.startswith("threshold_"):
            threshold = int(query.data.split("_")[1])
            self.signal_threshold = threshold

            await query.edit_message_text(f"Signal threshold set to {threshold}%.")

            # Return to settings
            await asyncio.sleep(1)  # Short delay for better UX
            await self.display_settings(query.message.chat_id, user_id) # Show updated settings

        elif query.data == "toggle_auto_signal":
            self.auto_signal = not self.auto_signal

            status = "enabled" if self.auto_signal else "disabled"
            await query.edit_message_text(f"Auto signal has been {status}.")

            # Return to settings
            await asyncio.sleep(1)  # Short delay for better UX
            await self.display_settings(query.message.chat_id, user_id) # Show updated settings

        elif query.data == "main_menu":
            keyboard = [
                [InlineKeyboardButton("Get Signals", callback_data="get_signals")],
                [InlineKeyboardButton("Settings", callback_data="settings")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                "Welcome back! Please select an option:", # Adjusted text
                reply_markup=reply_markup
            )

    async def display_settings(self, chat_id, user_id):
        """Helper method to display settings"""
        # Ensure self.application is set
        if not hasattr(self, 'application') or not self.application:
             logger.error("Application context not available for displaying settings.")
             # Attempt to send an error message if possible
             # await self.application.bot.send_message(chat_id=chat_id, text="Error: Bot context unavailable.")
             return

        keyboard = [
            [InlineKeyboardButton("Change Asset", callback_data="change_asset")],
            [InlineKeyboardButton("Change Platform", callback_data="change_platform")],
            [InlineKeyboardButton("Signal Threshold", callback_data="signal_threshold")],
            [
                InlineKeyboardButton("Auto Signal ON" if self.auto_signal else "Auto Signal OFF",
                                    callback_data="toggle_auto_signal")
            ],
            [InlineKeyboardButton("Back to Main Menu", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        preferences = self.user_preferences.get(user_id, {
            'asset': 'EUR/USD',
            'platform': 'Quotex',
            'timeframe': '1min'
        })

        settings_message = (
            "⚙️ *Bot Settings*\n\n"
            f"*Current Asset:* {preferences.get('asset', 'Not set')}\n"
            f"*Trading Platform:* {preferences.get('platform', 'Not set')}\n"
            # f"*Timeframe:* {preferences.get('timeframe', '1min')}\n" # Timeframe not part of settings here
            f"*Signal Threshold:* {self.signal_threshold}%\n"
            f"*Auto Signal:* {'Enabled' if self.auto_signal else 'Disabled'}\n"
            f"*Signal Interval:* {self.signal_interval} minute(s)\n\n"
            "Select an option to configure:"
        )
        # Use send_message or edit_message depending on context
        # Assuming called from button_callback, edit is appropriate if message exists
        try:
             await self.application.bot.edit_message_text(
                  chat_id=chat_id,
                  # We need message_id here - button_callback needs to pass it
                  # This refactoring needs careful handling of message context
                  # For now, let's assume we send a new message for simplicity
                  # message_id=???,
                  text=settings_message,
                  reply_markup=reply_markup,
                  parse_mode='Markdown'
             )
        except Exception as e:
            logger.warning(f"Could not edit message for settings, sending new one: {e}")
            # Fallback to sending a new message if edit fails or context is missing
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=settings_message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )


    async def generate_signal(self, asset, timeframe, platform, expiry_minutes=5):
        """
        Generate binary trading signal for the specified asset and timeframe

        Args:
            asset (str): Asset symbol
            timeframe (str): Timeframe interval (1min, 5min)
            platform (str): Trading platform
            expiry_minutes (int): Expiry time in minutes

        Returns:
            dict: Signal information or None if no signal
        """
        try:
            logger.info(f"Generating signal for {asset} on {timeframe} timeframe for {platform}")

            # Fetch asset data
            # Convert timeframe format if necessary (e.g., '1min' to '1m' if fetcher expects that)
            df = self.data_fetcher.get_asset_data(asset, timeframe, 100, platform)

            if df is None or df.empty:
                logger.warning(f"No data available for {asset}")
                return None

            # Analyze and generate signal
            _, signal_info = self.analyzer.analyze(df, expiry_minutes)

            if signal_info is None:
                logger.warning(f"Analysis failed for {asset}")
                return None

            # Set asset and platform in signal info
            signal_info['asset'] = asset
            signal_info['platform'] = platform

            # Only include signals that meet the threshold and are not "no signal"
            if signal_info['signal'] != 0 and signal_info['signal_strength'] >= self.signal_threshold:
                logger.info(f"Generated signal for {asset}: {signal_info['signal_type']} with {signal_info['accuracy']:.1f}% accuracy")
                return signal_info
            else:
                logger.info(f"No significant signal for {asset}")
                return None

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None

    def format_signal_message(self, signal):
        """
        Format signal into a readable message

        Args:
            signal (dict): Signal information

        Returns:
            str: Formatted message
        """
        if not signal:
            return "No signals available at this time."

        # Get current time in UTC
        now = datetime.now(pytz.UTC)

        # Format signal message
        message = f"🪙 *Binary Trading Signal* 🪙\n\n" # Added emoji
        message += f"📊 *Asset:* {signal['asset']}\n"
        message += f"🏢 *Platform:* {signal['platform']}\n"
        signal_emoji = "🔼" if signal['signal_type'] == "CALL" else "🔽" if signal['signal_type'] == "PUT" else "➖"
        message += f"{signal_emoji} *Signal:* {signal['signal_type']}\n"
        message += f"🎯 *Accuracy:* {signal['accuracy']:.1f}%\n"
        if signal.get('entry_price') is not None:
             message += f"💰 *Entry Price:* {signal['entry_price']:.5f}\n"

        # Format expiry time
        if signal.get('expiry_time'):
            expiry_time = signal['expiry_time']
            # Ensure expiry_time is timezone-aware (UTC)
            if expiry_time.tzinfo is None:
                expiry_time = pytz.UTC.localize(expiry_time)
            message += f"⏳ *Expiry Time:* {expiry_time.strftime('%H:%M:%S')} UTC\n"
            message += f"⏱️ *Expiry:* {signal['expiry_minutes']} minute(s)\n"

        # Add indicator values
        indicators = signal.get('indicators', {})
        message += f"\n*Indicators:*\n"
        message += f"• RSI: {indicators.get('rsi', 'N/A'):.1f}\n"
        message += f"• MACD: {indicators.get('macd', 'N/A'):.5f}\n"
        stoch_k = indicators.get('stoch_k', 'N/A')
        stoch_d = indicators.get('stoch_d', 'N/A')
        if stoch_k != 'N/A' and stoch_d != 'N/A':
             message += f"• Stochastic: {stoch_k:.1f} / {stoch_d:.1f}\n"
        else:
             message += f"• Stochastic: N/A\n"


        # Add timestamp
        message += f"\n*Generated at:* {now.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"

        # Add disclaimer
        message += "\n⚠️ *RISK DISCLAIMER:* This signal is for informational purposes only. " \
                  "Trade at your own risk and always use proper risk management."

        return message

    async def auto_signal_task(self, context: ContextTypes.DEFAULT_TYPE):
        """Background task to automatically send signals"""
        if not self.auto_signal:
            return

        logger.info("Running auto signal task")

        # Ensure application context is available
        if not hasattr(self, 'application') or not self.application:
            if context._application:
                 self.application = context._application
            else:
                 logger.error("Application context not available for auto signal task.")
                 return

        # Send signals to all authorized users
        for user_id in self.authorized_users:
            try:
                # Skip if user has no preferences
                if user_id not in self.user_preferences:
                    continue

                preferences = self.user_preferences[user_id]
                asset = preferences['asset']
                platform = preferences['platform']
                timeframe = preferences['timeframe']

                # Get expiry minutes based on timeframe
                expiry_minutes = 1 if timeframe == "1min" else 5

                # Generate signal
                signal = await self.generate_signal(asset, timeframe, platform, expiry_minutes)

                if signal:
                    message = self.format_signal_message(signal)
                    await context.bot.send_message( # Use context.bot here
                        chat_id=user_id,
                        text=message,
                        parse_mode='Markdown'
                    )
            except Exception as e:
                logger.error(f"Error sending auto signals to user {user_id}: {e}")

    async def setup_auto_signal_job(self, application):
        """Set up recurring job for auto signals"""
        # Store application context for later use
        self.application = application
        job_queue = application.job_queue

        # Run every minute for signals
        # Check if job already exists to prevent duplicates if setup is called multiple times
        current_jobs = job_queue.get_jobs_by_name('auto_signal_task')
        if not current_jobs:
            job_queue.run_repeating(
                self.auto_signal_task,
                interval=timedelta(minutes=self.signal_interval), # Use configured interval
                first=timedelta(seconds=10), # Start after 10 seconds
                name='auto_signal_task' # Name the job
            )
            logger.info(f"Auto signal job scheduled every {self.signal_interval} minute(s)")
        else:
            logger.info("Auto signal job already scheduled.")


async def main():
    """Start the bot"""
    # Create the bot instance
    binary_bot = BinarySignalBot()

    # Create the Application using the token from environment variable
    # The check for TELEGRAM_BOT_TOKEN happens earlier now
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Pass application context to the bot instance if methods need it early
    # binary_bot.application = application # This can be done here or in setup_auto_signal_job

    # Add command handlers
    application.add_handler(CommandHandler("start", binary_bot.start_command))
    application.add_handler(CommandHandler("help", binary_bot.help_command))
    application.add_handler(CommandHandler("signal", binary_bot.signal_command))
    application.add_handler(CommandHandler("settings", binary_bot.settings_command))
    application.add_handler(CommandHandler("platform", binary_bot.platform_command))

    # Add callback query handler
    application.add_handler(CallbackQueryHandler(binary_bot.button_callback))

    # Set up auto signal job
    # Pass application to the bot instance method if needed for setup
    await binary_bot.setup_auto_signal_job(application)

    # Start the Bot (Initialization and Polling)
    try:
        logger.info("Initializing application...")
        await application.initialize()
        logger.info("Starting polling...")
        await application.start()
        await application.updater.start_polling()
        logger.info("Bot polling started successfully.")

        # Keep the application running indefinitely (or until stopped)
        # In a threaded environment like app.py, this might not be reached
        # if run via asyncio.run() which blocks. The thread keeps it alive.
        # If running binary_bot.py directly, you'd need something here to keep main alive,
        # but start_polling itself usually does this. Let PTB handle the blocking.
        # await asyncio.Event().wait() # Example: Wait indefinitely


    except Exception as e:
         logger.critical(f"Error during bot startup or polling: {e}", exc_info=True)
         # Perform any necessary cleanup before exiting
         # await application.stop() # Ensure graceful shutdown if possible


# --- Main execution block ---
if __name__ == '__main__':
    # This block is relevant if running binary_bot.py directly
    # If run via app.py, app.py's asyncio.run(main()) calls the main() function above
    logger.info("Running bot directly (not via app.py)")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped manually.")
    except Exception as e:
        logger.critical(f"Unhandled error in main execution block: {e}", exc_info=True)

# --- End of binary_bot.py ---

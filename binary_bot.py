import os
import logging
import asyncio
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from binary_data import BinaryDataFetcher
from binary_analysis import BinaryAnalyzer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("binary_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get configuration from environment variables
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ADMIN_CHAT_ID = os.getenv('ADMIN_CHAT_ID')
SIGNAL_TIMEFRAMES = os.getenv('SIGNAL_TIMEFRAMES', '1,5').split(',')

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
        if ADMIN_CHAT_ID and ADMIN_CHAT_ID != 'YOUR_CHAT_ID_HERE':
            self.authorized_users.add(int(ADMIN_CHAT_ID))
        
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
            asset = query.data.split("_")[1]
            self.user_preferences[user_id]['asset'] = asset
            
            # After selecting asset, prompt for platform
            await query.edit_message_text(
                f"Asset selected: *{asset}*\n\nNow select your trading platform:",
                parse_mode='Markdown'
            )
            await self.show_platform_selection(query.message.chat_id)
        
        # Platform selection
        elif query.data.startswith("platform_"):
            platform = query.data.split("_")[1]
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
            
            await query.edit_message_text(settings_message, reply_markup=reply_markup, parse_mode='Markdown')
        
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
            await self.button_callback(update, context)  # Reuse callback to show settings
        
        elif query.data == "toggle_auto_signal":
            self.auto_signal = not self.auto_signal
            
            status = "enabled" if self.auto_signal else "disabled"
            await query.edit_message_text(f"Auto signal has been {status}.")
            
            # Return to settings
            await asyncio.sleep(1)  # Short delay for better UX
            await self.button_callback(update, context)  # Reuse callback to show settings
        
        elif query.data == "main_menu":
            keyboard = [
                [InlineKeyboardButton("Get Signals", callback_data="get_signals")],
                [InlineKeyboardButton("Settings", callback_data="settings")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                "Welcome to Binary Trading Signal Bot! Please select an option:",
                reply_markup=reply_markup
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
                logger.info(f"Generated signal for {asset}: {signal_info['signal_type']} with {signal_info['accuracy']}% accuracy")
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
        message = f"*Binary Trading Signal*\n\n"
        message += f"*Asset:* {signal['asset']}\n"
        message += f"*Platform:* {signal['platform']}\n"
        message += f"*Signal:* {signal['signal_type']}\n"
        message += f"*Accuracy:* {signal['accuracy']:.1f}%\n"
        message += f"*Entry Price:* {signal['entry_price']:.5f}\n"
        
        # Format expiry time
        if signal['expiry_time']:
            expiry_time = signal['expiry_time']
            message += f"*Expiry Time:* {expiry_time.strftime('%H:%M:%S')} UTC\n"
            message += f"*Expiry:* {signal['expiry_minutes']} minute(s)\n"
        
        # Add indicator values
        message += f"\n*Indicators:*\n"
        message += f"RSI: {signal['indicators']['rsi']:.1f}\n"
        message += f"MACD: {signal['indicators']['macd']:.5f}\n"
        message += f"Stochastic: {signal['indicators']['stoch_k']:.1f}/{signal['indicators']['stoch_d']:.1f}\n"
        
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
                    await context.bot.send_message(
                        chat_id=user_id,
                        text=message,
                        parse_mode='Markdown'
                    )
            except Exception as e:
                logger.error(f"Error sending auto signals to user {user_id}: {e}")
    
    async def setup_auto_signal_job(self, application):
        """Set up recurring job for auto signals"""
        self.application = application
        job_queue = application.job_queue
        
        # Run every minute for signals
        job_queue.run_repeating(
            self.auto_signal_task,
            interval=60,  # seconds
            first=10  # seconds until first execution
        )
        
        logger.info("Auto signal job scheduled")

async def main():
    """Start the bot"""
    # Create the bot instance
    binary_bot = BinarySignalBot()
    
    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", binary_bot.start_command))
    application.add_handler(CommandHandler("help", binary_bot.help_command))
    application.add_handler(CommandHandler("signal", binary_bot.signal_command))
    application.add_handler(CommandHandler("settings", binary_bot.settings_command))
    application.add_handler(CommandHandler("platform", binary_bot.platform_command))
    
    # Add callback query handler
    application.add_handler(CallbackQueryHandler(binary_bot.button_callback))
    
    # Set up auto signal job
    await binary_bot.setup_auto_signal_job(application)
    
    # Start the Bot
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    
    logger.info("Bot started")
    
    # Run the bot until the user presses Ctrl-C
    await application.updater.stop()
    await application.stop()

if __name__ == '__main__':
    asyncio.run(main())

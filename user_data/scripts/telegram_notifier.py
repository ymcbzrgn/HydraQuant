import os
import httpx
import logging

logger = logging.getLogger(__name__)

class AITelegramNotifier:
    def __init__(self, bot_token=None, chat_id=None):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram bot token or chat ID not set. Notifications will be disabled.")

    def _send_message(self, message: str):
        if not self.bot_token or not self.chat_id:
            return
            
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            response = httpx.post(url, json=payload, timeout=10.0)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    def send_trade_signal(self, pair: str, signal: str, confidence: float, reasoning_summary: str, position_pct: float = None):
        """Send notification when AI generates a trade signal."""
        direction = "BULLISH 🟢" if signal == "long" else "BEARISH 🔴" if signal == "short" else "NEUTRAL ⚪"
        msg = f"📊 *AI Signal:* {pair}\n"
        msg += f"Direction: *{direction}* (confidence: {confidence:.2f})\n\n"
        msg += f"Reasoning: {reasoning_summary}\n"
        if position_pct is not None:
            msg += f"\nPosition size: {position_pct:.1f}% of portfolio"
            
        self._send_message(msg)

    def send_daily_summary(self, stats: dict):
        """Send daily PNL and AI performance summary."""
        msg = "📈 *Daily AI Report*\n"
        open_trades = stats.get("open_trades", 0)
        closed_today = stats.get("closed_today", 0)
        msg += f"Open trades: {open_trades} | Closed today: {closed_today}\n"
        
        daily_pnl = stats.get("daily_pnl", 0.0)
        daily_pnl_pct = stats.get("daily_pnl_pct", 0.0)
        sign = "+" if daily_pnl > 0 else ""
        msg += f"Daily PNL: {sign}${daily_pnl:.2f} ({sign}{daily_pnl_pct:.2f}%)\n"
        
        accuracy = stats.get("accuracy", 0.0)
        correct_trades = stats.get("correct_trades", 0)
        total_trades = stats.get("total_eval_trades", 0)
        if total_trades > 0:
            msg += f"AI accuracy: {correct_trades}/{total_trades} ({accuracy:.0f}%)\n"
            
        api_cost = stats.get("api_cost_today", 0.0)
        msg += f"API cost today: ${api_cost:.2f}\n"
        
        autonomy_level = stats.get("autonomy_level", "L1")
        msg += f"Autonomy level: {autonomy_level}\n"
        
        portfolio_val = stats.get("portfolio_value", 0.0)
        if portfolio_val > 0:
            msg += f"Portfolio: *${portfolio_val:,.2f}*\n"

        # Asset breakdown with USD values
        assets = stats.get("assets", {})
        total_usd = 0.0
        asset_parts = []
        for currency, info in assets.items():
            if isinstance(info, dict) and "usd" in info:
                usd = info["usd"]
                total_usd += usd
                if usd >= 1.0:
                    asset_parts.append(f"{currency}: ${usd:,.2f}")
        if len(asset_parts) > 1:
            msg += "Varliklar: " + " | ".join(asset_parts) + "\n"
            msg += f"Toplam: *${total_usd:,.2f}*\n"

        forgone_pnl = stats.get("forgone_pnl", 0.0)
        msg += f"Forgone PNL: ${forgone_pnl:.2f} (signals NOT taken)\n"

        # $100 Hypothetical Portfolio
        hyp = stats.get("hypothetical", {})
        if hyp.get("total_trades", 0) > 0:
            balance = hyp["current_balance"]
            total_ret = hyp["total_return_pct"]
            total_trades = hyp["total_trades"]
            today_trades = hyp.get("today_trades", 0)
            today_pnl = hyp.get("today_pnl_pct", 0.0)
            sign = "+" if total_ret >= 0 else ""
            today_sign = "+" if today_pnl >= 0 else ""
            msg += f"\n{'='*24}\n"
            msg += f"*$100 ile oynasaydin:*\n"
            msg += f"Bakiye: *${balance:.2f}* ({sign}{total_ret:.2f}%)\n"
            msg += f"Toplam: {total_trades} trade\n"
            if today_trades > 0:
                msg += f"Bugun: {today_trades} trade ({today_sign}{today_pnl:.2f}%)"

        self._send_message(msg)

    def send_weekly_summary(self, stats: dict):
        """Send detailed weekly report."""
        msg = "📊 *Weekly AI Report*\n\n"
        msg += f"Win Rate: {stats.get('win_rate', 0):.1f}%\n"
        msg += f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}\n"
        msg += f"Max Drawdown: {stats.get('max_drawdown', 0):.2f}%\n"
        
        forgone = stats.get("forgone_pnl_total", 0.0)
        msg += f"Forgone PNL Total: ${forgone:.2f}\n"

        # $100 Hypothetical Portfolio (Weekly)
        hyp = stats.get("hypothetical", {})
        if hyp.get("total_trades", 0) > 0:
            balance = hyp["current_balance"]
            total_ret = hyp["total_return_pct"]
            best = hyp.get("best_trade_pct", 0.0)
            worst = hyp.get("worst_trade_pct", 0.0)
            sign = "+" if total_ret >= 0 else ""
            msg += f"\n*$100 Simulasyon:*\n"
            msg += f"Bakiye: *${balance:.2f}* ({sign}{total_ret:.2f}%)\n"
            msg += f"En iyi trade: {best:+.2f}%\n"
            msg += f"En kotu trade: {worst:+.2f}%"

        self._send_message(msg)

    def send_alert(self, message: str, level: str = "INFO"):
        """Send critical alerts like budget warnings or autonomy level changes."""
        icon = "⚠️"
        if level.upper() == "CRITICAL" or level.upper() == "ERROR":
            icon = "🔴"
        elif level.upper() == "WARNING":
            icon = "🟡"
            
        msg = f"{icon} *ALERT*: {message}"
        self._send_message(msg)

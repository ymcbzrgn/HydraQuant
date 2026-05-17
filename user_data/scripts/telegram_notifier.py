import os
import time
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
import httpx
import logging

logger = logging.getLogger(__name__)

# Module-level alert cooldown: maps message_key → last_sent_timestamp
_ALERT_COOLDOWNS: dict[str, float] = {}
# Default cooldown: 6 hours (21600 seconds)
_ALERT_COOLDOWN_SECS = 6 * 3600

class AITelegramNotifier:
    def __init__(self, bot_token=None, chat_id=None):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram bot token or chat ID not set. Notifications will be disabled.")

    def _send_message(self, message: str, parse_mode: str = "Markdown"):
        if not self.bot_token or not self.chat_id:
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }
        try:
            response = httpx.post(url, json=payload, timeout=10.0)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Telegram message ({parse_mode}): {e}")
            # 2026-05-18: Markdown often breaks on dynamic content with `_` or `*` (e.g.
            # exit_reason 'stale_8h_flat__global', blocked_by 'pnl_positive'). HTML
            # mode is stricter and safer for the rich daily/weekly reports. If the
            # caller passed HTML/MarkdownV2 and it still failed, retry as plain text
            # so the user at least sees the data.
            if parse_mode:
                try:
                    payload2 = {"chat_id": self.chat_id, "text": message}
                    httpx.post(url, json=payload2, timeout=10.0).raise_for_status()
                    logger.info("Telegram fallback (no parse_mode) succeeded")
                except Exception as e2:
                    logger.error(f"Plain-text Telegram fallback also failed: {e2}")

    def send_trade_signal(self, pair: str, signal: str, confidence: float, reasoning_summary: str, position_pct: float = None):
        """Send notification when AI generates a trade signal."""
        direction = "BULLISH 🟢" if signal == "long" else "BEARISH 🔴" if signal == "short" else "NEUTRAL ⚪"
        msg = f"📊 *AI Signal:* {pair}\n"
        msg += f"Direction: *{direction}* (confidence: {confidence:.2f})\n\n"
        msg += f"Reasoning: {reasoning_summary}\n"
        if position_pct is not None:
            msg += f"\nPosition size: {position_pct:.1f}% of portfolio"
            
        self._send_message(msg)

    @staticmethod
    def _fmt_usd(x: float) -> str:
        try:
            v = float(x)
        except (TypeError, ValueError):
            v = 0.0
        sign = "+" if v >= 0 else "−"
        return f"{sign}${abs(v):.2f}"

    @staticmethod
    def _fmt_pct(x: float) -> str:
        try:
            v = float(x)
        except (TypeError, ValueError):
            v = 0.0
        sign = "+" if v >= 0 else "−"
        return f"{sign}{abs(v):.2f}%"

    def send_daily_summary(self, stats: dict):
        """2026-05-18 rewrite: detailed end-of-day report.

        Section order is deliberate: PnL first (user's primary signal), then trades,
        then open positions, AI quality, infrastructure, promotion gate, forgone,
        portfolio, $100 sim. Every field is sourced from real DB queries in
        scheduler._compute_daily_stats() — no more placeholder zeros.
        """
        from datetime import datetime as _dt

        today = _dt.utcnow().strftime("%d %b %Y")
        L: list[str] = []
        L.append(f"🏁 *GÜNLÜK RAPOR — {today}*")
        L.append("━━━━━━━━━━━━━━━━━━━")

        # ── PnL (headline) ──────────────────────────────────────
        L.append("")
        L.append("💰 *PnL — TR-DRY paper bot*")
        L.append(f"  Bugün (24h):  *{self._fmt_usd(stats.get('pnl_abs_24h', 0))}*  "
                 f"({self._fmt_pct(stats.get('pnl_pct_24h', 0))})")
        L.append(f"  Bu hafta (7g): {self._fmt_usd(stats.get('pnl_abs_7d', 0))}  "
                 f"({self._fmt_pct(stats.get('pnl_pct_7d', 0))})")
        L.append(f"  Son 30 gün:   {self._fmt_usd(stats.get('pnl_abs_30d', 0))}")
        L.append(f"  Tüm zaman:    {self._fmt_usd(stats.get('pnl_abs_all', 0))}  "
                 f"({stats.get('closed_all', 0)} trade)")

        # ── Trades 24h ──────────────────────────────────────────
        L.append("")
        L.append("📊 *Trade'ler — 24 saat*")
        n_24h = stats.get("closed_24h", 0)
        w_24h = stats.get("wins_24h", 0)
        l_24h = max(n_24h - w_24h, 0)
        wr = (w_24h / n_24h * 100) if n_24h else 0
        L.append(f"  Kapanan: {n_24h}  (W: {w_24h} | L: {l_24h} | winrate %{wr:.0f})")
        if n_24h:
            L.append(f"  Avg hold: {stats.get('avg_hold_24h', 0):.1f}h  |  "
                     f"avg PnL/trade: {self._fmt_pct(stats.get('avg_pnl_pct_24h', 0))}")
        best = stats.get("best_24h")
        worst = stats.get("worst_24h")
        if best:
            side = "SHORT" if best.get("is_short") else "LONG"
            L.append(f"  En iyi:  {best['pair']} {side}  "
                     f"{self._fmt_pct((best.get('close_profit') or 0) * 100)}  "
                     f"({self._fmt_usd(best.get('close_profit_abs') or 0)})")
        if worst and (not best or worst['pair'] != best['pair']
                      or worst.get('close_profit_abs') != best.get('close_profit_abs')):
            side = "SHORT" if worst.get("is_short") else "LONG"
            L.append(f"  En kötü: {worst['pair']} {side}  "
                     f"{self._fmt_pct((worst.get('close_profit') or 0) * 100)}  "
                     f"({self._fmt_usd(worst.get('close_profit_abs') or 0)})")

        # ── Open positions ──────────────────────────────────────
        opens = stats.get("open_now") or []
        L.append("")
        L.append(f"🟢 *Açık pozisyon: {len(opens)}*")
        for o in opens[:5]:
            side = "SHORT" if o.get("is_short") else "LONG"
            L.append(f"  {o['pair']} {side}  ${(o.get('stake_amount') or 0):.0f} "
                     f"@ {(o.get('open_rate') or 0):.4f}  "
                     f"({(o.get('hours_open') or 0):.1f}h açık)")
        if len(opens) > 5:
            L.append(f"  …+{len(opens) - 5} more")

        # ── Exit reasons ────────────────────────────────────────
        er = stats.get("exit_reasons_24h") or {}
        if er:
            L.append("")
            L.append("⚠️ *Exit reason dağılımı (24h):*")
            for reason, count in list(er.items())[:5]:
                L.append(f"  {reason}: {count}")

        # ── AI decision quality (7d) ────────────────────────────
        total = stats.get("ai_decisions_7d_total", 0)
        resolved = stats.get("ai_decisions_7d_resolved", 0)
        winners = stats.get("ai_decisions_7d_winners", 0)
        avg_conf = stats.get("ai_decisions_7d_avg_conf", 0)
        L.append("")
        L.append("🤖 *AI Karar Kalitesi (son 7 gün)*")
        L.append(f"  Toplam BULLISH/BEARISH karar: {total}")
        if resolved:
            acc = winners / resolved * 100
            L.append(f"  Çözülen: {resolved}  doğru yön: {winners}/{resolved} (%{acc:.0f})")
        L.append(f"  Avg confidence: {avg_conf:.2f}")

        # ── RAG / LLM ───────────────────────────────────────────
        L.append("")
        L.append("📞 *RAG / LLM (24h)*")
        L.append(f"  /signal p95 latency: {(stats.get('rag_p95_latency_ms', 0) / 1000):.1f}s  |  "
                 f"call: {stats.get('rag_signal_calls_24h', 0)}  |  "
                 f"timeout: {stats.get('rag_timeout_breaches', 0)}")
        L.append(f"  LLM API: {stats.get('llm_calls_24h', 0)} çağrı  |  "
                 f"cost: ${stats.get('llm_cost_24h', 0):.4f}")

        # ── System health ───────────────────────────────────────
        L.append("")
        L.append("🚦 *Sistem*")
        L.append(f"  Servisler: {stats.get('services_active', 0)}/"
                 f"{stats.get('services_total', 0)} active  |  "
                 f"Total restart: {stats.get('restarts_total', 0)}")
        if stats.get("ram_used_gb"):
            L.append(f"  RAM: {stats['ram_used_gb']:.1f} / "
                     f"{stats.get('ram_total_gb', 0):.1f} GB  |  "
                     f"Autonomy: {stats.get('autonomy_level', '?')}")
        else:
            L.append(f"  Autonomy: {stats.get('autonomy_level', '?')}")

        # ── Promotion gate ──────────────────────────────────────
        pg = stats.get("promotion_gate") or {}
        if pg.get("metrics"):
            m = pg["metrics"]
            elig_raw = pg.get("eligibility_pct", 0)
            elig_pct = elig_raw * 100 if elig_raw <= 1 else elig_raw
            blocked = pg.get("blocked_by") or []
            L.append("")
            L.append("🔒 *Promotion Gate (Real-Capital)*")
            L.append(f"  Eligibility: %{elig_pct:.1f}  |  blocked: {len(blocked)}/8 kriter")
            L.append(f"  n_trades: {m.get('n_trades', 0)}  |  "
                     f"winrate: %{(m.get('winrate', 0) or 0) * 100:.0f}  |  "
                     f"sharpe: {m.get('sharpe', 0):.2f}")
            L.append(f"  pnl: ${m.get('pnl_usdt', 0):.2f}  |  "
                     f"max_dd: %{(m.get('max_dd', 0) or 0) * 100:.0f}  |  "
                     f"liquid: {m.get('n_liquid', 0)}")
            if blocked:
                L.append(f"  Engelleyen: {', '.join(blocked[:4])}"
                         + (f" +{len(blocked) - 4} more" if len(blocked) > 4 else ""))

        # ── Forgone ─────────────────────────────────────────────
        L.append("")
        L.append("🌑 *Forgone (alınmayan sinyaller — 24h)*")
        L.append(f"  Toplam: {stats.get('forgone_count_24h', 0)} sinyal  |  "
                 f"forgone PnL: {self._fmt_pct(stats.get('forgone_pnl_24h', 0))}")
        for f in (stats.get("forgone_top") or [])[:3]:
            L.append(f"  {f['pair']} {f['signal_type']} c={f.get('confidence', 0):.2f} → "
                     f"+{f.get('forgone_pnl', 0):.1f}%")

        # ── Portfolio ───────────────────────────────────────────
        pv = stats.get("portfolio_value", 0)
        assets = stats.get("assets") or {}
        if assets and pv:
            L.append("")
            L.append("💼 *Portfolyo*")
            L.append(f"  Toplam: *${pv:,.2f}*")
            parts = []
            for ccy, info in assets.items():
                if isinstance(info, dict) and info.get("usd", 0) >= 1:
                    parts.append(f"{ccy}: ${info['usd']:,.2f}")
            if parts:
                L.append("  " + " | ".join(parts[:5]))

        # ── $100 simulation ─────────────────────────────────────
        hyp = stats.get("hypothetical") or {}
        if hyp.get("total_trades", 0) > 0:
            L.append("")
            L.append("💯 *$100 Simülasyon (kümülatif)*")
            L.append(f"  Bakiye: *${hyp['current_balance']:.2f}*  "
                     f"({self._fmt_pct(hyp.get('total_return_pct', 0))})  |  "
                     f"Toplam: {hyp['total_trades']} trade")
            if hyp.get("today_trades"):
                L.append(f"  Bugün: {hyp['today_trades']} trade "
                         f"({self._fmt_pct(hyp.get('today_pnl_pct', 0))})")

        msg = "\n".join(L)
        if len(msg) > 4000:
            msg = msg[:3990] + "\n…(truncated)"
        self._send_message(msg)

    def send_weekly_summary(self, stats: dict):
        """2026-05-18 rewrite: real 7-day report.

        Previous version sent win_rate=0/sharpe=0/max_drawdown=0 placeholders. Now
        reads from _compute_daily_stats() (in scheduler) which pulls real trades,
        sharpe/max_dd from promotion_gate API, and per-day PnL breakdown.
        """
        from datetime import datetime as _dt

        today = _dt.utcnow().strftime("%d %b %Y")
        L: list[str] = []
        L.append(f"📅 *HAFTALIK RAPOR — {today}*")
        L.append("━━━━━━━━━━━━━━━━━━━")

        L.append("")
        L.append("💰 *Haftalık PnL (TR-DRY paper)*")
        L.append(f"  7 günlük PnL: *{self._fmt_usd(stats.get('pnl_abs_7d', 0))}*  "
                 f"({self._fmt_pct(stats.get('pnl_pct_7d', 0))})")
        L.append(f"  30 günlük:    {self._fmt_usd(stats.get('pnl_abs_30d', 0))}")
        L.append(f"  Tüm zaman:    {self._fmt_usd(stats.get('pnl_abs_all', 0))}  "
                 f"({stats.get('closed_all', 0)} trade)")

        L.append("")
        L.append("📊 *Trade istatistik (7g)*")
        n7 = stats.get("closed_7d", 0)
        w7 = stats.get("wins_7d", 0)
        wr = stats.get("win_rate_7d", 0.0)
        L.append(f"  Kapanan: {n7}  W: {w7}  L: {max(n7 - w7, 0)}  winrate %{wr:.1f}")
        L.append(f"  Sharpe: {stats.get('sharpe_ratio', 0):.2f}  |  "
                 f"Max DD: %{stats.get('max_drawdown_pct', 0):.1f}")
        L.append(f"  Avg hold: {stats.get('avg_hold_7d', 0):.1f}h  |  "
                 f"avg PnL/trade: {self._fmt_pct(stats.get('avg_pnl_pct_7d', 0))}")

        daily_pnl = stats.get("weekly_pnl_by_day") or []
        if daily_pnl:
            L.append("")
            L.append("📆 *Günlük dağılım (son 7 gün):*")
            for d in daily_pnl[:7]:
                L.append(f"  {d['day']}: {d['n']} trade  {self._fmt_usd(d['pnl'])}")

        opens = stats.get("open_now") or []
        L.append("")
        L.append(f"🟢 *Açık pozisyon şu an: {len(opens)}*")

        L.append("")
        L.append("📞 *RAG / LLM (24h örnek)*")
        L.append(f"  Signal calls: {stats.get('rag_signal_calls_24h', 0)}  |  "
                 f"p95: {(stats.get('rag_p95_latency_ms', 0) / 1000):.1f}s")
        L.append(f"  LLM: {stats.get('llm_calls_24h', 0)} çağrı  |  "
                 f"cost: ${stats.get('llm_cost_24h', 0):.4f}")

        pg = stats.get("promotion_gate") or {}
        if pg.get("metrics"):
            m = pg["metrics"]
            elig_raw = pg.get("eligibility_pct", 0)
            elig_pct = elig_raw * 100 if elig_raw <= 1 else elig_raw
            blocked = pg.get("blocked_by") or []
            L.append("")
            L.append("🔒 *Promotion Gate*")
            L.append(f"  Eligibility: %{elig_pct:.1f}  |  blocked: {len(blocked)}/8")
            if blocked:
                L.append(f"  Engelleyen: {', '.join(blocked[:4])}"
                         + (f" +{len(blocked) - 4} more" if len(blocked) > 4 else ""))

        L.append("")
        L.append("🌑 *Forgone (24h)*: "
                 f"{stats.get('forgone_count_24h', 0)} sinyal  "
                 f"{self._fmt_pct(stats.get('forgone_pnl_24h', 0))}")

        hyp = stats.get("hypothetical") or {}
        if hyp.get("total_trades", 0) > 0:
            L.append("")
            L.append("💯 *$100 Simülasyon*")
            L.append(f"  Bakiye: *${hyp['current_balance']:.2f}*  "
                     f"({self._fmt_pct(hyp.get('total_return_pct', 0))})")
            if hyp.get("best_trade_pct") is not None:
                L.append(f"  En iyi: {hyp['best_trade_pct']:+.2f}%  |  "
                         f"En kötü: {hyp.get('worst_trade_pct', 0):+.2f}%")

        msg = "\n".join(L)
        if len(msg) > 4000:
            msg = msg[:3990] + "\n…(truncated)"
        self._send_message(msg)

    def send_alert(self, message: str, level: str = "INFO", cooldown_secs: int = None):
        """Send critical alerts with dedup cooldown to prevent spam.

        Same alert message won't be re-sent within cooldown window (default 6h).

        Phase 30 A.5 — heartbeat_suppression second-level filter for non-CRITICAL
        events: if same (level, message_head) seen <300s ago, drop silently. This
        is on TOP of the per-message cooldown above and catches near-identical
        messages whose body text varies (timestamps, counts).
        """
        # ═══ PHASE 30 A.5 — Heartbeat suppression on lower-severity alerts ═══
        try:
            if level.upper() not in ("CRITICAL", "ERROR"):
                from heartbeat_suppression import should_emit as _phase30_he
                if not _phase30_he(f"telegram.{level.lower()}", message[:120]):
                    logger.debug(f"[Phase30:Heartbeat] suppressed {level} alert: {message[:60]}")
                    return
        except Exception:
            pass

        cooldown = cooldown_secs if cooldown_secs is not None else _ALERT_COOLDOWN_SECS

        # Dedup check: skip if same message was sent recently
        now = time.time()
        if message in _ALERT_COOLDOWNS:
            elapsed = now - _ALERT_COOLDOWNS[message]
            if elapsed < cooldown:
                logger.debug(f"Alert suppressed (cooldown {cooldown - elapsed:.0f}s remaining): {message}")
                return

        icon = "⚠️"
        if level.upper() == "CRITICAL" or level.upper() == "ERROR":
            icon = "🔴"
        elif level.upper() == "WARNING":
            icon = "🟡"

        msg = f"{icon} *ALERT*: {message}"
        self._send_message(msg)
        _ALERT_COOLDOWNS[message] = now

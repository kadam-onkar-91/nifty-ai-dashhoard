import requests
import time

class AlertManager:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.last_alert_time = 0
        self.cooldown_period = 300  # 5 minutes cooldown to avoid spam

    def send_trade_alert(self, signal_type, confidence, price, sentiment, logic):
        # Cooldown check
        current_time = time.time()
        if (current_time - self.last_alert_time) < self.cooldown_period:
            return False

        # Professional Message Formatting
        message = (
            f"🔔 *TRADE ALERT: {signal_type.upper()}*\n\n"
            f"💰 *Price:* ₹{price}\n"
            f"🎯 *Confidence:* {confidence}%\n"
            f"📊 *Sentiment:* {sentiment}\n"
            f"🧠 *Logic:* {logic}\n\n"
            f"🚀 _Institutional AI Engine_"
        )

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }

        try:
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                self.last_alert_time = current_time
                return True
            else:
                print(f"Failed to send alert: {response.text}")
                return False
        except Exception as e:
            print(f"Error sending Telegram alert: {e}")
            return False

    # =====================================================================
    # NEW — PRE-MOVE SQUEEZE ALERT (added on top, nothing above touched)
    # -------------------------------------------------------------------
    # Companion to send_trade_alert/send_full_confluence_alert -- those
    # fire on a CONFIRMED signal. This one fires EARLIER, while price is
    # still coiled (Bollinger squeeze), so the user gets a heads-up on
    # which way the coil is leaning before the breakout candle prints.
    # Deliberately has no strike/entry/SL/target -- it's a "watch this"
    # ping, not a trade call. Caller (app.py) handles anti-spam via
    # st.session_state, same pattern as the confluence alert.
    # =====================================================================
    def send_pre_move_alert(self, lean, price, squeeze_candles, lean_score_pct,
                             trigger_up, trigger_down, top_factor=None):
        if lean not in ("BULLISH", "BEARISH"):
            return False

        trigger_text = (f"↑ ₹{trigger_up:,.2f}" if lean == "BULLISH" else f"↓ ₹{trigger_down:,.2f}")
        message = (
            f"⚡ *PRE-MOVE SQUEEZE ALERT: {lean} LEAN*\n\n"
            f"💰 *Price:* ₹{price:,.2f}\n"
            f"🌀 *Coiling:* {squeeze_candles} candles\n"
            f"📊 *Lean Score:* {lean_score_pct}%\n"
            f"🎯 *Confirm Trigger:* {trigger_text}\n"
            + (f"🧠 *Top Factor:* {top_factor}\n" if top_factor else "")
            + f"\n⚠️ _Compression building — this is a watch alert, NOT a trade call. Wait for the trigger to actually break._"
        )

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }

        try:
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                return True
            else:
                print(f"Failed to send pre-move alert: {response.text}")
                return False
        except Exception as e:
            print(f"Error sending Telegram pre-move alert: {e}")
            return False

    # =====================================================================
    # NEW — FULL CONFLUENCE CHECKLIST ALERT (added on top, nothing above
    # this line was touched)
    # -------------------------------------------------------------------
    # Sends a Telegram alert ONLY when every item on the trade checklist
    # lines up at once: (1) main signal is BUY/SELL with decent confidence,
    # (2) the ML model agrees with that same direction, (3) the nearest
    # support/resistance level's Break/Bounce % is >= min_level_pct in that
    # direction, (4) Bank Nifty is NOT flagging a divergence warning.
    #
    # This method only CHECKS the criteria and SENDS -- it does not decide
    # whether this exact setup was already alerted moments ago. That "have
    # I already sent this" memory needs to live in st.session_state on the
    # caller's side (app.py), because Streamlit reruns the whole script on
    # every refresh -- an instance attribute here would reset to nothing
    # every 30 seconds and the anti-spam logic would never actually work.
    # See app.py for the edge-triggered, once-per-new-setup wiring that
    # calls this.
    # =====================================================================
    def send_full_confluence_alert(self, direction, price, main_confidence, ml_confidence,
                                    level_price, level_pct, level_read, oi_note,
                                    banknifty_note, global_note=None, breadth_note=None,
                                    min_level_pct=65.0):
        """
        Returns True if an alert was actually sent, False if criteria
        weren't met or the send failed.
        """
        if not direction or direction not in ("BUY", "SELL"):
            return False
        if main_confidence is None or main_confidence < 55:
            return False
        if ml_confidence is None:  # caller passes None when the ML model disagrees/unavailable
            return False
        if level_pct is None or level_pct < min_level_pct:
            return False
        if banknifty_note and "DIVERGENCE WARNING" in banknifty_note.upper():
            return False

        message = (
            f"🚨 *FULL CONFLUENCE ALERT: {direction}*\n\n"
            f"💰 *Price:* ₹{price:,.2f}\n"
            f"🎯 *Main Signal Confidence:* {main_confidence}%\n"
            f"🤖 *ML Model Confidence:* {ml_confidence}%\n"
            f"📐 *Nearest Level:* ₹{level_price:,.2f} → {level_read} ({level_pct}%)\n"
            f"📊 *OI/PCR:* {oi_note}\n"
            f"🏦 *Bank Nifty:* {banknifty_note or 'No divergence flagged'}\n"
            + (f"🌍 *Global:* {global_note}\n" if global_note else "")
            + (f"📈 *Breadth:* {breadth_note}\n" if breadth_note else "")
            + f"\n✅ _All checklist conditions aligned right now — verify before acting, this is not a guarantee._"
        )

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }

        try:
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                return True
            else:
                print(f"Failed to send full confluence alert: {response.text}")
                return False
        except Exception as e:
            print(f"Error sending Telegram full confluence alert: {e}")
            return False

    # =====================================================================
    # NEW — TIERED CONFLUENCE ALERT (added on top of everything above --
    # send_full_confluence_alert() above is left exactly as-is and unused
    # by app.py now, in case it's ever wanted again)
    # -------------------------------------------------------------------
    # Answers the "good trades AND more trades" trade-off directly:
    # instead of all-or-nothing, this SCORES how many of the 4 checklist
    # items are met and sends ONE of two alert strengths:
    #   - "FULL CONFLUENCE"  (all 4 met, including level% >= strict 65%)
    #     -- rare, highest quality, same bar as before
    #   - "STRONG SETUP"     (at least 3 of 4 met, level% can be as low
    #     as the looser 55%) -- noticeably more frequent, still filters
    #     out genuinely weak setups (below 3/4 sends nothing)
    # =====================================================================
    def evaluate_and_send_confluence_alert(self, direction, price, main_confidence, ml_confidence,
                                            level_price, level_pct, level_read, oi_note,
                                            banknifty_note, global_note=None, breadth_note=None,
                                            min_level_pct_full=65.0, min_level_pct_partial=55.0):
        """
        Returns (sent: bool, tier: 'FULL' | 'STRONG' | None).
        tier is None whenever sent is False.
        """
        if not direction or direction not in ("BUY", "SELL"):
            return False, None

        checks_met = 0
        if main_confidence is not None and main_confidence >= 55:
            checks_met += 1
        if ml_confidence is not None:
            checks_met += 1
        level_ok_full = level_pct is not None and level_pct >= min_level_pct_full
        level_ok_partial = level_pct is not None and level_pct >= min_level_pct_partial
        if level_ok_partial:
            checks_met += 1
        banknifty_ok = not (banknifty_note and "DIVERGENCE WARNING" in banknifty_note.upper())
        if banknifty_ok:
            checks_met += 1

        if checks_met >= 4 and level_ok_full:
            tier, emoji, label = "FULL", "🚨", "FULL CONFLUENCE ALERT"
        elif checks_met >= 3:
            tier, emoji, label = "STRONG", "⚡", "STRONG SETUP ALERT"
        else:
            return False, None  # still too weak -- stay silent, don't spam a low-quality setup

        message = (
            f"{emoji} *{label}: {direction}* ({checks_met}/4 checklist items met)\n\n"
            f"💰 *Price:* ₹{price:,.2f}\n"
            f"🎯 *Main Signal Confidence:* {main_confidence}%\n"
            f"🤖 *ML Model:* {f'{ml_confidence}% agrees' if ml_confidence is not None else 'Disagrees / no confident read'}\n"
            f"📐 *Nearest Level:* ₹{level_price:,.2f} → {level_read} ({level_pct}%)\n"
            f"📊 *OI/PCR:* {oi_note}\n"
            f"🏦 *Bank Nifty:* {banknifty_note or 'No divergence flagged'}\n"
            + (f"🌍 *Global:* {global_note}\n" if global_note else "")
            + (f"📈 *Breadth:* {breadth_note}\n" if breadth_note else "")
            + (
                "\n✅ _All 4 checklist conditions aligned — the higher-quality setup. Still not a guarantee._"
                if tier == "FULL" else
                "\n⚠️ _Most (not all) conditions aligned — decent setup but treat with extra caution. Still not a guarantee._"
            )
        )

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }

        try:
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                return True, tier
            else:
                print(f"Failed to send confluence alert: {response.text}")
                return False, None
        except Exception as e:
            print(f"Error sending Telegram confluence alert: {e}")
            return False, None

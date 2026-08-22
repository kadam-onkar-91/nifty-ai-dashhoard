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

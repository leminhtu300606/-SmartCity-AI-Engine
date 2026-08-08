import json
import logging
import threading
import paho.mqtt.client as mqtt
import config

logging.basicConfig(level=logging.INFO)


class SensorMQTTConsumer:
    """Module tiêu thụ và phân tích Sensor Data từ MQTT Stream."""

    def __init__(self):
        self.client = mqtt.Client()
        self.client.username_pw_set(config.MQTT_USER, config.MQTT_PASS)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.latest_sensor_data = {}
        self.sensor_alerts = []

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info(f"[MQTT] Connected successfully to {config.MQTT_BROKER}")
            self.client.subscribe(config.MQTT_TOPIC)
        else:
            logging.error(f"[MQTT] Connect failed with code {rc}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            device_id = payload.get("device_id", "UNKNOWN")
            metrics = payload.get("metrics", {})

            self.latest_sensor_data[device_id] = payload

            # Rule kiểm tra an toàn điện / Anomaly Detection từ Sensor Data
            voltage = metrics.get("voltage_v")
            load_pct = metrics.get("load_pct")

            # Cảnh báo quá tải hoặc sụt áp bất thường (chỉ khi metric thực tế có dữ liệu)
            if (
                (load_pct is not None and load_pct > 90.0)
                or (voltage is not None and (voltage < 180.0 or voltage > 250.0))
            ):
                alert = {
                    "event_type": "ELECTRICAL_ANOMALY",
                    "device_id": device_id,
                    "voltage": voltage,
                    "load_pct": load_pct,
                    "description": f"Bất thường điện áp ({voltage}V) hoặc Quá tải ({load_pct}%)",
                }
                self.sensor_alerts.append(alert)
                logging.warning(f"[SENSOR ALERT] {alert['description']}")

        except Exception as e:
            logging.error(f"[MQTT] Error parsing payload: {e}")

    def start(self):
        """Khởi chạy client MQTT ở background thread."""
        try:
            self.client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
            t = threading.Thread(target=self.client.loop_forever, daemon=True)
            t.start()
            logging.info("[MQTT] Background thread started.")
        except Exception as e:
            logging.error(f"[MQTT] Could not start MQTT thread: {e}")

    def get_latest_alerts(self):
        alerts = list(self.sensor_alerts)
        self.sensor_alerts.clear()
        return alerts
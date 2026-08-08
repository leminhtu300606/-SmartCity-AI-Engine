import json
import time
import uuid
import numpy as np
import paho.mqtt.client as mqtt
import config

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom Encoder xử lý triệt để các kiểu dữ liệu NumPy"""
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

class MQTTPublisher:
    def __init__(self):
        self.client = mqtt.Client(client_id=f"vision_engine_{uuid.uuid4().hex[:6]}")
        self.client.username_pw_set(config.MQTT_USER, config.MQTT_PASS)
        self.last_sent_time = {}  # Key: (camera_id, event_type, track_tuple) -> timestamp

        try:
            self.client.connect(config.MQTT_HOST, config.MQTT_PORT, 60)
            self.client.loop_start()
            print(f"[MQTT] Connected successfully to {config.MQTT_HOST}")
        except Exception as e:
            print(f"[MQTT ERROR] Connection failed: {e}")

    def build_payload(self, camera_id, event_data):
        return {
            "event_id": str(uuid.uuid4()),
            "schema_version": "1.0",
            "source": "smartvision_ai_engine",
            "camera_id": camera_id,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "event_type": event_data["event_type"],
            "confidence": round(float(event_data["confidence"]), 4),
            "track_ids": event_data["track_ids"],
            "zone": event_data.get("zone_name"),
            "description": event_data["description"],
        }

    def publish_event(self, camera_id, event_data):
        now = time.time()
        track_key = tuple(sorted(event_data.get("track_ids", [])))
        event_key = (camera_id, event_data["event_type"], track_key)

        # Anti-Spam / Cooldown Check
        if event_key in self.last_sent_time:
            if now - self.last_sent_time[event_key] < config.MQTT_COOLDOWN_SEC:
                return

        self.last_sent_time[event_key] = now
        topic = f"{config.MQTT_TOPIC_PREFIX}/{camera_id}"
        payload = self.build_payload(camera_id, event_data)

        self.client.publish(topic, json.dumps(payload, cls=NumpyJSONEncoder), qos=1)
        print(f"[MQTT SENT -> {topic}] {event_data['event_type']} (Confidence: {payload['confidence']})")
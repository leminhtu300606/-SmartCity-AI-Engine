import cv2
import time
import threading
from queue import Queue
import config

class ThreadedFrameReader:
    """Thread riêng để kéo luồng HLS/RTSP. Giúp AI luôn đọc frame mới nhất, không bị tích tụ trễ."""
    def __init__(self, camera_id, stream_url):
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.queue = Queue(maxsize=config.QUEUE_SIZE)
        self.stopped = False
        self.cap = cv2.VideoCapture(self.stream_url)

    def start(self):
        t = threading.Thread(target=self._update, name=f"Reader-{self.camera_id}", daemon=True)
        t.start()
        return self

    def _update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                print(f"[{self.camera_id}] Reconnecting stream...")
                self.cap.open(self.stream_url)
                time.sleep(1.0)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Đẩy frame vào Queue. Nếu full, drop frame cũ nhất để duy trì real-time
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Exception:
                    pass
            self.queue.put(frame)

    def read(self):
        if not self.queue.empty():
            return True, self.queue.get()
        return False, None

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()
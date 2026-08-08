import threading
import cv2
import torch
from ultralytics import YOLO

import config
from core.pipeline import CameraPipelineWorker
from output.mqtt_publisher import MQTTPublisher


def main():
    print("=" * 60)
    print("  SMARTVISION AI ENGINE PHASE 1 - SMOOTH GUI DISPLAY ")
    print("=" * 60)

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] Hardware Acceleration Device: {device.upper()}")

    det_model = YOLO(config.MODEL_DET_PATH)
    pose_model = YOLO(config.MODEL_POSE_PATH)
    mqtt_pub = MQTTPublisher()

    workers = {}
    for cam_id, stream_url in config.CAMERA_STREAMS.items():
        worker = CameraPipelineWorker(cam_id, stream_url, det_model, pose_model, mqtt_pub, device)
        t = threading.Thread(target=worker.run, daemon=True)
        t.start()
        workers[cam_id] = worker

    print(f"[SYSTEM] Running pipelines for {len(workers)} cameras.")
    print("[SYSTEM] Press 'q' on any video window to exit.")

    # Luồng chính (Main Thread) chuyên làm nhiệm vụ render hiển thị video
    try:
        while True:
            for cam_id, worker in workers.items():
                if worker.latest_display_frame is not None:
                    cv2.imshow(f"SmartVision AI - {cam_id}", worker.latest_display_frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("\n[SYSTEM] Stopped successfully.")


if __name__ == "__main__":
    main()
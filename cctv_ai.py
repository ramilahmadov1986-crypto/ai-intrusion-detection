import os

# FFmpeg mesajlarını azalt
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import time
import threading
import numpy as np
import pygame
from datetime import datetime
from ultralytics import YOLO

# Pygame Audio
pygame.mixer.init()

# ============================================================
# CAMERA & SETTINGS
# ============================================================
RTSP_URL=rtsp://USERNAME:PASSWORD@CAMERA_IP:554/media/video1"
MODEL_NAME = "yolo11n.pt"
CONFIDENCE = 0.30
AI_SIZE = 416
NOTIFICATION_COOLDOWN = 3.0
SIREN_FILE = "siren.mp3"
RECORD_DURATION = 10  # Qeyd olunacaq video müddəti (saniyə)

# Global dəyişənlər
FORBIDDEN_ZONE = []
zone_selected = False
latest_frame = None
frame_lock = threading.Lock()
running = True

last_notification_time = {}
alert_active = False
alert_time = 0

# ============================================================
# SIREN & VIDEO RECORDING FUNCTIONS
# ============================================================
def play_siren():
    try:
        if os.path.exists(SIREN_FILE):
            pygame.mixer.music.load(SIREN_FILE)
            pygame.mixer.music.play()
    except Exception as e:
        print("Siren xətası:", e)

def record_video_clip(duration=10):
    """Xəbərdarlıq anından etibarən 5 saniyəlik video qeyd edir"""
    output_dir = "intrusions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"intrusion_{timestamp}.mp4")

    # Kadr ölçülərini götürürük
    with frame_lock:
        if latest_frame is None:
            return
        h, w = latest_frame.shape[:2]

    # Video yazar obyekt (20 FPS ilə)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, 20.0, (w, h))

    start_time = time.time()
    print(f"\n[VIDEO] Qeydiyyat başladı: {filepath}")

    while time.time() - start_time < duration:
        with frame_lock:
            if latest_frame is not None:
                out.write(latest_frame)
        time.sleep(0.04)  # ~25 FPS

    out.release()
    print(f"[VIDEO] Video uğurla saxlanıldı: {filepath}\n")

# ============================================================
# MOUSE EVENT - ZONA SEÇİMİ
# ============================================================
def draw_zone(event, x, y, flags, param):
    global FORBIDDEN_ZONE, zone_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(FORBIDDEN_ZONE) < 4:
            FORBIDDEN_ZONE.append([x, y])
            print(f"Nöqtə {len(FORBIDDEN_ZONE)} seçildi: ({x}, {y})")

        if len(FORBIDDEN_ZONE) == 4:
            zone_selected = True
            print("\n4 nöqtə seçildi! Təsdiqləmək üçün 'C', sıfırlamaq üçün 'R' sıxın.")

def select_zone_interactive(cap):
    global FORBIDDEN_ZONE, zone_selected

    cv2.namedWindow("ZONA SECIMI")
    cv2.setMouseCallback("ZONA SECIMI", draw_zone)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            frame = cv2.resize(frame, (1280, int(height * scale)), interpolation=cv2.INTER_AREA)

        display_frame = frame.copy()

        for pt in FORBIDDEN_ZONE:
            cv2.circle(display_frame, (pt[0], pt[1]), 5, (0, 0, 255), -1)

        if len(FORBIDDEN_ZONE) > 1:
            pts = np.array(FORBIDDEN_ZONE, np.int32)
            cv2.polylines(display_frame, [pts], isClosed=zone_selected, color=(0, 0, 255), thickness=2)

        cv2.putText(display_frame, f"Secilen noqte: {len(FORBIDDEN_ZONE)}/4", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("ZONA SECIMI", display_frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            FORBIDDEN_ZONE = []
            zone_selected = False
            print("Seçim sıfırlandı.")

        elif key == ord('c') and len(FORBIDDEN_ZONE) == 4:
            cv2.destroyWindow("ZONA SECIMI")
            return np.array(FORBIDDEN_ZONE, np.int32)

# ============================================================
# CAMERA THREAD
# ============================================================
def camera_thread():
    global latest_frame, running
    cap = None

    while running:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                time.sleep(1)
                continue

        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = None
            time.sleep(0.5)
            continue

        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            frame = cv2.resize(frame, (1280, int(height * scale)), interpolation=cv2.INTER_AREA)

        with frame_lock:
            latest_frame = frame

    if cap is not None:
        cap.release()

# ============================================================
# AI THREAD (INTRUSION + SIREN + RECORDING)
# ============================================================
def ai_thread(zone_polygon):
    global running, alert_active, alert_time

    model = YOLO(MODEL_NAME)

    while running:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.005)
                continue
            frame = latest_frame.copy()

        try:
            results = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=CONFIDENCE,
                imgsz=AI_SIZE,
                tracker="bytetrack.yaml",
                verbose=False
            )

            result = results[0]
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)

            now = time.time()

            for box, person_id in zip(boxes, ids):
                person_id = int(person_id)
                x1, y1, x2, y2 = map(int, box)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                is_inside = cv2.pointPolygonTest(zone_polygon, (center_x, center_y), False) >= 0

                if is_inside:
                    last_alert = last_notification_time.get(person_id, 0)

                    if now - last_alert > NOTIFICATION_COOLDOWN:
                        last_notification_time[person_id] = now
                        alert_active = True
                        alert_time = now

                        print(f"\n[ALARM] ID {person_id} zonadadır! {datetime.now().strftime('%H:%M:%S')}")
                        
                        # 1. Sireni çaldırırıq
                        threading.Thread(target=play_siren, daemon=True).start()

                        # 2. 5 saniyəlik video qeydiyyatını başlatırıq
                        threading.Thread(target=record_video_clip, args=(RECORD_DURATION,), daemon=True).start()

        except Exception as e:
            print("AI error:", e)
            time.sleep(0.05)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    temp_cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    SELECTED_ZONE = select_zone_interactive(temp_cap)
    temp_cap.release()

    camera = threading.Thread(target=camera_thread, daemon=True)
    ai = threading.Thread(target=ai_thread, args=(SELECTED_ZONE,), daemon=True)

    camera.start()
    ai.start()

    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.005)
                continue
            frame = latest_frame.copy()

        # Zonanı çək
        cv2.polylines(frame, [SELECTED_ZONE], isClosed=True, color=(0, 0, 255), thickness=3)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [SELECTED_ZONE], (0, 0, 255))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        if alert_active and (time.time() - alert_time < 2.0):
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
            cv2.putText(frame, "SIREN & RECORDING STARTED!", (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        else:
            alert_active = False

        cv2.imshow("UNIVIEW INTRUSION DETECTION", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False
            break

    running = False
    cv2.destroyAllWindows()
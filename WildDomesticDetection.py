from ultralytics import YOLO
import cv2
import subprocess
import threading
import time
import os

# --- Config ---
MODEL_PATH = "/home/asus/Downloads/Detection30epoch.pt"
ALARM_PATH = "/home/asus/Downloads/air-raid-siren.mp3"
CONF_THRESHOLD = 0.6
ALARM_COOLDOWN = 10.0        # seconds between successive wild alerts
DOMESTIC_COOLDOWN = 10.0     # seconds between successive domestic alerts
CAM_INDEX = 0

wild_animals = {
    "tiger", "leopard", "lion", "elephant",
    "bear", "rhinoceros", "wild_boar", "wolf",
    "fox", "crocodile"
}

domestic_animals = {
    "cat", "dog", "cow", "horse", "sheep",
    "goat", "chicken", "duck", "pig", "rabbit"
}

# --- Helper: notify-send wrapper ---
def notify_desktop(title: str, message: str) -> bool:
    notify_cmd = shutil_which("notify-send")
    if not notify_cmd:
        print("notify-send not found; cannot show desktop notification.")
        return False
    try:
        subprocess.run([notify_cmd, title, message], check=True)
        return True
    except Exception as e:
        print("notify-send failed:", e)
        return False

def shutil_which(prog):
    try:
        import shutil
        return shutil.which(prog)
    except Exception:
        return None

# --- Helper: play alarm in background ---
def play_alarm():
    if not os.path.isfile(ALARM_PATH):
        print("Alarm file not found:", ALARM_PATH)
        return
    try:
        # try playsound if available, else use mpg123 as fallback
        try:
            from playsound import playsound
            playsound(ALARM_PATH)
            return
        except Exception:
            subprocess.run(["mpg123", "-q", ALARM_PATH])
    except Exception as e:
        print("Error playing alarm:", e)

# --- Basic checks ---
if not os.path.isfile(MODEL_PATH):
    raise SystemExit(f"Model file not found: {MODEL_PATH}")
if not os.access(MODEL_PATH, os.R_OK):
    raise SystemExit(f"Permission denied for model file: {MODEL_PATH}")

# Load model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise SystemExit(f"Failed to load model: {e}")

# Open camera
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise SystemExit(f"Could not open webcam index {CAM_INDEX}.")

print("Webcam opened. Press 'q' to quit.")

last_wild_alert = 0.0
last_domestic_alert = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed; retrying...")
            time.sleep(0.1)
            continue

        try:
            results = model(frame)
        except Exception as e:
            print("Inference error:", e)
            continue

        annotated = results[0].plot() if len(results) > 0 else frame

        wild_detected = False
        wild_names = []
        domestic_detected = False
        domestic_names = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id] if cls_id < len(model.names) else str(cls_id)
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            name_lower = class_name.lower()
            if name_lower in wild_animals:
                wild_detected = True
                wild_names.append(f"{class_name.upper()} ({conf:.2f})")
                # draw red label for wild
                cv2.putText(annotated, f"{class_name.upper()} {conf:.2f}",
                            (20, 50 + 30 * len(wild_names)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
            elif name_lower in domestic_animals:
                domestic_detected = True
                domestic_names.append(f"{class_name.upper()} ({conf:.2f})")
                # draw green label for domestic
                cv2.putText(annotated, f"{class_name.upper()} {conf:.2f}",
                            (20, 200 + 30 * len(domestic_names)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)
            else:
                # ignore other classes (non-living or unneeded)
                continue

        now = time.time()

        # Wild alerts: notification + alarm (with cooldown)
        if wild_detected and (now - last_wild_alert) > ALARM_COOLDOWN:
            last_wild_alert = now
            msg = " & ".join(wild_names[:3])
            print("WILD ALERT:", msg)
            threading.Thread(target=notify_desktop, args=("⚠️ Wild Animal Alert!", msg), daemon=True).start()
            threading.Thread(target=play_alarm, daemon=True).start()

        # Domestic alerts: only notification (with separate cooldown)
        if domestic_detected and (now - last_domestic_alert) > DOMESTIC_COOLDOWN:
            last_domestic_alert = now
            dmsg = " & ".join(domestic_names[:3])
            print("DOMESTIC ALERT:", dmsg)
            threading.Thread(target=notify_desktop, args=("Domestic Animal Detected", dmsg), daemon=True).start()

        cv2.imshow("Wild vs Domestic Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
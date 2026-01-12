# fast_detection_mp.py
import cv2, time, os, subprocess
import multiprocessing as mp
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "/home/asus/Animal_detection.pt"   # use yolov8n based model for speed
CAM_INDEX = 0
CAM_WIDTH, CAM_HEIGHT = 640, 480   # capture resolution
INFER_SIZE = 320                   # inference square size (try 224 or 160 for more speed)
CONF_THRESHOLD = 0.35
DRAW_EVERY_N = 1                   # draw boxes every N frames (1 = every frame)
USE_HALF = False                   # rarely useful on Pi CPU; leave False
DEVICE = "cpu"                     # 'cpu' for Pi; change only if you have GPU
# ----------------------------------------

# lists for classification coloring (lowercase)
WILD = { "tiger","leopard","lion","elephant","bear","rhinoceros","wild_boar","wolf","fox","crocodile" }
DOMESTIC = { "cat","dog","cow","horse","sheep","goat","chicken","duck","pig","rabbit" }

def inference_worker(frame_q, result_q, stop_event):
    """Runs in separate process: loads model and infers on latest frames."""
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        result_q.put({"error": f"Model load error: {e}"})
        return

    while not stop_event.is_set():
        try:
            frame = frame_q.get(timeout=0.5)  # blocking until a frame arrives
        except Exception:
            continue
        if frame is None:
            break

        # Resize for inference
        small = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))
        try:
            # run inference, return raw boxes; use conf param to reduce noise
            results = model(small, imgsz=INFER_SIZE, device=DEVICE, conf=CONF_THRESHOLD, half=USE_HALF)
        except Exception as e:
            result_q.put({"error": f"Inference error: {e}"})
            continue

        # pack only needed detection info (class id, conf, xyxy coords scaled to small)
        detections = []
        if len(results) > 0:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # xyxy relative to small image:
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], "cpu") else box.xyxy[0].numpy()
                x1, y1, x2, y2 = map(float, xyxy)
                detections.append((cls, conf, (x1, y1, x2, y2)))
        # push minimal result
        result_q.put({"detections": detections, "shape": frame.shape})

    # done
    result_q.put({"stopped": True})

def main():
    # prepare queues and process
    ctx = mp.get_context("spawn")
    frame_q = ctx.Queue(maxsize=1)   # always keep only latest frame
    result_q = ctx.Queue(maxsize=2)
    stop_event = ctx.Event()

    p = ctx.Process(target=inference_worker, args=(frame_q, result_q, stop_event), daemon=True)
    p.start()

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Cannot open camera")
        stop_event.set()
        p.join(timeout=1)
        return

    last_draw = 0
    frame_count = 0
    start_time = time.time()
    current_result = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01); continue

            # send the latest frame (drop old)
            if frame_q.full():
                try:
                    _ = frame_q.get_nowait()
                except:
                    pass
            try:
                frame_q.put_nowait(frame)
            except:
                pass

            # get latest result if available (non-blocking)
            try:
                while True:
                    res = result_q.get_nowait()
                    current_result = res
            except:
                pass

            # draw boxes from current_result
            annotated = frame.copy()
            if current_result and "detections" in current_result:
                dets = current_result["detections"]
                # scale coords from INFER_SIZE -> frame size
                h, w = frame.shape[:2]
                sx = w / INFER_SIZE
                sy = h / INFER_SIZE

                # Draw every N frames to save CPU
                if (frame_count % DRAW_EVERY_N) == 0:
                    y_offset = 20
                    for (cls, conf, (x1,y1,x2,y2)) in dets:
                        try:
                            name = model_names_lookup(cls)  # below function tries to get names from file
                        except:
                            name = str(cls)
                        # choose color
                        lname = name.lower()
                        if lname in WILD:
                            color = (0,0,255)  # red
                        elif lname in DOMESTIC:
                            color = (0,255,0)  # green
                        else:
                            color = (255,255,0)  # yellow for others
                        # scale
                        X1 = int(x1 * sx); Y1 = int(y1 * sy)
                        X2 = int(x2 * sx); Y2 = int(y2 * sy)
                        cv2.rectangle(annotated, (X1,Y1), (X2,Y2), color, 2)
                        label = f"{name} {conf:.2f}"
                        cv2.putText(annotated, label, (X1, max(15,Y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        y_offset += 20

            # display FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed>0 else 0.0
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            cv2.imshow("Fast Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        # push sentinel to stop worker cleanly
        try:
            frame_q.put(None, timeout=1)
        except:
            pass
        p.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()

# Helper to load names quickly from model file if possible
_model_names_cache = None
def model_names_lookup(cls_id):
    global _model_names_cache
    if _model_names_cache is None:
        try:
            # try to load names from the model weights via ultralytics' YOLO for quick mapping
            tmp = YOLO(MODEL_PATH)
            _model_names_cache = tmp.names
        except:
            _model_names_cache = {}
    return _model_names_cache.get(int(cls_id), str(cls_id))

if _name_ == "main":
    main()
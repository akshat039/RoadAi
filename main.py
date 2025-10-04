# main_with_plate_ocr.py
from ultralytics import YOLO
import cv2
import csv
from pathlib import Path
import time
import os
import easyocr

# ------------------- CONFIG -------------------
MODEL_PATH = 'yolov8n.pt'                   # vehicle model (unchanged)
PLATE_MODEL_PATH = 'license_plate_detector.pt'  # your downloaded plate model
VIDEO_PATH = '/home/akshat/VehicleDetectionProject/test_data.mp4'
OUTPUT_VIDEO = 'input_video_out.avi'        # same output video name as before
VEHICLE_CSV = 'vehicle_counts.csv'         # keep previous vehicle counts CSV as-is
PLATE_CSV = 'plates.csv'                   # NEW file for plate texts
CONF_THRESHOLD = 0.3
RESIZE_SCALE = 0.75
TRACKER_CFG = 'bytetrack.yaml'             # keep as before (may be unused if missing)
HEADLESS = True                            # True = no GUI windows
# ---------------------------------------------

# ------------------- Preparations -------------------
# check files existence (models/video); raise helpful errors if missing
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Vehicle model not found: {MODEL_PATH}")
if not Path(PLATE_MODEL_PATH).exists():
    raise FileNotFoundError(f"Plate model not found: {PLATE_MODEL_PATH}")
if not Path(VIDEO_PATH).exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

# load models
model = YOLO(MODEL_PATH)           # vehicle detector
plate_model = YOLO(PLATE_MODEL_PATH)  # plate detector

# easyocr reader
reader = easyocr.Reader(['en'], gpu=False)  # gpu=True if you have GPU support and want speed

# class names and LMV/HMV lists (unchanged logic)
CLASS_NAMES = model.model.names or {i: f'class_{i}' for i in range(model.model.model.nc)}
normalize_class = lambda name: name.lower().replace(" ", "").strip()
LMV_CLASSES = {'car', 'motorbike', 'motorcycle', 'auto', 'autorickshaw', 'scooter'}
HMV_CLASSES = {'bus', 'truck', 'tractor'}
LMV_IDS = {k for k, v in CLASS_NAMES.items() if normalize_class(v) in LMV_CLASSES}
HMV_IDS = {k for k, v in CLASS_NAMES.items() if normalize_class(v) in HMV_CLASSES}

# ------------------- Video IO setup -------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise Exception("Error: Video file open nahi ho raha")

# sizes + fps
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_SCALE)
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_SCALE)
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (frame_w, frame_h)
)
if not out.isOpened():
    raise Exception("Error: VideoWriter create nahi ho raha")

# ------------------- CSV setups -------------------
# Keep vehicle CSV (existing behavior) — append or create fresh? We'll create fresh to match prior runs:
Path(VEHICLE_CSV).unlink(missing_ok=True)
v_csv = open(VEHICLE_CSV, 'w', newline='')
v_writer = csv.writer(v_csv)
v_writer.writerow(['Frame', 'LMV Count', 'HMV Count'])

# New plates CSV
Path(PLATE_CSV).unlink(missing_ok=True)
p_csv = open(PLATE_CSV, 'w', newline='')
p_writer = csv.writer(p_csv)
p_writer.writerow(['Frame', 'Vehicle_Type', 'Plate_Number'])

# ------------------- Helper functions -------------------
def count_vehicles(boxes):
    lmv, hmv = 0, 0
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id in LMV_IDS:
            lmv += 1
        elif cls_id in HMV_IDS:
            hmv += 1
    return lmv, hmv

def draw_boxes(frame, boxes):
    # draw vehicles (exact same labels/colors as before)
    for box in boxes:
        conf = box.conf[0].item() if hasattr(box, 'conf') else 1.0
        if conf < CONF_THRESHOLD:
            continue
        cls_id = int(box.cls[0])
        if cls_id not in (LMV_IDS | HMV_IDS):
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if getattr(box, 'id', None) is not None else None
        vehicle_type = 'LMV' if cls_id in LMV_IDS else 'HMV'
        color = (0, 255, 0) if vehicle_type == 'LMV' else (0, 0, 255)
        label = f"{CLASS_NAMES[cls_id]} | {vehicle_type} | {conf:.2f}"
        if track_id is not None:
            label = f"{CLASS_NAMES[cls_id]} | ID:{track_id} | {vehicle_type} | {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def detect_license_plates_in_crop(crop):
    """
    Run the plate detector on the cropped vehicle image.
    Returns list of plate boxes in crop coordinates: (x1,y1,x2,y2)
    """
    plates = []
    try:
        res = plate_model.predict(crop, conf=0.25, imgsz=640, verbose=False)[0]
        if res and res.boxes is not None:
            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                # ensure box inside crop
                if x2 > x1 and y2 > y1:
                    plates.append((x1, y1, x2, y2))
    except Exception as e:
        # don't crash — return empty plates if model fails on a crop
        # print("Plate detection error:", e)
        return []
    return plates

def ocr_plate_text(plate_img):
    """
    Run EasyOCR on plate image and return best text (string) or empty string.
    Performs small preprocessing: gray, resize, contrast enhancement
    """
    try:
        # convert to gray and slightly resize for better OCR
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        scale = 2 if max(h, w) < 200 else 1
        if scale != 1:
            gray = cv2.resize(gray, (w*scale, h*scale))
        # optional histogram equalization to enhance contrast
        gray = cv2.equalizeHist(gray)
        # EasyOCR expects RGB images but works with gray too; we'll pass original crop for better results
        result = reader.readtext(gray, detail=1)
        # result is list of (bbox, text, conf) for easyocr; but when passing gray, sometimes returns as expected
        # We'll try to pick the longest/highest-confidence text
        if not result:
            # try passing original color image as fallback
            result = reader.readtext(plate_img, detail=1)
        if not result:
            return ""
        # choose the text with highest confidence if available, else longest text
        best_text = ""
        best_conf = -1.0
        for item in result:
            # item format may be (bbox, text, conf)
            if len(item) == 3:
                _, text, conf = item
            elif len(item) == 2:
                text, conf = item
            else:
                text = str(item[1]) if len(item) > 1 else str(item)
                conf = 0.0
            text = text.strip()
            if text == "":
                continue
            try:
                conf_val = float(conf)
            except:
                conf_val = 0.0
            # prefer higher confidence and alphanumeric text
            if conf_val > best_conf or (conf_val == best_conf and len(text) > len(best_text)):
                best_text = text
                best_conf = conf_val
        # basic cleanup: keep only alnum and dash/space
        clean = "".join(ch for ch in best_text if ch.isalnum() or ch in ['-', ' '])
        return clean
    except Exception:
        return ""

# ------------------- MAIN LOOP -------------------
frame_idx = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # resize for processing
    resized = cv2.resize(frame, (frame_w, frame_h))

    # 1) vehicle detection + tracking (same as before)
    try:
        res = model.track(
            resized,
            tracker=TRACKER_CFG,
            persist=True,
            verbose=False,
            conf=CONF_THRESHOLD,
            iou=0.5
        )[0]
    except Exception as e:
        # if tracking fails, try plain predict fallback
        try:
            res = model.predict(resized, conf=CONF_THRESHOLD)[0]
        except Exception:
            res = None

    if res and res.boxes is not None:
        boxes = res.boxes
        draw_boxes(resized, boxes)
        lmv_count, hmv_count = count_vehicles(boxes)
    else:
        boxes = []
        lmv_count, hmv_count = 0, 0

    # 2) For every vehicle box, try to detect plate(s) inside that crop and OCR them
    plate_texts_for_frame = []  # collect (vehicle_type, plate_text) tuples
    for box in boxes:
        conf = box.conf[0].item() if hasattr(box, 'conf') else 1.0
        if conf < CONF_THRESHOLD:
            continue
        cls_id = int(box.cls[0])
        if cls_id not in (LMV_IDS | HMV_IDS):
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # ensure coords inside image
        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(frame_w - 1, x2); y2c = min(frame_h - 1, y2)
        if x2c <= x1c or y2c <= y1c:
            continue

        vehicle_crop = resized[y1c:y2c, x1c:x2c]

        # detect plates in this crop
        plates_in_crop = detect_license_plates_in_crop(vehicle_crop)

        # if plate boxes found, do OCR on each
        if plates_in_crop:
            for (px1, py1, px2, py2) in plates_in_crop:
                # map plate box coords back to resized crop coords (they already are relative to crop)
                # extract plate image (add small padding)
                pad = 3
                px1b = max(0, px1 - pad); py1b = max(0, py1 - pad)
                px2b = min(vehicle_crop.shape[1]-1, px2 + pad); py2b = min(vehicle_crop.shape[0]-1, py2 + pad)
                plate_img = vehicle_crop[py1b:py2b, px1b:px2b]
                if plate_img.size == 0:
                    continue
                plate_text = ocr_plate_text(plate_img)
                vehicle_type = 'LMV' if cls_id in LMV_IDS else 'HMV'
                if plate_text:
                    # annotate on resized frame (draw rectangle in global coords)
                    # compute global coords of plate box
                    gx1 = x1c + px1b; gy1 = y1c + py1b; gx2 = x1c + px2b; gy2 = y1c + py2b
                    cv2.rectangle(resized, (gx1, gy1), (gx2, gy2), (255, 255, 0), 2)
                    cv2.putText(resized, plate_text, (gx1, max(gy1-8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    # record to plates CSV
                    p_writer.writerow([frame_idx, vehicle_type, plate_text])
                    plate_texts_for_frame.append((vehicle_type, plate_text))
                else:
                    # if OCR failed, still log as "PLATE_DETECTED" without text (optional)
                    p_writer.writerow([frame_idx, vehicle_type, ""])
        else:
            # no plate detected inside this vehicle crop -> optional: try whole-frame plate detection fallback
            try:
                # try detect plates on full resized frame (less efficient; keeps coverage)
                global_plates = detect_license_plates_in_crop(resized)
                for (gpx1, gpy1, gpx2, gpy2) in global_plates:
                    plate_img = resized[gpy1:gpy2, gpx1:gpx2]
                    if plate_img.size == 0:
                        continue
                    plate_text = ocr_plate_text(plate_img)
                    vehicle_type = 'LMV' if cls_id in LMV_IDS else 'HMV'
                    if plate_text:
                        cv2.rectangle(resized, (gpx1, gpy1), (gpx2, gpy2), (255, 255, 0), 2)
                        cv2.putText(resized, plate_text, (gpx1, max(gpy1-8, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        p_writer.writerow([frame_idx, vehicle_type, plate_text])
                        plate_texts_for_frame.append((vehicle_type, plate_text))
            except Exception:
                pass

    # write vehicle counts CSV (unchanged behavior)
    v_writer.writerow([frame_idx, lmv_count, hmv_count])

    # Write annotated frame to video
    out.write(resized)

    # optional GUI (off by default)
    if not HEADLESS:
        cv2.imshow("YOLOv8 + Plate OCR", resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_idx += 1

# ------------------- CLEANUP -------------------
cap.release()
out.release()
v_csv.close()
p_csv.close()
# do not call cv2.destroyAllWindows() in headless envs
if not HEADLESS:
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

print(f"✅ Processing complete!")
print(f"Video saved as: {os.path.abspath(OUTPUT_VIDEO)}")
print(f"Vehicle counts CSV: {os.path.abspath(VEHICLE_CSV)}")
print(f"Plates CSV: {os.path.abspath(PLATE_CSV)}")

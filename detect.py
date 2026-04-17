"""
==========================================================
  AI Object Detection using YOLOv8
  Author  : KP
  Project : AI Object Detection (Python)
  Model   : YOLOv8s (Ultralytics - pretrained on COCO)

  ACCURACY IMPROVEMENTS APPLIED:
    1. Upgraded model: yolov8n → yolov8s (higher mAP)
    2. Lower CONF_THRESH: 0.45 → 0.25 (detect more objects)
    3. Higher IOU_THRESH: 0.45 → 0.50 (reduce duplicate boxes)
    4. Higher INPUT_SIZE: 640 → 1280 (better small-object detection)
    5. Test-Time Augmentation (TTA): augment=True (image mode)
    6. Agnostic NMS: merges overlapping boxes across classes
==========================================================

USAGE:
  python detect.py --mode image  --source path/to/image.jpg
  python detect.py --mode webcam
  python detect.py --mode video  --source path/to/video.mp4
  python detect.py --mode image  --source img.jpg --model yolov8m.pt
"""

import argparse
import cv2
import time
from ultralytics import YOLO

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
MODEL_PATH   = "yolov8s.pt"    # ✅ Upgraded from yolov8n → yolov8s (~22 MB)
                                #    Use yolov8m.pt / yolov8l.pt for even more accuracy
CONF_THRESH  = 0.25            # ✅ Lowered from 0.45 → detects more objects
IOU_THRESH   = 0.50            # ✅ Raised from 0.45 → reduces duplicate boxes
INPUT_SIZE   = 1280            # ✅ Higher resolution → better small-object detection
AGNOSTIC_NMS = True            # ✅ Merge overlapping boxes across different classes
BOX_COLOR    = (0, 200, 0)     # Green bounding boxes (BGR)
TEXT_COLOR   = (255, 255, 255) # White label text
FONT         = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────
# HELPER – draw detections on frame
# ──────────────────────────────────────────
def draw_detections(frame, results, class_names):
    """Draw bounding boxes and labels on the given frame."""
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESH:
            continue

        cls_id = int(box.cls[0])
        label  = f"{class_names[cls_id]}  {conf:.0%}"

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        # Label background pill
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), BOX_COLOR, -1)

        # Label text
        cv2.putText(frame, label, (x1 + 3, y1 - 5), FONT, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

    # Object count overlay
    count = len(results[0].boxes)
    cv2.putText(frame, f"Objects: {count}", (10, 28), FONT, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
    return frame


# ──────────────────────────────────────────
# MODE 1 – Static Image
# ──────────────────────────────────────────
def detect_image(model, source):
    print(f"\n[INFO] Running detection on image: {source}")

    # ✅ augment=True enables Test-Time Augmentation (TTA) for better accuracy
    # ✅ imgsz=INPUT_SIZE uses higher resolution for small-object detection
    results = model.predict(
        source=source,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=INPUT_SIZE,
        augment=True,
        agnostic_nms=AGNOSTIC_NMS,
        verbose=False
    )

    frame = cv2.imread(source)
    if frame is None:
        print("[ERROR] Could not load image. Check the path.")
        return

    frame = draw_detections(frame, results, model.names)

    # Print results to terminal
    print(f"[RESULT] {len(results[0].boxes)} object(s) detected:")
    for box in results[0].boxes:
        cls  = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        print(f"   → {cls:<20} confidence: {conf:.2%}")

    output_path = "output_image.jpg"
    cv2.imwrite(output_path, frame)
    print(f"\n[SAVED] Result saved to '{output_path}'")

    cv2.imshow("AI Object Detection – Image", frame)
    print("[INFO] Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────
# MODE 2 – Webcam (Real-time)
# ──────────────────────────────────────────
def detect_webcam(model):
    print("\n[INFO] Starting webcam... Press 'q' to quit, 's' to save a screenshot.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Make sure it is connected.")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # ✅ augment=False for webcam (too slow for real-time)
        # ✅ imgsz=640 kept for webcam to maintain live FPS
        results = model.predict(
            source=frame,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            imgsz=640,
            augment=False,
            agnostic_nms=AGNOSTIC_NMS,
            stream=True,
            verbose=False
        )

        for r in results:
            frame = draw_detections(frame, [r], model.names)

        # FPS counter
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 55), FONT, 0.7, (255, 180, 0), 2)

        cv2.imshow("AI Object Detection – Webcam (q = quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshot_{ts}.jpg", frame)
            print(f"[SAVED] Screenshot saved as 'screenshot_{ts}.jpg'")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam session ended.")


# ──────────────────────────────────────────
# MODE 3 – Video File
# ──────────────────────────────────────────
def detect_video(model, source):
    print(f"\n[INFO] Running detection on video: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Cannot open video file.")
        return

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out = cv2.VideoWriter("output_video.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ augment=False for video (performance); raise to True for offline processing
        results = model.predict(
            source=frame,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            imgsz=INPUT_SIZE,
            augment=False,
            agnostic_nms=AGNOSTIC_NMS,
            verbose=False
        )
        frame = draw_detections(frame, results, model.names)
        out.write(frame)
        frame_count += 1

        cv2.imshow("AI Object Detection – Video (q = quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Processed {frame_count} frames. Saved to 'output_video.mp4'")


# ──────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI Object Detection using YOLOv8")
    parser.add_argument("--mode",   choices=["image", "webcam", "video"],
                        default="webcam", help="Detection mode (default: webcam)")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to image or video file (required for image/video mode)")
    parser.add_argument("--model",  type=str, default=MODEL_PATH,
                        help="YOLOv8 model file (default: yolov8s.pt)")
    args = parser.parse_args()

    # Load model (downloads automatically if not present)
    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"[INFO] Model loaded. Classes: {len(model.names)}")

    if args.mode == "image":
        if not args.source:
            print("[ERROR] Please provide --source path/to/image.jpg")
        else:
            detect_image(model, args.source)

    elif args.mode == "webcam":
        detect_webcam(model)

    elif args.mode == "video":
        if not args.source:
            print("[ERROR] Please provide --source path/to/video.mp4")
        else:
            detect_video(model, args.source)


if __name__ == "__main__":
    main()

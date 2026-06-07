## AI Object Detection using YOLOv8
**Language:** Python 3.8+  |  **Model:** YOLOv8n (COCO – 80 classes)

---

### ⚡ Quick Setup (3 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt 

# 2. Run on your webcam 
python detect.py --mode webcam 
 
# 3. Run on an image
python detect.py --mode image --source your_image.jpg
```

The YOLOv8 model (~6 MB) downloads **automatically** on first run.

---

### Modes

| Mode     | Command                                           |
|----------|---------------------------------------------------|
| Webcam   | `python detect.py --mode webcam`                  |
| Image    | `python detect.py --mode image --source file.jpg` |
| Video    | `python detect.py --mode video --source file.mp4` |

---

### Keyboard Shortcuts (Webcam)
- **q** → Quit
- **s** → Save screenshot

---

### What it detects
80 common objects including: person, car, bicycle, dog, cat, chair, phone, laptop, bottle, and more.

---

### Outputs
- `output_image.jpg` – annotated image result
- `output_video.mp4` – annotated video result
- `screenshot_YYYYMMDD_HHMMSS.jpg` – webcam screenshots

---

### System Requirements
- Python 3.8 or higher
- A webcam (for webcam mode)
- 4 GB RAM minimum (8 GB recommended)
- CPU is fine; GPU (CUDA) will run faster

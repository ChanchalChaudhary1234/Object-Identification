# FruitVision AI — Streamlit UI

This UI is based on `Fruits_identifying_and_count_detection.ipynb`.

## What the notebook currently does
- Loads `YOLO("yolo26n.pt")`, a pretrained Ultralytics model.
- Filters detections to COCO class IDs `[46, 47, 49]` (banana, apple, orange).
- Counts fruits in an image and draws bounding boxes.
- Processes uploaded video frame-by-frame and writes an annotated MP4.
- Reports maximum simultaneous counts per fruit class in the video.

## Run locally

1. Install Python 3.10 or 3.11 (recommended).
2. Extract this folder.
3. Open a terminal in the folder and run:

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On the first run, Ultralytics may download `yolo26n.pt`, so an internet connection is needed.

## Notes
- The UI currently detects only apples, bananas and oranges because that is what the notebook filters.
- `Camera Snapshot` takes a single browser camera image; it is not continuous live video tracking.
- Video counts are maximum simultaneous counts in a frame, not unique fruit identities tracked across the whole clip.
- To use a custom-trained model later, change the model path in `load_model()` and update `FRUIT_CLASSES` according to the custom model's class names.

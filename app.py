import os
import tempfile
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="FruitVision AI | Smart Fruit Detection",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

FRUIT_CLASSES = [46, 47, 49]  # COCO IDs: banana, apple, orange
FRUIT_INFO = {
    "apple": {"label": "Apples", "emoji": "🍎", "color": "#ef4444", "description": "Crisp, round fruit"},
    "banana": {"label": "Bananas", "emoji": "🍌", "color": "#eab308", "description": "Yellow, curved fruit"},
    "orange": {"label": "Oranges", "emoji": "🍊", "color": "#f97316", "description": "Citrus fruit"},
}
FRUIT_ORDER = ["apple", "banana", "orange"]


@st.cache_resource(show_spinner="Loading the YOLO detection model…")
def load_model():
    return YOLO("yolo26n.pt")


def count_detections(result):
    counts = Counter({fruit: 0 for fruit in FRUIT_ORDER})
    confidence_values = defaultdict(list)
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            fruit = result.names[class_id].lower()
            if fruit in FRUIT_INFO:
                counts[fruit] += 1
                confidence_values[fruit].append(float(box.conf.item()))
    return counts, confidence_values


def rows_for_counts(counts, confidence_values=None):
    rows = []
    for fruit in FRUIT_ORDER:
        values = confidence_values.get(fruit, []) if confidence_values else []
        rows.append({
            "Fruit": f"{FRUIT_INFO[fruit]['emoji']}  {FRUIT_INFO[fruit]['label']}",
            "Detected count": int(counts.get(fruit, 0)),
            "Average confidence": f"{np.mean(values) * 100:.1f}%" if values else "—",
            "Detection status": "Detected" if counts.get(fruit, 0) else "Not found",
        })
    return rows


def detect_image(model, image, confidence):
    image_array = np.array(image.convert("RGB"))
    result = model.predict(source=image_array, conf=confidence, classes=FRUIT_CLASSES, verbose=False)[0]
    annotated_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    counts, confidence_values = count_detections(result)
    return annotated_rgb, counts, confidence_values


def render_fruit_summary(counts, confidence_values=None):
    total = sum(counts.values())
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f'<div class="metric-card total"><div class="metric-label">TOTAL FRUITS</div><div class="metric-number">{total}</div><div class="metric-foot">Objects detected</div></div>', unsafe_allow_html=True)
    for col, fruit in zip(cols[1:], FRUIT_ORDER):
        info = FRUIT_INFO[fruit]
        n = int(counts.get(fruit, 0))
        values = confidence_values.get(fruit, []) if confidence_values else []
        avg = f"{np.mean(values) * 100:.0f}% avg. confidence" if values else "No objects found"
        with col:
            st.markdown(f'<div class="metric-card"><div class="fruit-icon">{info["emoji"]}</div><div class="metric-label">{info["label"].upper()}</div><div class="metric-number">{n}</div><div class="metric-foot">{avg}</div></div>', unsafe_allow_html=True)


def render_count_table(counts, confidence_values=None):
    st.markdown("#### Fruit-wise detection report")
    st.dataframe(pd.DataFrame(rows_for_counts(counts, confidence_values)), hide_index=True, use_container_width=True)


def inject_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#172033; --muted:#667085; --purple:#6d4aff; --blue:#3b82f6; }
    .stApp { background: linear-gradient(180deg,#f7f8ff 0%,#f5f7fb 50%,#f9fafc 100%); color:var(--ink); }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#17152f 0%,#252052 100%); }
    [data-testid="stSidebar"] * { color:#f5f4ff; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding:7px 10px; border-radius:10px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#ffffff18; }
    .brand { display:flex; align-items:center; gap:10px; margin:4px 0 2px; }
    .brand-mark { width:45px;height:45px;border-radius:14px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#a78bfa,#60a5fa);font-size:25px;box-shadow:0 8px 24px #6d4aff44; }
    .brand-name { font-family:Manrope,sans-serif;font-weight:800;font-size:22px;letter-spacing:-.8px; }
    .brand-sub { font-size:12px;color:#c4c2e8; }
    .hero { position:relative; overflow:hidden; padding:30px 32px; border-radius:24px; color:white; background:linear-gradient(115deg,#211b50 0%,#4934a6 52%,#3574d4 100%); margin:2px 0 24px; box-shadow:0 16px 40px #342b7c24; }
    .hero:after { content:'🍎'; position:absolute; right:50px; top:5px; font-size:100px; opacity:.14; transform:rotate(-16deg); }
    .eyebrow { font-size:11px; letter-spacing:2px; text-transform:uppercase; font-weight:700; color:#d8d2ff; }
    .hero h1 { font-family:Manrope,sans-serif; font-size:clamp(30px,4vw,43px); line-height:1.12; letter-spacing:-1.5px; margin:9px 0 10px; color:white; }
    .hero p { max-width:680px; margin:0; color:#e6e7ff; font-size:15px; line-height:1.65; }
    .pill-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:19px; }
    .pill { display:inline-block; padding:7px 11px; border-radius:99px; border:1px solid #ffffff38; background:#ffffff14; color:#fff; font-size:12px; font-weight:600; }
    .section-kicker { color:#6d4aff; text-transform:uppercase; font-size:11px; font-weight:800; letter-spacing:1.7px; margin-bottom:4px; }
    .section-title { font-family:Manrope,sans-serif; font-size:25px; font-weight:800; letter-spacing:-.7px; color:#172033; margin:0 0 5px; }
    .section-desc { color:#667085; margin:0 0 17px; font-size:14px; }
    .metric-card { background:#fff; border:1px solid #e8eaf4; border-radius:18px; padding:17px 18px 15px; min-height:145px; box-shadow:0 6px 20px #1b24500a; }
    .metric-card.total { background:linear-gradient(140deg,#6d4aff,#4d66df); border-color:transparent; color:#fff; }
    .metric-label { color:#7a8195; font-size:10px; font-weight:800; letter-spacing:1.1px; }
    .total .metric-label,.total .metric-foot { color:#e5e0ff; }
    .metric-number { font-family:Manrope,sans-serif; font-size:36px; font-weight:800; line-height:1.3; color:#1d2340; }
    .total .metric-number { color:white; }
    .metric-foot { font-size:11px; color:#9298a8; margin-top:2px; }
    .fruit-icon { font-size:23px; line-height:1.2; margin-bottom:8px; }
    .feature-card,.step-card,.info-card { background:white; border:1px solid #e8eaf4; border-radius:17px; padding:18px; height:100%; box-shadow:0 6px 20px #1b245008; }
    .feature-icon { width:42px;height:42px;display:flex;align-items:center;justify-content:center;border-radius:13px;background:#f0edff;font-size:22px;margin-bottom:13px; }
    .feature-title { font-family:Manrope,sans-serif;font-size:15px;font-weight:800;color:#252844;margin-bottom:5px; }
    .feature-text { font-size:13px;line-height:1.55;color:#737b8f; }
    .step-number { width:30px;height:30px;border-radius:10px;display:flex;align-items:center;justify-content:center;background:#ede9ff;color:#6545e8;font-weight:800;margin-bottom:13px; }
    .result-banner { background:#eefff5;border:1px solid #c7f0d8;border-radius:14px;padding:13px 16px;color:#17663b;font-size:14px;margin:12px 0 18px; }
    .note-box { background:#f0edff;border:1px solid #ded7ff;border-radius:14px;padding:14px 16px;color:#45398c;font-size:13px;line-height:1.6; }
    .footer { text-align:center;color:#9aa0b0;font-size:11px;padding:24px 0 6px; }
    div.stButton > button, div.stDownloadButton > button { border-radius:11px; font-weight:700; min-height:43px; }
    div[data-testid="stFileUploader"] { background:#fff;border:1px dashed #c9c6e7;border-radius:15px;padding:10px; }
    [data-testid="stMetric"] { background:white;border:1px solid #e8eaf4;border-radius:15px;padding:14px; }
    </style>
    """, unsafe_allow_html=True)


def hero():
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Computer Vision · Object Detection</div>
      <h1>Meet your smart fruit counter.</h1>
      <p>Turn an image or video into a clear fruit detection report. Visualize every detected object, review fruit-wise counts, and export your results in just a few clicks.</p>
      <div class="pill-row"><span class="pill">⚡ YOLO-powered</span><span class="pill">🍎 Apple detection</span><span class="pill">🍌 Banana detection</span><span class="pill">🍊 Orange detection</span></div>
    </div>
    """, unsafe_allow_html=True)


def section_heading(kicker, title, description):
    st.markdown(f'<div class="section-kicker">{kicker}</div><div class="section-title">{title}</div><p class="section-desc">{description}</p>', unsafe_allow_html=True)


inject_styles()
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">🍎</div><div><div class="brand-name">FruitVision AI</div><div class="brand-sub">Smart Detection Studio</div></div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("YOUR WORKSPACE", ["Overview", "Image Detection", "Video Detection", "Camera Snapshot", "How to Use", "About Project"], label_visibility="visible")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**DETECTION SETTINGS**")
    confidence = st.slider("Confidence threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05, help="Higher values show fewer, more confident detections. Lower values may detect more objects but can add false positives.")
    st.caption(f"Current threshold: **{confidence:.0%}**")
    st.divider()
    st.markdown("<span style='font-size:12px;color:#c4c2e8'>MODEL STATUS</span>", unsafe_allow_html=True)
    st.markdown("🟢 YOLO26n · Pretrained")
    st.markdown("<span style='font-size:12px;color:#c4c2e8'>Classes: Apple · Banana · Orange</span>", unsafe_allow_html=True)

if page == "Overview":
    hero()
    section_heading("Your workflow", "From fruit detection to actionable insights", "Choose an input, run detection, then present a clear and downloadable report.")
    cols = st.columns(3, gap="medium")
    cards = [
        ("📷", "Image detection", "Upload a fruit photo and compare the original image with AI annotations."),
        ("🎬", "Video analysis", "Process a short clip frame by frame and export the annotated video."),
        ("📊", "Readable results", "Review total fruit count, class-wise counts and average confidence."),
    ]
    for col, (icon, title, desc) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div><div class="feature-text">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    section_heading("Get started", "Run your first detection", "For the smoothest live demo, start with a clear image containing apples, bananas or oranges.")
    steps = st.columns(4, gap="small")
    for col, num, title, desc in zip(steps, ["01", "02", "03", "04"], ["Choose a mode", "Upload your file", "Run detection", "Present results"], ["Image, video or camera snapshot.", "Use a JPG, PNG or supported video.", "Adjust the threshold if needed.", "Show counts and download output."]):
        with col:
            st.markdown(f'<div class="step-card"><div class="step-number">{num}</div><div class="feature-title">{title}</div><div class="feature-text">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    # st.markdown('<div class="note-box"><b>Demo tip:</b> Keep the fruit visible, avoid heavy blur, and use good lighting. This version detects apples, bananas and oranges using a pretrained model; it is not trained on the separate 30-category fruit dataset.</div>', unsafe_allow_html=True)

elif page == "Image Detection":
    hero()
    section_heading("Image lab", "Fruit image analysis", "Upload one image to detect fruit, visualize bounding boxes and generate a class-wise count report.")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"], key="image_upload", help="Supported formats: JPG, JPEG, PNG and WEBP.")
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.markdown(f"<div class='note-box'>📄 <b>Selected file:</b> {uploaded.name} &nbsp; · &nbsp; <b>Image size:</b> {image.width} × {image.height}px &nbsp; · &nbsp; <b>Threshold:</b> {confidence:.0%}</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✨  Detect fruits in this image", type="primary", use_container_width=True):
                with st.spinner("Analysing image and preparing your report…"):
                    annotated, counts, confs = detect_image(load_model(), image, confidence)
                st.session_state["image_result"] = {"file": uploaded.name, "annotated": annotated, "counts": counts, "confs": confs}
            saved = st.session_state.get("image_result")
            if saved and saved.get("file") == uploaded.name:
                st.markdown('<div class="result-banner">✓ Analysis complete — your fruit detection report is ready.</div>', unsafe_allow_html=True)
                left, right = st.columns(2, gap="large")
                with left:
                    st.markdown("**Original image**")
                    st.image(image, use_container_width=True)
                with right:
                    st.markdown("**AI detection output**")
                    st.image(saved["annotated"], use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                render_fruit_summary(saved["counts"], saved["confs"])
                render_count_table(saved["counts"], saved["confs"])
                total = sum(saved["counts"].values())
                detected = [FRUIT_INFO[f]["label"] for f in FRUIT_ORDER if saved["counts"].get(f, 0)]
                if total:
                    st.success(f"Report summary: detected **{total} fruit object(s)** — " + ", ".join(detected) + ".")
                else:
                    st.info("No configured fruit classes were detected. Try a clearer image or lower the confidence threshold.")
                ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(saved["annotated"], cv2.COLOR_RGB2BGR))
                if ok:
                    st.download_button("⬇️  Download annotated image", encoded.tobytes(), "fruitvision_detection_result.jpg", "image/jpeg", type="primary")
        except Exception as exc:
            st.error(f"Could not read or process this image: {exc}")
    else:
        st.markdown('<div class="info-card"><div class="feature-icon">🖼️</div><div class="feature-title">Your results will appear here</div><div class="feature-text">Upload an image to see the original photo, annotated detection output, total count, fruit-wise breakdown and download option.</div></div>', unsafe_allow_html=True)

elif page == "Video Detection":
    hero()
    section_heading("Video lab", "Frame-by-frame fruit analysis", "Upload a short video and create an annotated MP4. The summary reports the maximum simultaneous count seen in a single frame.")
    uploaded_video = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"], key="video_upload", help="For a faster demo, start with a short clip (around 10–20 seconds).")
    st.markdown('<div class="note-box">⏱️ Video processing can take time because each frame is analysed. The video summary is <b>not a unique-fruit tracker</b>; the same fruit may appear in many frames.</div>', unsafe_allow_html=True)
    if uploaded_video and st.button("🎬  Process video", type="primary", use_container_width=True):
        input_path = None
        output_path = None
        try:
            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                temp.write(uploaded_video.getbuffer())
                input_path = temp.name
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError("The uploaded video could not be opened. Try a different video file.")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            output_path = out.name
            out.close()
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError("Could not create the output video. Try another clip format.")
            max_counts = Counter({f: 0 for f in FRUIT_ORDER})
            all_frame_counts = Counter({f: 0 for f in FRUIT_ORDER})
            progress = st.progress(0)
            status = st.empty()
            frame_number = 0
            model = load_model()
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    result = model.predict(source=frame, conf=confidence, classes=FRUIT_CLASSES, verbose=False)[0]
                    current, _ = count_detections(result)
                    for fruit in FRUIT_ORDER:
                        max_counts[fruit] = max(max_counts[fruit], current[fruit])
                        all_frame_counts[fruit] += current[fruit]
                    writer.write(result.plot())
                    frame_number += 1
                    if total_frames > 0:
                        progress.progress(min(frame_number / total_frames, 1.0))
                    status.caption(f"Processed {frame_number:,}" + (f" of {total_frames:,} frames" if total_frames > 0 else " frames"))
            finally:
                cap.release()
                writer.release()
            with open(output_path, "rb") as f:
                video_bytes = f.read()
            st.markdown('<div class="result-banner">✓ Video analysis complete — annotated video and summary are ready.</div>', unsafe_allow_html=True)
            render_fruit_summary(max_counts)
            render_count_table(max_counts)
            st.caption("Counts above show the maximum number of each fruit visible at once in any single frame. They do not represent unique fruit identities across the full clip.")
            st.markdown("#### Processed video preview")
            st.video(video_bytes)
            st.download_button("⬇️  Download annotated video", video_bytes, "fruitvision_processed_video.mp4", "video/mp4", type="primary")
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
        finally:
            for path in [input_path, output_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

elif page == "Camera Snapshot":
    hero()
    section_heading("Camera studio", "Capture and detect", "Take a snapshot with your device camera, then run the same fruit analysis used for uploaded images.")
    camera_image = st.camera_input("Take a fruit photo")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        if st.button("📸  Analyse camera snapshot", type="primary", use_container_width=True):
            with st.spinner("Detecting fruit in your snapshot…"):
                annotated, counts, confs = detect_image(load_model(), image, confidence)
            st.session_state["camera_result"] = {"annotated": annotated, "counts": counts, "confs": confs}
        saved = st.session_state.get("camera_result")
        if saved:
            left, right = st.columns(2, gap="large")
            with left:
                st.markdown("**Captured snapshot**")
                st.image(image, use_container_width=True)
            with right:
                st.markdown("**Detection output**")
                st.image(saved["annotated"], use_container_width=True)
            render_fruit_summary(saved["counts"], saved["confs"])
            render_count_table(saved["counts"], saved["confs"])
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(saved["annotated"], cv2.COLOR_RGB2BGR))
            if ok:
                st.download_button("⬇️  Download snapshot result", encoded.tobytes(), "fruitvision_camera_result.jpg", "image/jpeg")
    else:
        st.markdown('<div class="info-card"><div class="feature-icon">📷</div><div class="feature-title">Camera ready when you are</div><div class="feature-text">Allow camera access in your browser, take a clear photo, and select Analyse camera snapshot.</div></div>', unsafe_allow_html=True)

elif page == "How to Use":
    hero()
    section_heading("Quick guide", "How to use FruitVision AI", "Follow these steps during your project demo or presentation.")
    guide = [
        ("01", "Choose a detection mode", "Use Image Detection for a photo, Video Detection for a short clip, or Camera Snapshot to take a picture directly."),
        ("02", "Upload or capture", "Choose a supported file. For best results, use clear, well-lit fruit images where objects are not heavily hidden."),
        ("03", "Set the confidence threshold", "Start at 25%. Raise it if you see false detections; lower it if visible fruit is being missed. Compare results on the same image."),
        ("04", "Run the analysis", "Click the relevant detection button and wait for the result. Video processing may take longer than image processing."),
        ("05", "Explain the results", "The annotated image shows boxes and labels. The report lists total fruit objects and the count for apples, bananas and oranges."),
        ("06", "Export your output", "Use the download button to save the annotated image or processed video for your report or presentation."),
    ]
    for i in range(0, len(guide), 2):
        cols = st.columns(2, gap="medium")
        for col, (num, title, desc) in zip(cols, guide[i:i+2]):
            with col:
                st.markdown(f'<div class="step-card"><div class="step-number">{num}</div><div class="feature-title">{title}</div><div class="feature-text">{desc}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Tips for a stronger demo")
    st.markdown("- Use a high-resolution image with good lighting and visible fruit.\n- Start with the default 25% confidence threshold.\n- Show the annotated output beside the original image.\n- Explain that confidence is the model's score, not a guarantee of correctness.\n- For video, describe the count as the maximum simultaneous count per frame, not unique fruit tracking.")
    st.markdown('<div class="note-box"><b>Current model scope:</b> FruitVision AI currently filters a pretrained COCO model to detect apples, bananas and oranges. It does not yet use custom-trained weights from the 30-category fruit and vegetable dataset.</div>', unsafe_allow_html=True)

elif page == "About Project":
    hero()
    section_heading("Project details", "FruitVision AI", "A simple computer-vision application for fruit object detection and counting.")
    cols = st.columns(3)
    for col, icon, title, desc in zip(cols, ["🧠", "🖥️", "📈"], ["Detection engine", "User interface", "Reporting"], ["Ultralytics YOLO26n pretrained model.", "Streamlit interface with image, video and camera workflows.", "Fruit-wise object counts, average image confidence and downloadable annotations."]):
        with col:
            st.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div><div class="feature-text">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Current scope and limitations")
    st.markdown("- Supported classes in this interface: **apple, banana and orange**.\n- Image counts represent detected objects in that image.\n- Video counts represent the maximum simultaneous count in a frame; this app does not track unique fruit identities across time.\n- The pretrained model may miss small, occluded, unusual or poorly lit fruit.\n- The separate 30-category dataset is not automatically used by this app; custom training and a trained weights file would be needed to use it.")

st.markdown('<div class="footer">FRUITVISION AI · Smart Fruit Detection Studio · Built with Streamlit and YOLO</div>', unsafe_allow_html=True)

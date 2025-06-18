import streamlit as st
import cv2
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageOps

st.set_page_config(page_title="🤟 ASL Hand‑Sign Detector", page_icon="🤟")

st.title("🤟 Real‑time ASL Detector (YOLO‑v8)")

# -----------------------------------------------------------------------------
# Sidebar – settings
# -----------------------------------------------------------------------------
settings = st.sidebar.container()
settings.header("⚙️  Settings")

# 1️⃣  Model weights
weights_path = settings.text_input(
    "Weights file (.pt)", value="models/best_mixed.pt", help="Path to YOLO‑v8 weight file"
)

# 2️⃣  Confidence slider
conf = settings.slider("Confidence threshold", 0.01, 1.0, 0.25, 0.01)

# 3️⃣  Input source
source_mode = settings.radio("Input source", ("Upload Image", "Webcam"))

# 4️⃣  Camera ID (builtin cam is usually 0 or 1 on macOS)
cam_id = settings.number_input("Camera ID", 0, 5, value=1, step=1)

# -----------------------------------------------------------------------------
# Load YOLO model (cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_model(path):
    return YOLO(str(Path(path)))

try:
    model = load_model(weights_path)
    settings.success("✅  Weights loaded")
except Exception as e:
    settings.error(f"❌ Failed to load weights: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Helper to render predictions
# -----------------------------------------------------------------------------
def render(pred, caption=None):
    img_bgr = pred.plot(line_width=4, font_size=1.0)
    st.image(img_bgr[:, :, ::-1], caption=caption, use_container_width=True)
    with st.expander("🔬 Raw detections"):
        st.write(pred.boxes)

# -----------------------------------------------------------------------------
# Upload‑image workflow
# -----------------------------------------------------------------------------
if source_mode == "Upload Image":
    file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if file:
        img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
        pred = model.predict(img, conf=conf, verbose=False)[0]
        st.write(f"Detections: {len(pred)}")
        render(pred, caption="Uploaded image")

# -----------------------------------------------------------------------------
# Webcam workflow (uses session_state to avoid duplicate widgets)
# -----------------------------------------------------------------------------
else:
    start_cam = settings.checkbox("📷 Start webcam", key="start_cam")

    if start_cam:
        cap = cv2.VideoCapture(int(cam_id))
        if not cap.isOpened():
            st.error(f"Cannot open camera {cam_id}")
        else:
            frame_placeholder = st.empty()
            st.info("Click the sidebar switch again to stop the webcam.")

            while st.session_state.get("start_cam", False):
                ok, frame = cap.read()
                if not ok:
                    st.error("Failed to grab frame"); break

                pred = model.predict(frame, conf=conf, verbose=False)[0]
                frame_placeholder.image(pred.plot(line_width=4)[:, :, ::-1], use_container_width=True)

            cap.release()
            frame_placeholder.empty()

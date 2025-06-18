ASL Project – Real‑Time Hand‑Sign Detection

This repository contains an end‑to‑end pipeline for American Sign Language (ASL) hand‑sign detection powered by Ultralytics YOLO v8. It covers every stage of the workflow: semi‑automatic data annotation, dataset management in Roboflow, model fine‑tuning, and a Streamlit web application for real‑time inference with a laptop webcam.

⸻

Project Contents

Folder / File	Purpose
notebooks/	Jupyter notebooks for data annotation (data_annotation.ipynb) and model training (ASL_roboflow_model_training*.ipynb).
models/best_mixed.pt	Fine‑tuned YOLO v8 weights trained on mixed‑background images (default model used by the app).
models/best.pt	Alternative weights trained on a clean‑background dataset.
app/streamlit_app.py	Streamlit web application: upload images or use webcam for live detection.
live_testing/	Example scripts for quick CLI or webcam tests.
data/	Optional sample images.
test.py	Minimal Python script that runs a single image through the model from the command line.


⸻

How the Pipeline Works
	1.	Data Annotation
	•	The notebook data_annotation.ipynb uses Grounding‑DINO to create bounding boxes from a text prompt (“hand”), converts them to YOLO format, and uploads the images plus labels to a Roboflow project.
	2.	Dataset Management
	•	Each upload becomes a dataset version inside Roboflow.
	•	Training notebooks download a frozen YOLO‑formatted ZIP directly from Roboflow via the API.
	3.	Model Training
	•	The notebooks fine‑tune the pre‑trained yolov8s.pt backbone for 50 epochs at an image size of 640 pixels.
	•	Two datasets are provided: clean and mixed background. The best mAP checkpoint for each run is exported as a .pt file.
	4.	Real‑Time Inference
	•	app/streamlit_app.py loads models/best_mixed.pt by default.
	•	In Upload mode you can drag‑and‑drop a JPEG/PNG and see detections drawn on the image.
	•	In Webcam mode the script opens the laptop camera, performs inference frame‑by‑frame, and displays the annotated stream.

⸻

Quick Start

The instructions below assume macOS/Linux with Python 3.10+ installed.

# Clone repository
git clone https://github.com/your‑handle/ASL_project.git
cd ASL_project

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install ultralytics opencv‑python‑headless streamlit pillow

# Run the Streamlit app
streamlit run app/streamlit_app.py

Open http://localhost:8501 in your browser.
	•	Upload an image or toggle Start webcam in the sidebar to see live detections.

⸻

Command‑Line Tests

Run inference on a single image:

python test.py path/to/image.jpg --weights models/best_mixed.pt

Use the Ultralytics CLI:

yolo predict model=models/best_mixed.pt source=path/to/image.jpg conf=0.25


⸻

Repository Structure

ASL_project/
├── .venv/                 # optional virtual‑env (ignored by Git)
├── app/
│   └── streamlit_app.py   # Streamlit web UI
├── models/
│   ├── best_mixed.pt      # default model
│   └── best.pt            # alternative model
├── notebooks/
│   ├── data_annotation.ipynb
│   └── ASL_roboflow_model_training*.ipynb
├── data/                  # sample images (optional)
├── live_testing/          # quick scripts
├── test.py                # CLI image test
└── README.md


⸻

⸻

Acknowledgements
	•	Ultralytics YOLO v8 for the detection framework.
	•	Grounding‑DINO for prompt‑based bounding‑box generation.
	•	Roboflow for dataset hosting and augmentation.
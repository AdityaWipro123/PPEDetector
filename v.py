# import streamlit as st
# from PIL import Image
# import numpy as np
# import cv2
# from ultralytics import YOLO

# st.title("Object Detection")

# st.write("Upload an image and YOLOv8 will detect objects it knows (person, car, etc.)")

# # ---------------- IMAGE UPLOAD ----------------
# uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     # Convert PIL image to numpy array
#     image = Image.open(uploaded_file).convert("RGB")
#     img = np.array(image)

#     # Convert RGB → BGR for OpenCV
#     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#     # ---------------- LOAD YOLO MODEL ----------------
#     @st.cache_resource
#     def load_model():
#         return YOLO("yolov8n.pt")  # base pretrained YOLOv8

#     model = load_model()

#     # ---------------- RUN DETECTION ----------------
#     results = model(img_bgr)[0]  # get first result

#     # ---------------- DRAW DETECTIONS ----------------
#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         conf = float(box.conf)
#         cls = int(box.cls)
#         label = f"{model.names[cls]} {conf:.2f}"

#         # Draw rectangle and label
#         cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(img_bgr, label, (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     # Convert BGR → RGB before displaying
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#     st.image(img_rgb, caption="YOLOv8 Detection Result", use_column_width=True)


import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import os

st.title("Object Detection")
st.write("YOLOv8 will detect objects. Default image runs on app load.")

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRIAL_IMAGES_DIR = os.path.join(BASE_DIR, "TrialImages")

# ---------------- LOAD YOLO MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # base pretrained YOLOv8

model = load_model()

# ---------------- HELPER FUNCTION ----------------
def detect_and_display(img_pil, caption="YOLOv8 Detection Result"):
    img = np.array(img_pil)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = model(img_bgr)[0]  # YOLO detection

    # Draw boxes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        cls = int(box.cls)
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption=caption, use_column_width=True)

# ---------------- SELECT IMAGE ----------------
uploaded_file = st.file_uploader("Upload a new image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # User uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    detect_and_display(image, caption="YOLOv8 Detection on Uploaded Image")
else:
    # Load default image from TrialImages
    default_images = [f for f in os.listdir(TRIAL_IMAGES_DIR)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if default_images:
        default_image_path = os.path.join(TRIAL_IMAGES_DIR, default_images[0])
        image = Image.open(default_image_path).convert("RGB")
        st.subheader("Default Image Detection")
        detect_and_display(image, caption=f"YOLOv8 Detection on {default_images[0]}")
    else:
        st.warning("No images found in TrialImages folder")

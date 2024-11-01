import os
import cv2
import torch
import datetime
import numpy as np
import ailia
import streamlit as st
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLOv10
from insightface.face_model import recognize_from_video, recognize_from_image
from torchvision import transforms
from PIL import Image
from collections import defaultdict, deque
import random
from scipy.spatial.distance import cosine
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define global variables for tracking
perform_recognition = False

# Transform function for pre-processing images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Initialize the model
def initialize_model():
    model_path = "./weights/yolov10x.pt"
    if not os.path.exists(model_path):
        logger.error(f"Model weights not found at {model_path}")
        raise FileNotFoundError("Model weights file not found")

    model = YOLOv10(model_path)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    logger.info(f"Using {device} as processing device")
    return model

# Load class names for object detection
def load_class_names():
    classes_path = "./configs/coco.names"
    if not os.path.exists(classes_path):
        logger.error(f"Class names file not found at {classes_path}")
        raise FileNotFoundError("Class names file not found")
    
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")
    return class_names

# Process video frame
def process_frame(frame, model, tracker, class_names, colors):
    results = model(frame, verbose=False)[0]
    detections = []
    for det in results.boxes:
        label, confidence, bbox = det.cls, det.conf, det.xyxy[0]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if confidence < 0.7:
            continue
        
        detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks, detections

# Draw bounding boxes and perform face recognition
def draw_tracks(frame, tracks, detections, class_names, colors, class_counters, feature_database, identity_database, det_model, rec_model):
    global perform_recognition
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        class_id = track.get_det_class()
        bbox = track.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)

        # Get the track's feature vector
        feature = track.get_feature()

        # Check feature similarity with the database
        max_similarity = 0
        best_match_id = None
        for db_id, db_feature in feature_database.items():
            for ft in db_feature:
                similarity = 1 - cosine(feature, ft)
                if similarity > max_similarity and similarity > 0.7:
                    max_similarity = similarity
                    best_match_id = db_id

        if best_match_id is not None:
            class_specific_id = best_match_id
            identity = identity_database[class_specific_id]
        else:
            class_counters[class_id] += 1
            class_specific_id = class_counters[class_id]
            identity_database[class_specific_id] = "Unknown"
            identity = "Unknown"
            feature_database[class_specific_id] = deque(maxlen=120)

        # Perform face recognition if needed
        if perform_recognition and identity == "Unknown":
            crop_img = frame[y1:y2, x1:x2]
            if crop_img.size > 0:
                identity = recognize_from_image(crop_img, det_model, rec_model)
                identity_database[class_specific_id] = identity
                
            perform_recognition = False

        feature_database[class_specific_id].append(feature)

        # Draw bounding box and identity
        text = f"Hello {identity}"
        color = tuple(map(int, colors[class_id]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# Streamlit app
def main():
    st.title("Face Attendance System")

    # Select video source
    video_source = st.sidebar.selectbox("Select Video Source", options=["Webcam (0)", "Video File"])
    if video_source == "Webcam (0)":
        video_input = 0
    else:
        video_input = st.sidebar.file_uploader("Upload a Video File", type=["mp4", "avi", "mkv"])

    perform_recognition_button = st.sidebar.button("Trigger Face Recognition")

    # Update global perform_recognition flag
    global perform_recognition
    perform_recognition = perform_recognition_button

    # Initialize models and class names
    model = initialize_model()
    class_names = load_class_names()
    
    tracker = DeepSort(max_age=10, n_init=3, embedder="torchreid", embedder_model_name="osnet_x1_0")

    # Set up colors
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3))

    # Class and feature databases
    class_counters = defaultdict(int)
    feature_database = {}
    identity_database = {}

    WEIGHT_DET_PATH = './insightface/retinaface_resnet.onnx'
    MODEL_DET_PATH = './insightface/retinaface_resnet.onnx.prototxt'
    WEIGHT_REC_R100_PATH = './insightface/arcface_r100_v1.onnx'
    MODEL_REC_R100_PATH = './insightface/arcface_r100_v1.onnx.prototxt'
    WEIGHT_REC_R50_PATH = './insightface/arcface_r50_v1.onnx'
    MODEL_REC_R50_PATH = './insightface/arcface_r50_v1.onnx.prototxt'
    WEIGHT_REC_R34_PATH = './insightface/arcface_r34_v1.onnx'
    MODEL_REC_R34_PATH = './insightface/arcface_r34_v1.onnx.prototxt'
    WEIGHT_REC_MF_PATH = './insightface/arcface_mobilefacenet.onnx'
    MODEL_REC_MF_PATH = './insightface/arcface_mobilefacenet.onnx.prototxt'
    REMOTE_PATH = \
        'https://storage.googleapis.com/ailia-models/insightface/'

    # IMAGE_PATH = 'demo.jpg'
    # SAVE_IMAGE_PATH = 'output.png'
    rec_model = {
        'resnet100': (WEIGHT_REC_R100_PATH, MODEL_REC_R100_PATH),
        'resnet50': (WEIGHT_REC_R50_PATH, MODEL_REC_R50_PATH),
        'resnet34': (WEIGHT_REC_R34_PATH, MODEL_REC_R34_PATH),
        'mobileface': (WEIGHT_REC_MF_PATH, MODEL_REC_MF_PATH),
    }
    WEIGHT_REC_PATH, MODEL_REC_PATH = rec_model['resnet100']

    # initialize
    det_model = ailia.Net(MODEL_DET_PATH, WEIGHT_DET_PATH, env_id=ailia.get_gpu_environment_id())
    rec_model = ailia.Net(MODEL_REC_PATH, WEIGHT_REC_PATH, env_id=ailia.get_gpu_environment_id())

    # Open video capture
    if video_input == 0:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_input)

    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        # print(perform_recognition)
        if not ret:
            break

        tracks, detections = process_frame(frame, model, tracker, class_names, colors)
        frame = draw_tracks(frame, tracks, detections, class_names, colors, class_counters, feature_database, identity_database, det_model, rec_model)
        # Convert to RGB for display in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()

if __name__ == "__main__":
    main()

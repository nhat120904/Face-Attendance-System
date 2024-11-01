import os
import cv2
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime
import json
import onnx
import onnxruntime
import numpy as np
import ailia
import logging
from collections import defaultdict
from absl import app, flags
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLOv10
# from insightface.insightface import recognize_from_image
from scipy.spatial.distance import cosine
from insightface.face_model import recognize_from_image
from torchvision import transforms
from PIL import Image
from collections import defaultdict, deque
import random
# from uniform_checking.uniform_checking import check_uniform
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define command line flags
flags.DEFINE_string("video", "0", "Path to input video or webcam index (0)")
flags.DEFINE_string("output", "./output/output.mp4", "Path to output video")
flags.DEFINE_float("conf", 0.7, "Confidence threshold")
flags.DEFINE_integer("blur_id", None, "Class ID to apply Gaussian Blur")
flags.DEFINE_integer("class_id", 0, "Class ID to track")
flags.DEFINE_string("result_dir", "./result", "Path to save cropped images")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
FLAGS = flags.FLAGS

def initialize_video_capture(video_input):
    if video_input.isdigit():
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)
    else:
        cap = cv2.VideoCapture(video_input)
    
    if not cap.isOpened():
        logger.error("Error: Unable to open video source.")
        raise ValueError("Unable to open video source")
    
    return cap

def bbox_overlap(bbox1, bbox2):
    # Calculate overlap between two bounding boxes
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    # Calculate the area of intersection
    intersection_area = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
    # Calculate the area of both bounding boxes
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x4 - x3) * (y4 - y3)
    # Calculate the IoU (Intersection over Union)
    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)
    return iou > 0.8

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
    print("device: ", device)
    model.to(device)
    logger.info(f"Using {device} as processing device")
    return model


def load_class_names():
    classes_path = "./configs/coco.names"
    if not os.path.exists(classes_path):
        logger.error(f"Class names file not found at {classes_path}")
        raise FileNotFoundError("Class names file not found")
    
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")
    return class_names

def preprocess_crop_img(img):
    img = img[..., ::-1]
    img = Image.fromarray(img)
    img.save(f'result2/{random.random()}.jpg')
    x = transform(img)
    x = x.unsqueeze(0)
    return x

def process_frame(frame, model, tracker, class_names, colors):
    results = model(frame, verbose=False)[0]
    detections = []
    for det in results.boxes:
        label, confidence, bbox = det.cls, det.conf, det.xyxy[0]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)
        
        if FLAGS.class_id is None:
            if confidence < FLAGS.conf:
                continue
        else:
            if class_id != FLAGS.class_id or confidence < FLAGS.conf:
                continue
        
        detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
        
    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks, detections

def calculate_feature_similarity(feature1, feature2):
    return 1 - cosine(feature1, feature2)

def draw_tracks(frame, tracks, detections, class_names, colors, class_counters, track_class_mapping, last_save_time, feature_database, identity_database, det_model, rec_model, perform_recognition):

    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        class_id = track.get_det_class()
        # Get bounding box
        bbox = track.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get the track's feature vector
        feature = track.get_feature()
        
        # Check if this feature matches any in our database
        max_similarity = 0
        best_match_id = None
        for db_id, db_feature in feature_database.items():
            for ft in db_feature:
                similarity = calculate_feature_similarity(feature, ft)
                if similarity > max_similarity and similarity > 0.7:  # Adjust threshold as needed
                    max_similarity = similarity
                    best_match_id = db_id
            
        # print("track: ", track_id, " ,max similarity: ", max_similarity, "best macth id: ", best_match_id)
        
        if best_match_id is not None:
            # found a match, use the existing ID
            # print("Found a match")
            class_specific_id = best_match_id
            # identity = identity_database.get(class_specific_id, "Unknown")
            identity = identity_database[class_specific_id]
            # ident_name.append(identity)
        else:
            # New person, assign a new ID
            # print("New person detected")
            class_counters[class_id] += 1
            class_specific_id = class_counters[class_id]
            # class_specific_id += 1
            # print("assign new class_specific_id: ", class_specific_id)
            identity_database[class_specific_id] = "Unknown"
            identity = identity_database[class_specific_id]
            # feature_database[class_specific_id] = []
            feature_database[class_specific_id] = deque(maxlen=120)
            
        # Perform face recognition
        if perform_recognition and identity == "Unknown":
            # print("perform_recognition now is True")
            crop_img = frame[y1:y2, x1:x2]
            if crop_img.size > 0:  # Check if the cropped image is not empty
                identity = recognize_from_image(crop_img, det_model, rec_model)
                # print("identity: ", identity)
            else:
                # print("Skipping face recognition: Empty cropped image")
                identity = None
            if identity is not None and identity != "Unknown":
                # class_counters[class_id] += 1
                # class_specific_id = class_counters[class_id]
                identity_database[class_specific_id] = identity
                # Update the feature database
                
        # # Update the feature database
        # feature_database[class_specific_id].append(feature)
        
        ##TODO:
        #update feature database only if the is no bbox overlap
        # Check for bounding box overlap with existing tracks
        overlap_found = False
        for existing_track in tracks:
            if existing_track.track_id != track_id:  # Skip the current track
                existing_bbox = existing_track.to_tlbr()
                if bbox_overlap(bbox, existing_bbox):  # Check for overlap
                    overlap_found = True
                    break
        
        if not overlap_found:
            # Update the feature database only if there is no overlap
            feature_database[class_specific_id].append(feature)
        
        # Update the track_class_mapping
        # track_class_mapping[track_id] = class_specific_id
            
        text = f"{class_specific_id} - {identity_database[class_specific_id]}"
        
        # Ensure color is in the correct format (BGR)
        color = tuple(map(int, colors[class_id]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if FLAGS.blur_id is not None and class_id == FLAGS.blur_id:
            if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)
            
    return frame, last_save_time

def main(_argv):
    try:
        cap = initialize_video_capture(FLAGS.video)
        model = initialize_model()
        class_names = load_class_names()
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(FLAGS.output, fourcc, fps, (frame_width, frame_height))
        
        tracker = DeepSort(max_age=10, n_init=3, embedder="torchreid", embedder_model_name="osnet_x1_0")
        
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(class_names), 3))
        
        class_counters = defaultdict(int)
        track_class_mapping = {}
        frame_count = 0
        last_save_time = datetime.datetime.now()
        
        feature_database = {}
        identity_database = {}
        # Initialize a list to hold all frame data for JSON output
        all_frame_data = []
        errors = []
        
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
        # ort_session = onnxruntime.InferenceSession("uniform_check_v2.onnx", providers=["CPUExecutionProvider"])

        perform_recognition = False
        uniform_check = False

        while True:
            start = datetime.datetime.now()
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get key press
            key = cv2.waitKey(1) & 0xFF

            # Check key press
            if key == ord("c"):
                perform_recognition = True
                print("Face recognition triggered")
                
            elif key == ord("q"):
                break  # Exit the loop to quit the program
                
            elif key == ord("k"):
                print("Feature Database:")
                for class_specific_id, feature in feature_database.items():
                    print(f"ID: {class_specific_id}")
                    print(f"Identity: {identity_database[class_specific_id]}")
                    print("---")
            
            elif key == ord("u"):
                uniform_check = True
                print("Uniform checking triggered")
        
            
            tracks, detections = process_frame(frame, model, tracker, class_names, colors)
            frame, last_save_time = draw_tracks(frame, tracks, detections, class_names, colors, class_counters, track_class_mapping, last_save_time, feature_database, identity_database, det_model, rec_model, perform_recognition)
            
            # time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            # save_path = os.path.join(FLAGS.result_dir, f"{time_stamp}.jpg")
            # cv2.imwrite(save_path, frame)
            
            # Reset the flag after processing
            perform_recognition = False
            uniform_check = False
            end = datetime.datetime.now()
            
            # logger.info(f"Time to process frame {frame_count}: {(end - start).total_seconds():.2f} seconds")
            frame_count += 1
            
            fps_text = f"FPS: {(1 / (end - start).total_seconds())*2:.2f}"
            cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
            
            writer.write(frame)
            # cv2.imshow("YOLOv10 Object tracking", cv2.resize(frame, (480, 800)))
            cv2.imshow("YOLOv10 Object tracking", frame)
            
        
        logger.info("Class counts:")
        for class_id, count in class_counters.items():
            logger.info(f"{class_names[class_id]}: {count}")

    except Exception as e:
        logger.exception("An error occurred during processing")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass

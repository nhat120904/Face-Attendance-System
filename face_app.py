import sys
import os
import time
import glob
from collections import namedtuple
import torch
import numpy as np
from numpy.linalg import norm
import cv2
import streamlit as st
import ailia

# import original modules
sys.path.append('./util')
sys.path.append('./Face-AntiSpoofing/src')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402
from insightface_utils import PriorBox, decode, decode_landm, nms, \
    face_align_norm_crop, draw_detection  # noqa: E402
from FaceAntiSpoofing import AntiSpoof
# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_DET_PATH = './weights/retinaface_resnet.onnx'
MODEL_DET_PATH = './weights/retinaface_resnet.onnx.prototxt'
WEIGHT_REC_R100_PATH = './weights/arcface_r100_v1.onnx'
MODEL_REC_R100_PATH = './weights/arcface_r100_v1.onnx.prototxt'
WEIGHT_REC_R50_PATH = './weights/arcface_r50_v1.onnx'
MODEL_REC_R50_PATH = './weights/arcface_r50_v1.onnx.prototxt'
WEIGHT_REC_R34_PATH = './weights/arcface_r34_v1.onnx'
MODEL_REC_R34_PATH = './weights/arcface_r34_v1.onnx.prototxt'
WEIGHT_REC_MF_PATH = './weights/arcface_mobilefacenet.onnx'
MODEL_REC_MF_PATH = './weights/arcface_mobilefacenet.onnx.prototxt'
WEIGHT_GA_PATH = './weights/genderage_v1.onnx'
MODEL_GA_PATH = './weights/genderage_v1.onnx.prototxt'
WEIGHT_ANTISPOOF_PATH = './weights/AntiSpoofing_print-replay_1.5_128.onnx'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/insightface/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 512

Face = namedtuple('Face', [
    'category', 'prob', 'cosin_metric',
    'landmark', 'x', 'y', 'w', 'h',
    'embedding', 'gender', 'age', 'spoof', 'label'
])


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('InsightFace model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '--det_thresh', type=float, default=0.95,
    help='det_thresh'
)
parser.add_argument(
    '--nms_thresh', type=float, default=0.4,
    help='nms_thresh'
)
parser.add_argument(
    '--ident_thresh', type=float, default=0.25572845,
    help='ident_thresh'
)
parser.add_argument(
    '--top_k', type=int, default=5000,
    help='top_k'
)
parser.add_argument(
    '-r', '--rec_model', type=str, default='resnet100',
    choices=('resnet100', 'resnet50', 'resnet34', 'mobileface'),
    help='recognition model'
)

args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================
def preprocess(img):
    img = np.float32(img)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img


def post_processing(im_height, im_width, loc, conf, landms):
    cfg_re50 = {
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
    }

    priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
    priors = priorbox.forward()
    boxes = decode(loc[0], priors, cfg_re50['variance'])

    scale = np.array([im_width, im_height, im_width, im_height])
    boxes = boxes * scale
    scores = conf[0][:, 1]

    landms = decode_landm(landms[0], priors, cfg_re50['variance'])
    scale1 = np.array([
        im_width, im_height, im_width, im_height,
        im_width, im_height, im_width, im_height,
        im_width, im_height
    ])
    landms = landms * scale1

    inds = np.where(scores > args.det_thresh)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    dets = np.hstack(
        (boxes, scores[:, np.newaxis])
    ).astype(np.float32, copy=False)
    keep = nms(dets, args.nms_thresh)
    dets = dets[keep, :]
    landms = landms[keep]

    return dets, landms

def increased_crop(img, bbox : tuple, bbox_inc : float = 1.5):
    # Crop face based on its bounding box
    real_h, real_w = img.shape[:2]
    
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    
    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1 = 0 if x < 0 else x 
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
    y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
    
    img = img[y1:y2,x1:x2,:]
    img = cv2.copyMakeBorder(img, 
                             y1-y, int(l*bbox_inc-y2+y), 
                             x1-x, int(l*bbox_inc)-x2+x, 
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # print("check shape: ", img.shape)
    return img

def face_identification(faces, ident_feats):
    ident_faces = []
    for i in range(len(faces)):
        face = faces[i]
        emb = face.embedding
        metrics = ident_feats.dot(emb)
        category = np.argmax(metrics)
        face = face._replace(cosin_metric=metrics[category])
        if args.ident_thresh <= face.cosin_metric:
            face = face._replace(category=category)

        ident_faces.append(face)

    return ident_faces


def load_identities(rec_model):
    names = []
    feats = []
    for path in glob.glob("identities/*.PNG"):
        name = ".".join(
            path.replace(os.sep, '/').split('/')[-1].split('.')[:-1]
        )
        names.append(name)

        img = load_image(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        output = rec_model.predict({'data': img})[0]

        embedding = output[0]
        embedding_norm = norm(embedding)
        normed_embedding = embedding / embedding_norm
        feats.append(normed_embedding)

    feats = np.vstack(feats)

    return names, feats


# ======================
# Main functions
# ======================
def predict(img, det_model, rec_model, ga_model, anti_spoof):
    # initial preprocesses
    im_height, im_width, _ = img.shape
    _img = preprocess(img)

    # feedforward
    output = det_model.predict({'img': _img})

    loc, conf, landms = output

    bboxes, landmarks = post_processing(im_height, im_width, loc, conf, landms)

    faces = []
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, 0:4]
        prob = bboxes[i, 4]
        landmark = landmarks[i].reshape((5, 2))

        _img = face_align_norm_crop(img, landmark=landmark)
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        
        # crop_height, crop_width, _ = _img.shape
        # min_size_threshold = 200  # Minimum size in pixels for width and height
        # if crop_height < min_size_threshold or crop_width < min_size_threshold:
        #     st.warning("Face detected is too small. Please move closer to the camera.")
            # return
         
        _img_spoof = increased_crop(img, bbox, bbox_inc=1.5)
        _img_spoof = cv2.cvtColor(_img_spoof, cv2.COLOR_BGR2RGB)
        _img_spoof = cv2.resize(_img_spoof, (360, 360))
        cv2.imwrite('aligned_face_{}.png'.format(i), _img_spoof)
        
        # check anti-spoofing
        # print("check shape: ", _img_spoof.shape)
        isSpoof = anti_spoof([_img_spoof])[0]
        isSpoof = isSpoof[0][0]
        label = np.argmax(isSpoof)
        
        # recognize people in image
        _img = np.transpose(_img, (2, 0, 1))
        _img = np.expand_dims(_img, axis=0)
        _img = _img.astype(np.float32)
        output = rec_model.predict({'data': _img})[0]

        embedding = output[0]
        embedding_norm = norm(embedding)
        normed_embedding = embedding / embedding_norm

        output = ga_model.predict({'data': _img})[0]

        g = output[0, 0:2]
        gender = np.argmax(g)
        a = output[0, 2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))

        face = Face(
            category=None,
            prob=prob,
            cosin_metric=1,
            landmark=landmark,
            x=bbox[0] / im_width,
            y=bbox[1] / im_height,
            w=(bbox[2] - bbox[0]) / im_width,
            h=(bbox[3] - bbox[1]) / im_height,
            embedding=normed_embedding,
            gender=gender,
            age=age,
            spoof=isSpoof,
            label=label
        )
        faces.append(face)

    return faces

def recognize_from_video(video, det_model, rec_model, ga_model, anti_spoof):
    # Initialize the webcam capture
    capture = cv2.VideoCapture(video)
    
    # Load identities for recognition
    ident_names, ident_feats = load_identities(rec_model)
    
    # Streamlit image display window
    frame_window = st.image([])

    # Process video frames in a loop
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            st.warning("Webcam feed not accessible.")
            break

        # Run prediction on the frame
        faces = predict(frame, det_model, rec_model, ga_model, anti_spoof)
        faces = face_identification(faces, ident_feats)

        # Draw detection results on the frame
        res_img = draw_detection(frame, faces, ident_names)

        # Display the processed frame in Streamlit
        frame_window.image(res_img, channels="BGR")

    # Release resources after the loop ends
    capture.release()
    
def register_face(frame, identity, det_model, anti_spoof):
    # Detect faces in the frame
    im_height, im_width, _ = frame.shape
    _img = preprocess(frame)
    # print("original image shape: ", _img.shape)
    # feedforward
    results = det_model.predict({'img': _img})
    loc, conf, landms = results
    bboxes, landmarks = post_processing(im_height, im_width, loc, conf, landms)
    # print("results: ", results)
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, 0:4]
        prob = bboxes[i, 4]
        landmark = landmarks[i].reshape((5, 2))

        _img = face_align_norm_crop(frame, landmark=landmark)

        _img_spoof = increased_crop(frame, bbox, bbox_inc=1.5)
        _img_spoof = cv2.cvtColor(_img_spoof, cv2.COLOR_BGR2RGB)
        _img_spoof = cv2.resize(_img_spoof, (360, 360))
        cv2.imwrite(f"spoof_check.PNG", _img_spoof)
        # print("register shape: ", _img.shape)
        
        # Check if the detected face is real using anti-spoofing
        is_real = anti_spoof([_img_spoof])[0]
        # print("check real: ", is_real[0][0])
        is_real = is_real[0][0] > 0.1
        if not is_real:
            st.error("Please use a real face for registration.")
            return
        
        # Save the cropped image to the identities folder
        identities_folder = "./identities"
        if not os.path.exists(identities_folder):
            os.makedirs(identities_folder)
            
        if os.path.exists(os.path.join(identities_folder, f"{identity}.PNG")):
            st.error(f"Face already registered for {identity}.")
            return
        
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(identities_folder, f"{identity}.PNG")
        cv2.imwrite(image_path, _img)
        
        st.success(f"Face registered successfully for {identity}. Image saved at {image_path}.")
        return
        
    st.error("No face detected. Please try again.")

@st.cache_resource
def load_models(WEIGHT_REC_PATH, MODEL_REC_PATH):
    try:
        # Initialize with CPU first
        env_id = 0  # CPU
        
        # Test GPU initialization
        try:
            det_model_test = ailia.Net(MODEL_DET_PATH, WEIGHT_DET_PATH, env_id=2)
            env_id = 2  # GPU
        except:
            st.warning("GPU initialization failed, falling back to CPU")
        
        # Initialize models
        det_model = ailia.Net(MODEL_DET_PATH, WEIGHT_DET_PATH, env_id=env_id)
        rec_model = ailia.Net(MODEL_REC_PATH, WEIGHT_REC_PATH, env_id=env_id)
        ga_model = ailia.Net(MODEL_GA_PATH, WEIGHT_GA_PATH, env_id=env_id)
        
        return det_model, rec_model, ga_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise

def main():
    st.title("Ultra Face Attendace System")
    st.write("This is a face recognition system that uses the InsightFace model for face detection and recognition.")
    st.write("The system uses the RetinaFace model for face detection and the ArcFace model for face recognition.")
    # st.write("Make sure to put your face close to the camera and no light shining directly into the camera, otherwise the system might not behave accurately, ")
    
    # Sidebar for model selection and configuration
    st.sidebar.header("Model Configuration")
    
    # Recognition model selection
    rec_model_options = {
        'resnet100': (WEIGHT_REC_R100_PATH, MODEL_REC_R100_PATH),
        'resnet50': (WEIGHT_REC_R50_PATH, MODEL_REC_R50_PATH),
        'resnet34': (WEIGHT_REC_R34_PATH, MODEL_REC_R34_PATH),
        'mobileface': (WEIGHT_REC_MF_PATH, MODEL_REC_MF_PATH),
    }
    selected_rec_model = st.sidebar.selectbox("Recognition Model", list(rec_model_options.keys()))
    WEIGHT_REC_PATH, MODEL_REC_PATH = rec_model_options[selected_rec_model]

    # Load models
    det_model, rec_model, ga_model = load_models(WEIGHT_REC_PATH, MODEL_REC_PATH)
    anti_spoof = AntiSpoof(WEIGHT_ANTISPOOF_PATH)
    
    # Add file upload section in sidebar
    st.sidebar.markdown("### Face Registration")
    uploaded_file = st.sidebar.file_uploader("Upload your face image", type=['jpg', 'jpeg', 'png'])
    identity = st.sidebar.text_input("Enter your name")
    register_button = st.sidebar.button("Register Face")
    
    if register_button:
        if uploaded_file is not None and identity:
            # Convert uploaded file to opencv image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            try:
                # Register face using the uploaded image
                register_face(frame, identity, det_model, anti_spoof)
            except Exception as e:
                st.sidebar.error(f"Error registering face: {str(e)}")
        else:
            if not uploaded_file:
                st.sidebar.warning("Please upload an image first")
            if not identity:
                st.sidebar.warning("Please enter your name")
    
    # Webcam input
    video_input = 0
    if video_input == 0:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_input)
        
    # Create registration form
    with st.form(key='registration_form'):
        identity = st.text_input("Enter your name for registration")
        submit_button = st.form_submit_button(label='Register Face')
        
        if submit_button:
            if identity:
                ret, frame = cap.read()
                if ret:
                    st.text("Capturing frame for registration...")
                    register_face(frame, identity, det_model, anti_spoof)
                else:
                    st.error("Unable to capture frame. Please check your video source.")
            else:
                st.warning("Please enter a name before registering.")
        
    # Video stream from webcam
    st.write("**Live Video Feed**")
    recognize_from_video(0, det_model, rec_model, ga_model, anti_spoof)

if __name__ == '__main__':
    main()

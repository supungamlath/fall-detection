import torch
import torch.multiprocessing as mp
import logging
import av
from constants import *
from detection import *
from model.model import LSTMModel

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass


class Arguments:
    def __init__(self):
        self.fps = OUTPUT_VIDEO_FPS
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")


class LiveFallDetector:
    def __init__(self):
        self.args = Arguments()
        self.flip_video = False
        self.model = LSTMModel(hidden_dim=48, num_layers=2, dropout=0.1, num_classes=7)
        self.model.load_state_dict(
            torch.load("model/lstm_weights.sav", map_location=self.args.device)
        )
        self.model.eval()
        self.ip_set = []
        self.lstm_set = []
        self.num_matched = 0
        logging.basicConfig(level=logging.ERROR)

    def set_flip(self, flip):
        self.flip_video = flip

    def process_frame(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.flip_video:
            img = cv2.flip(img, 1)
        image = extract_pose_detect_fall(
            img, self.model, self.args, self.ip_set, self.lstm_set, self.num_matched
        )
        return av.VideoFrame.from_ndarray(image, format="bgr24")


class VideoFallDetector:
    def __init__(self):
        self.args = Arguments()
        logging.basicConfig(level=logging.INFO)

    def begin(self, video_path, fps):
        self.args.fps = fps
        e = mp.Event()
        queue = mp.Queue()

        process1 = mp.Process(
            target=extract_pose_keyframes_mp,
            args=(queue, video_path, self.args, e),
        )
        process1.start()

        process2 = mp.Process(
            target=detect_with_lstm_mp,
            args=(queue, self.args, e),
        )
        process2.start()

        process1.join()
        process2.join()

        return

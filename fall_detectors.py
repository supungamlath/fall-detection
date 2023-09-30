import torch
import torch.multiprocessing as mp
import logging
from constants import *
from detection import *

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


class FallDetector:
    def __init__(self):
        self.args = Arguments()
        logging.basicConfig(level=logging.INFO)
        self.e = mp.Event()
        self.queue = mp.Queue()
        self.progress = mp.Value("i", 0)

    def begin(self, video_path, fps):
        self.args.fps = fps

        self.process1 = mp.Process(
            target=extract_pose_keyframes,
            args=(self.queue, video_path, self.args, self.e),
        )
        self.process2 = mp.Process(
            target=detect_with_lstm,
            args=(self.queue, self.args, self.e),
        )

        self.process1.start()
        self.process2.start()

        self.process1.join()
        self.process2.join()

        return

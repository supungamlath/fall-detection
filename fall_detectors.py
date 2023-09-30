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

    def begin(self, video_path, fps):
        self.args.fps = fps
        e = mp.Event()
        queue = mp.Queue()

        process1 = mp.Process(
            target=extract_pose_keyframes,
            args=(queue, video_path, self.args, e),
        )
        process1.start()

        process2 = mp.Process(
            target=detect_with_lstm,
            args=(queue, self.args, e),
        )
        process2.start()

        process1.join()
        process2.join()

        return

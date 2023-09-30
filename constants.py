DEFAULT_CONSEC_FRAMES = 36
RESOLUTION_SCALER = 0.4
OUTPUT_VIDEO_FPS = 18

MIN_THRESH = 0.5
HEAD_THRESHOLD = 1e-5
EMA_FRAMES = DEFAULT_CONSEC_FRAMES * 3
EMA_BETA = 1 / (EMA_FRAMES + 1)
FEATURE_SCALAR = {
    "ratio_bbox": 1,
    "gf": 1,
    "angle_vertical": 1,
    "re": 1,
    "ratio_derivative": 1,
    "log_angle": 1,
}
FEATURE_LIST = ["ratio_bbox", "log_angle", "re", "ratio_derivative", "gf"]
FRAME_FEATURES = 2

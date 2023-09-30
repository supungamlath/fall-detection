from enum import IntEnum, unique
from typing import List
import cv2
import numpy as np


@unique
class BodyPart(IntEnum):
    """Body part locations in the 'coordinates' list."""

    Nose = 0
    LEye = 1
    REye = 2
    LEar = 3
    REar = 4
    LShoulder = 5
    RShoulder = 6
    LElbow = 7
    RElbow = 8
    LWrist = 9
    RWrist = 10
    LHip = 11
    RHip = 12
    LKnee = 13
    RKnee = 14
    LAnkle = 15
    RAnkle = 16


SKELETON_CONNECTIONS = [
    ("H", "N", (210, 182, 247)),
    ("N", "B", (210, 182, 247)),
    ("B", "KL", (210, 182, 247)),
    ("B", "KR", (210, 182, 247)),
    ("KL", "KR", (210, 182, 247)),
]

COLOR_ARRAY = [
    (210, 182, 247),
    (127, 127, 127),
    (194, 119, 227),
    (199, 199, 199),
    (34, 189, 188),
    (141, 219, 219),
    (207, 190, 23),
    (150, 152, 255),
    (189, 103, 148),
    (138, 223, 152),
]

UNMATCHED_COLOR = (180, 119, 31)


def write_on_image(img: np.ndarray, text: str, color: List) -> np.ndarray:
    """Write text at the top of the image."""
    # Add a white border to top of image for writing text
    img = cv2.copyMakeBorder(
        src=img,
        top=int(0.1 * img.shape[0]),
        bottom=0,
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        dst=None,
        value=[255, 255, 255],
    )

    # Calculate the size of the text bounding box
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    font_thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Calculate the position to center the text
    text_x = (img.shape[1] - text_size[0]) // 2  # Center horizontally
    text_y = 30  # Place on top vertically

    cv2.putText(
        img=img,
        text=text,
        org=(text_x, text_y),
        fontFace=font,
        fontScale=font_scale,
        color=color,
        thickness=font_thickness,
    )

    return img


def visualise_tracking(
    img: np.ndarray,
    keypoint_sets: List,
    width: int,
    height: int,
    num_matched: int,
    vis_keypoints: bool = True,
    vis_skeleton: bool = True,
) -> np.ndarray:
    """Draw keypoints/skeleton on the output video frame."""

    if vis_keypoints or vis_skeleton:
        for i, keypoints in enumerate(keypoint_sets):
            if keypoints is None:
                continue
            keypoints = keypoints["keypoints"]
            if vis_skeleton:
                for p1i, p2i, color in SKELETON_CONNECTIONS:
                    if keypoints[p1i] is None or keypoints[p2i] is None:
                        continue

                    p1 = (
                        int(keypoints[p1i][0] * width),
                        int(keypoints[p1i][1] * height),
                    )
                    p2 = (
                        int(keypoints[p2i][0] * width),
                        int(keypoints[p2i][1] * height),
                    )

                    if p1 == (0, 0) or p2 == (0, 0):
                        continue
                    if i < num_matched:
                        color = COLOR_ARRAY[i % 10]
                    else:
                        color = UNMATCHED_COLOR

                    cv2.line(img=img, pt1=p1, pt2=p2, color=color, thickness=3)

    return img

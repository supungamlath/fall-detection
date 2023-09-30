import torch
import math
import cv2
import logging
import time
import numpy as np
from pose.visuals import write_on_image, visualise_tracking
from pose.processor import Processor
from pose.pose_features import *
from helper_functions import get_hist
from constants import *
from model.model import LSTMModel


def get_source(source_file):
    cam = cv2.VideoCapture(source_file)
    return cam


def resize(img):
    # Resize the video
    height, width = img.shape[:2]
    width_height = (
        int(width * RESOLUTION_SCALER // 16) * 16,
        int(height * RESOLUTION_SCALER // 16) * 16,
    )
    return width, height, width_height


def extract_pose_keyframes_mp(queue, video_path, args, event):
    try:
        video = get_source(video_path)
        ret_val, img = video.read()
    except Exception as e:
        queue.put(None)
        event.set()
        print("Exception occurred:", e)
        print("The video doesn't exist")
        return

    width, height, width_height = resize(img)
    logging.debug(f"Target width and height = {width_height}")
    processor_singleton = Processor(width_height, args)

    frame = 0
    while not event.is_set():
        ret_val, img = video.read()
        frame += 1
        curr_time = time.time()
        print("Frame:", frame, end="\r")
        if img is None:
            print("No more images captured")
            if not event.is_set():
                event.set()
            break

        img = cv2.resize(img, (width, height))
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        keypoint_sets, bb_list, width_height = processor_singleton.single_image(img)
        assert bb_list is None or (type(bb_list) == list)
        if bb_list:
            assert type(bb_list[0]) == tuple
            assert type(bb_list[0][0]) == tuple
        # assume bb_list is a of the form [(x1,y1),(x2,y2)),etc.]

        anns = [get_kp(keypoints.tolist()) for keypoints in keypoint_sets]
        ubboxes = [
            (np.asarray([width, height]) * np.asarray(ann[1])).astype("int32")
            for ann in anns
        ]
        lbboxes = [
            (np.asarray([width, height]) * np.asarray(ann[2])).astype("int32")
            for ann in anns
        ]
        bbox_list = [
            (np.asarray([width, height]) * np.asarray(box)).astype("int32")
            for box in bb_list
        ]
        uhist_list = [get_hist(hsv_img, bbox) for bbox in ubboxes]
        lhist_list = [get_hist(img, bbox) for bbox in lbboxes]
        keypoint_sets = [
            {
                "keypoints": keyp[0],
                "up_hist": uh,
                "lo_hist": lh,
                "time": curr_time,
                "box": box,
            }
            for keyp, uh, lh, box in zip(anns, uhist_list, lhist_list, bbox_list)
        ]

        cv2.polylines(img, ubboxes, True, (255, 0, 0), 2)
        cv2.polylines(img, lbboxes, True, (0, 255, 0), 2)
        for box in bbox_list:
            cv2.rectangle(img, tuple(box[0]), tuple(box[1]), ((0, 0, 255)), 2)

        dict_vis = {
            "img": img,
            "keypoint_sets": keypoint_sets,
            "width": width,
            "height": height,
            "tagged_df": {
                "text": "",
                "color": [0, 0, 0],
            },
        }
        queue.put(dict_vis)

    queue.put(None)
    return


def show_tracked_img(img_dict, ip_set, num_matched, output_video, args):
    img = img_dict["img"]
    tagged_df = img_dict["tagged_df"]
    keypoints_frame = [person[-1] for person in ip_set]
    img = visualise_tracking(
        img=img,
        keypoint_sets=keypoints_frame,
        width=img_dict["width"],
        height=img_dict["height"],
        num_matched=num_matched,
    )

    img = write_on_image(img=img, text=tagged_df["text"], color=tagged_df["color"])

    if output_video is None:
        filename = "fall_detected_video.mp4"
        output_video = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=args.fps,
            frameSize=img.shape[:2][::-1],
        )
    else:
        output_video.write(img)
    return img, output_video


def detect_with_lstm_mp(queue, args, event):
    model = LSTMModel(hidden_dim=48, num_layers=2, dropout=0.1, num_classes=7)
    model.load_state_dict(
        torch.load("model/lstm_weights.sav", map_location=args.device)
    )
    model.eval()
    output_video = None
    ip_set = []
    lstm_set = []
    max_length_mat = DEFAULT_CONSEC_FRAMES
    num_matched = 0

    while True:
        if not queue.empty():
            dict_frame = queue.get()

            if dict_frame is None:
                if not event.is_set():
                    event.set()
                break

            kp_frame = dict_frame["keypoint_sets"]
            num_matched, new_num, indxs_unmatched = match_ip(
                ip_set, kp_frame, lstm_set, num_matched, max_length_mat
            )
            valid1_idxs, prediction = get_all_features(ip_set, lstm_set, model)
            text, color = activity_name(prediction + 5)
            dict_frame["tagged_df"]["text"] = text
            dict_frame["tagged_df"]["color"] = color
            img, output_video = show_tracked_img(
                dict_frame, ip_set, num_matched, output_video, args
            )

    output_video.release()
    del model
    return


def get_all_features(ip_set, lstm_set, model):
    valid_idxs = []
    invalid_idxs = []
    predictions = [15] * len(ip_set)  # 15 is the tag for None

    for i, ips in enumerate(ip_set):
        # ip set for a particular person
        last1 = None
        last2 = None
        for j in range(-2, -1 * DEFAULT_CONSEC_FRAMES - 1, -1):
            if ips[j] is not None:
                if last1 is None:
                    last1 = j
                elif last2 is None:
                    last2 = j
        if ips[-1] is None:
            invalid_idxs.append(i)
            # continue
        else:
            ips[-1]["features"] = {}
            # get re, gf, angle, bounding box ratio, ratio derivative
            ips[-1]["features"]["height_bbox"] = get_height_bbox(ips[-1])
            ips[-1]["features"]["ratio_bbox"] = FEATURE_SCALAR[
                "ratio_bbox"
            ] * get_ratio_bbox(ips[-1])

            body_vector = ips[-1]["keypoints"]["N"] - ips[-1]["keypoints"]["B"]
            ips[-1]["features"]["angle_vertical"] = FEATURE_SCALAR[
                "angle_vertical"
            ] * get_angle_vertical(body_vector)
            # print(ips[-1]["features"]["angle_vertical"])
            ips[-1]["features"]["log_angle"] = FEATURE_SCALAR["log_angle"] * np.log(
                1 + np.abs(ips[-1]["features"]["angle_vertical"])
            )

            if last1 is None:
                invalid_idxs.append(i)
                # continue
            else:
                ips[-1]["features"]["re"] = FEATURE_SCALAR["re"] * get_rot_energy(
                    ips[last1], ips[-1]
                )
                ips[-1]["features"]["ratio_derivative"] = FEATURE_SCALAR[
                    "ratio_derivative"
                ] * get_ratio_derivative(ips[last1], ips[-1])
                if last2 is None:
                    invalid_idxs.append(i)
                    # continue
                else:
                    ips[-1]["features"]["gf"] = get_gf(ips[last2], ips[last1], ips[-1])
                    valid_idxs.append(i)

        xdata = []
        if ips[-1] is None:
            if last1 is None:
                xdata = [0] * len(FEATURE_LIST)
            else:
                for feat in FEATURE_LIST[:FRAME_FEATURES]:
                    xdata.append(ips[last1]["features"][feat])
                xdata += [0] * (len(FEATURE_LIST) - FRAME_FEATURES)
        else:
            for feat in FEATURE_LIST:
                if feat in ips[-1]["features"]:
                    xdata.append(ips[-1]["features"][feat])
                else:
                    xdata.append(0)

        xdata = torch.Tensor(xdata).view(-1, 1, 5)
        outputs, lstm_set[i][0] = model(xdata, lstm_set[i][0])
        if i == 0:
            prediction = torch.max(outputs.data, 1)[1][0].item()
            confidence = torch.max(outputs.data, 1)[0][0].item()
            fpd = True
            # fpd = False
            if fpd:
                if prediction in [1, 2, 3, 5]:
                    lstm_set[i][3] -= 1
                    lstm_set[i][3] = max(lstm_set[i][3], 0)

                    if lstm_set[i][2] < EMA_FRAMES:
                        if ips[-1] is not None:
                            lstm_set[i][2] += 1
                            lstm_set[i][1] = (
                                lstm_set[i][1] * (lstm_set[i][2] - 1)
                                + get_height_bbox(ips[-1])
                            ) / lstm_set[i][2]
                    else:
                        if ips[-1] is not None:
                            lstm_set[i][1] = (1 - EMA_BETA) * get_height_bbox(
                                ips[-1]
                            ) + EMA_BETA * lstm_set[i][1]

                elif prediction == 0:
                    if (
                        ips[-1] is not None
                        and lstm_set[i][1] != 0
                        and abs(ips[-1]["features"]["angle_vertical"]) < math.pi / 4
                    ) or confidence < 0.4:
                        prediction = 7
                    else:
                        lstm_set[i][3] += 1
                        if lstm_set[i][3] < DEFAULT_CONSEC_FRAMES // 4:
                            prediction = 7
                else:
                    lstm_set[i][3] -= 1
                    lstm_set[i][3] = max(lstm_set[i][3], 0)
            predictions[i] = prediction

    return valid_idxs, predictions[0] if len(predictions) > 0 else 15

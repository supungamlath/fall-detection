import numpy as np
import cv2
from PIL import Image, ImageDraw


def pop_and_add(l, val, max_length):
    if len(l) == max_length:
        l.pop(0)
    l.append(val)


def last_ip(ips):
    for i, ip in enumerate(reversed(ips)):
        if ip is not None:
            return ip, len(ips) - i


def dist(ip1, ip2):
    ip1 = ip1["keypoints"]
    ip2 = ip2["keypoints"]
    return np.sqrt(np.sum((ip1["N"] - ip2["N"]) ** 2 + (ip1["B"] - ip2["B"]) ** 2))


def valid_candidate_hist(ip):
    if ip is not None:
        return ip["up_hist"] is not None
    else:
        return False


def get_hist(img, bbox, nbins=3):
    if not np.any(bbox):
        return None

    mask = Image.new("L", (img.shape[1], img.shape[0]), 0)
    ImageDraw.Draw(mask).polygon(list(bbox.flatten()), outline=1, fill=1)
    mask = np.array(mask)
    hist = cv2.calcHist([img], [0, 1], mask, [nbins, 2 * nbins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)

    return hist

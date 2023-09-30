from .visuals import BodyPart
import numpy as np
from helper_functions import *
from constants import *


def activity_name(activity_no):
    if 1 <= activity_no <= 4:
        return "Falling", [0, 0, 255]
    elif activity_no == 5:
        return "Fallen", [0, 0, 255]
    elif activity_no == 12:
        return "Warning", [0, 50, 200]
    elif 6 <= activity_no <= 11:
        return "Normal", [100, 255, 0]
    elif activity_no == 20:
        return "None", [0, 0, 0]


def match_ip(
    ip_set, new_ips, lstm_set, num_matched, consecutive_frames=DEFAULT_CONSEC_FRAMES
):
    len_ip_set = len(ip_set)
    added = [False for _ in range(len_ip_set)]

    new_len_ip_set = len_ip_set
    for new_ip in new_ips:
        if not is_valid(new_ip):
            continue
        # assert valid_candidate_hist(new_ip)
        cmin = [MIN_THRESH, -1]
        for i in range(len_ip_set):
            if not added[i] and dist(last_ip(ip_set[i])[0], new_ip) < cmin[0]:
                # here add dome condition that last_ip(ip_set[0] >-5 or someting)
                cmin[0] = dist(last_ip(ip_set[i])[0], new_ip)
                cmin[1] = i

        if cmin[1] == -1:
            ip_set.append([None for _ in range(consecutive_frames - 1)] + [new_ip])
            lstm_set.append([None, 0, 0, 0])  # Initial hidden state of lstm is None
            new_len_ip_set += 1

        else:
            added[cmin[1]] = True
            pop_and_add(ip_set[cmin[1]], new_ip, consecutive_frames)

    new_matched = num_matched

    removed_indx = []
    removed_match = []

    for i in range(len(added)):
        if not added[i]:
            pop_and_add(ip_set[i], None, consecutive_frames)
        if ip_set[i] == [None for _ in range(consecutive_frames)]:
            if i < num_matched:
                new_matched -= 1
                removed_match.append(i)

            new_len_ip_set -= 1
            removed_indx.append(i)

    for i in sorted(removed_indx, reverse=True):
        ip_set.pop(i)
        lstm_set.pop()

    return new_matched, new_len_ip_set, removed_match


def extend_vector(p1, p2, l):
    p1 += (p1 - p2) * l / (2 * np.linalg.norm((p1 - p2), 2))
    p2 -= (p1 - p2) * l / (2 * np.linalg.norm((p1 - p2), 2))
    return p1, p2


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def get_kp(kp):
    threshold1 = 5e-3

    # dict of np arrays of coordinates
    inv_pend = {}
    numx = (
        kp[BodyPart.LEar][2] * kp[BodyPart.LEar][0]
        + kp[BodyPart.LEye][2] * kp[BodyPart.LEye][0]
        + kp[BodyPart.REye][2] * kp[BodyPart.REye][0]
        + kp[BodyPart.REar][2] * kp[BodyPart.REar][0]
    )
    numy = (
        kp[BodyPart.LEar][2] * kp[BodyPart.LEar][1]
        + kp[BodyPart.LEye][2] * kp[BodyPart.LEye][1]
        + kp[BodyPart.REye][2] * kp[BodyPart.REye][1]
        + kp[BodyPart.REar][2] * kp[BodyPart.REar][1]
    )
    den = (
        kp[BodyPart.LEar][2]
        + kp[BodyPart.LEye][2]
        + kp[BodyPart.REye][2]
        + kp[BodyPart.REar][2]
    )

    if den < HEAD_THRESHOLD:
        inv_pend["H"] = None
    else:
        inv_pend["H"] = np.array([numx / den, numy / den])

    if all(
        [
            kp[BodyPart.LShoulder],
            kp[BodyPart.RShoulder],
            kp[BodyPart.LShoulder][2] > threshold1,
            kp[BodyPart.RShoulder][2] > threshold1,
        ]
    ):
        inv_pend["N"] = np.array(
            [
                (kp[BodyPart.LShoulder][0] + kp[BodyPart.RShoulder][0]) / 2,
                (kp[BodyPart.LShoulder][1] + kp[BodyPart.RShoulder][1]) / 2,
            ]
        )
    else:
        inv_pend["N"] = None

    if all(
        [
            kp[BodyPart.LHip],
            kp[BodyPart.RHip],
            kp[BodyPart.LHip][2] > threshold1,
            kp[BodyPart.RHip][2] > threshold1,
        ]
    ):
        inv_pend["B"] = np.array(
            [
                (kp[BodyPart.LHip][0] + kp[BodyPart.RHip][0]) / 2,
                (kp[BodyPart.LHip][1] + kp[BodyPart.RHip][1]) / 2,
            ]
        )
    else:
        inv_pend["B"] = None

    if kp[BodyPart.LKnee] is not None and kp[BodyPart.LKnee][2] > threshold1:
        inv_pend["KL"] = np.array([kp[BodyPart.LKnee][0], kp[BodyPart.LKnee][1]])
    else:
        inv_pend["KL"] = None

    if kp[BodyPart.RKnee] is not None and kp[BodyPart.RKnee][2] > threshold1:
        inv_pend["KR"] = np.array([kp[BodyPart.RKnee][0], kp[BodyPart.RKnee][1]])
    else:
        inv_pend["KR"] = None

    if inv_pend["B"] is not None:
        if inv_pend["N"] is not None:
            height = np.linalg.norm(inv_pend["N"] - inv_pend["B"], 2)
            LS, RS = extend_vector(
                np.asarray(kp[BodyPart.LShoulder][:2]),
                np.asarray(kp[BodyPart.RShoulder][:2]),
                height / 4,
            )
            LB, RB = extend_vector(
                np.asarray(kp[BodyPart.LHip][:2]),
                np.asarray(kp[BodyPart.RHip][:2]),
                height / 3,
            )
            ubbox = (LS, RS, RB, LB)

            if inv_pend["KL"] is not None and inv_pend["KR"] is not None:
                lbbox = (LB, RB, inv_pend["KR"], inv_pend["KL"])
            else:
                lbbox = ([0, 0], [0, 0])
                # lbbox = None
        else:
            ubbox = ([0, 0], [0, 0])
            # ubbox = None
            if inv_pend["KL"] is not None and inv_pend["KR"] is not None:
                lbbox = (
                    np.array(kp[BodyPart.LHip][:2]),
                    np.array(kp[BodyPart.RHip][:2]),
                    inv_pend["KR"],
                    inv_pend["KL"],
                )
            else:
                lbbox = ([0, 0], [0, 0])
                # lbbox = None
    else:
        ubbox = ([0, 0], [0, 0])
        lbbox = ([0, 0], [0, 0])
        # ubbox = None
        # lbbox = None
    # condition = (inv_pend["H"] is None) and (inv_pend['N'] is not None and inv_pend['B'] is not None)
    # if condition:
    #     print("half disp")

    return inv_pend, ubbox, lbbox


def get_angle(v0, v1):
    return np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))


def is_valid(ip):
    assert ip is not None

    ip = ip["keypoints"]
    return ip["B"] is not None and ip["N"] is not None and ip["H"] is not None


def get_rot_energy(ip0, ip1):
    t = ip1["time"] - ip0["time"]
    ip0 = ip0["keypoints"]
    ip1 = ip1["keypoints"]
    m1 = 1
    m2 = 5
    m3 = 5
    energy = 0
    den = 0
    N1 = ip1["N"] - ip1["B"]
    N0 = ip0["N"] - ip0["B"]
    d2sq = N1.dot(N1)
    w2sq = (get_angle(N0, N1) / t) ** 2
    energy += m2 * d2sq * w2sq

    den += m2 * d2sq
    H1 = ip1["H"] - ip1["B"]
    H0 = ip0["H"] - ip0["B"]
    d1sq = H1.dot(H1)
    w1sq = (get_angle(H0, H1) / t) ** 2
    energy += m1 * d1sq * w1sq
    den += m1 * d1sq

    energy = energy / (2 * den)
    # energy = energy/2
    return energy


def get_angle_vertical(v):
    return np.math.atan2(-v[0], -v[1])


def get_gf(ip0, ip1, ip2):
    t1 = ip1["time"] - ip0["time"]
    t2 = ip2["time"] - ip1["time"]
    ip0 = ip0["keypoints"]
    ip1 = ip1["keypoints"]
    ip2 = ip2["keypoints"]

    m1 = 1
    m2 = 15
    g = 10
    H2 = ip2["H"] - ip2["N"]
    H1 = ip1["H"] - ip1["N"]
    H0 = ip0["H"] - ip0["N"]
    d1 = np.sqrt(H1.dot(H1))
    theta_1_plus_2_2 = get_angle_vertical(H2)
    theta_1_plus_2_1 = get_angle_vertical(H1)
    theta_1_plus_2_0 = get_angle_vertical(H0)
    # print("H: ",H0,H1,H2)
    N2 = ip2["N"] - ip2["B"]
    N1 = ip1["N"] - ip1["B"]
    N0 = ip0["N"] - ip0["B"]
    d2 = np.sqrt(N1.dot(N1))
    # print("N: ",N0,N1,N2)
    theta_2_2 = get_angle_vertical(N2)
    theta_2_1 = get_angle_vertical(N1)
    theta_2_0 = get_angle_vertical(N0)
    # print("theta_2_2:",theta_2_2,"theta_2_1:",theta_2_1,"theta_2_0:",theta_2_0,sep=", ")
    theta_1_0 = theta_1_plus_2_0 - theta_2_0
    theta_1_1 = theta_1_plus_2_1 - theta_2_1
    theta_1_2 = theta_1_plus_2_2 - theta_2_2

    # print("theta1: ",theta_1_0,theta_1_1,theta_1_2)
    # print("theta2: ",theta_2_0,theta_2_1,theta_2_2)

    theta2 = theta_2_1
    theta1 = theta_1_1

    del_theta1_0 = (get_angle(H0, H1)) / t1
    del_theta1_1 = (get_angle(H1, H2)) / t2

    del_theta2_0 = (get_angle(N0, N1)) / t1
    del_theta2_1 = (get_angle(N1, N2)) / t2
    # print("del_theta2_1:",del_theta2_1,"del_theta2_0:",del_theta2_0,sep=",")
    del_theta1 = 0.5 * (del_theta1_1 + del_theta1_0)
    del_theta2 = 0.5 * (del_theta2_1 + del_theta2_0)

    doubledel_theta1 = (del_theta1_1 - del_theta1_0) / 0.5 * (t1 + t2)
    doubledel_theta2 = (del_theta2_1 - del_theta2_0) / 0.5 * (t1 + t2)
    # print("doubledel_theta2:",doubledel_theta2)

    d1 = d1 / d2
    d2 = 1
    # print("del_theta",del_theta1,del_theta2)
    # print("doubledel_theta",doubledel_theta1,doubledel_theta2)

    Q_RD1 = 0
    Q_RD1 += m1 * d1 * doubledel_theta1 * doubledel_theta1
    Q_RD1 += (m1 * d1 * d1 + m1 * d1 * d2 * np.cos(theta1)) * doubledel_theta2
    Q_RD1 += m1 * d1 * d2 * np.sin(theta1) * del_theta2 * del_theta2
    Q_RD1 -= m1 * g * d2 * np.sin(theta1 + theta2)

    Q_RD2 = 0
    Q_RD2 += (m1 * d1 * d1 + m1 * d1 * d2 * np.cos(theta1)) * doubledel_theta1
    Q_RD2 += (
        (m1 + m2) * d2 * d2 + m1 * d1 * d1 + 2 * m1 * d1 * d2 * np.cos(theta1)
    ) * doubledel_theta2
    Q_RD2 -= (
        2 * m1 * d1 * d2 * np.sin(theta1) * del_theta2 * del_theta1
        + m1 * d1 * d2 * np.sin(theta1) * del_theta1 * del_theta1
    )
    Q_RD2 -= (m1 + m2) * g * d2 * np.sin(theta2) + m1 * g * d1 * np.sin(theta1 + theta2)

    # print("Energy: ", Q_RD1 + Q_RD2)
    return Q_RD1 + Q_RD2


def get_height_bbox(ip):
    bbox = ip["box"]
    assert isinstance(bbox, np.ndarray)
    diff_box = bbox[1] - bbox[0]
    return diff_box[1]


def get_ratio_bbox(ip):
    bbox = ip["box"]
    assert isinstance(bbox, np.ndarray)
    diff_box = bbox[1] - bbox[0]
    if diff_box[1] == 0:
        diff_box[1] += 1e5 * diff_box[0]
    assert np.any(diff_box > 0)
    ratio = diff_box[0] / diff_box[1]
    return ratio


def get_ratio_derivative(ip0, ip1):
    ratio_der = None
    time = ip1["time"] - ip0["time"]
    diff_box = ip1["features"]["ratio_bbox"] - ip0["features"]["ratio_bbox"]
    assert time != 0
    ratio_der = diff_box / time

    return ratio_der

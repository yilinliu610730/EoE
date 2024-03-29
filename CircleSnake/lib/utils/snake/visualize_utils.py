import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from lib.utils import img_utils
from lib.utils.img_utils import colors
from lib.utils.snake import snake_cityscapes_utils, snake_config

R = 8
GREEN = (18, 127, 15)
WHITE = (255, 255, 255)

iter_det = 0


def visualize_snake_detection_circle(orig_img, data):
    # img = orig_img.copy()
    # img = img_utils.bgr_to_rgb(img)
    # def blend_hm_img(hm, img):
    #     hm = np.max(hm, axis=0)
    #     h, w = hm.shape[:2]
    #     img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    #     hm = np.array([255, 255, 255]) - (hm.reshape(h, w, 1) * colors[0]).astype(np.uint8)
    #     ratio = 0.5
    #     blend = (img * ratio + hm * (1 - ratio)).astype(np.uint8)
    #     return blend
    #
    # img = blend_hm_img(data["ct_hm"], img)

    img = ((data["inp"].transpose(1, 2, 0) * snake_config.std + snake_config.mean) * 255).astype(
        np.uint8
    )
    ct_ind = np.array(data["ct_ind"])
    w = img.shape[1] // snake_config.down_ratio
    xs = ct_ind % w
    ys = ct_ind // w

    img = cv2.resize(img, (512, 512))

    # Show each bounding circle
    for i in range(len(data["radius"])):
        radius = data["radius"][i][0]
        x, y = data["circle_center"][i]
        # assert(abs(xs[i] - x) <= 1)
        # assert(abs(ys[i] - y) <= 1)
        # assert (x >= 0)
        # assert(y >= 0)
        x *= 4
        y *= 4
        radius *= 4
        cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 0), 1)
    cv2.imshow("Ground Truth - Detection", img)
    cv2.waitKey(0)
    global iter_det
    path = os.path.join("/home/ethan/Documents/CircleSnake/data/debug", str(iter_det))
    if not os.path.exists(path):
        os.makedirs(path)

    cv2.imwrite(os.path.join(path, "gt_det.png"), img)
    iter_det += 1


def visualize_snake_detection(img, data):
    def blend_hm_img(hm, img):
        hm = np.max(hm, axis=0)
        h, w = hm.shape[:2]
        img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        hm = np.array([255, 255, 255]) - (hm.reshape(h, w, 1) * colors[0]).astype(np.uint8)
        ratio = 0.5
        blend = (img * ratio + hm * (1 - ratio)).astype(np.uint8)
        return blend

    img = img_utils.bgr_to_rgb(img)
    blend = blend_hm_img(data["ct_hm"], img)

    # Show the raw image
    plt.imshow(blend)
    ct_ind = np.array(data["ct_ind"])
    w = img.shape[1] // snake_config.down_ratio
    xs = ct_ind % w
    ys = ct_ind // w

    # Show each bounding box
    for i in range(len(data["wh"])):
        w, h = data["wh"][i]
        x_min, y_min = xs[i] - w / 2, ys[i] - h / 2
        x_max, y_max = xs[i] + w / 2, ys[i] + h / 2
        plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min])
    plt.show()


def visualize_cp_detection(img, data):
    act_ind = data["act_ind"]
    awh = data["awh"]

    act_hm_w = data["act_hm"].shape[2]
    cp_h, cp_w = data["cp_hm"][0].shape[1], data["cp_hm"][0].shape[2]

    img = img_utils.bgr_to_rgb(img)
    plt.imshow(img)

    for i in range(len(act_ind)):
        act_ind_ = act_ind[i]
        ct = act_ind_ % act_hm_w, act_ind_ // act_hm_w
        w, h = awh[i]
        abox = np.array([ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2])

        cp_ind_ = data["cp_ind"][i]
        cp_wh_ = data["cp_wh"][i]

        for j in range(len(cp_ind_)):
            ct = cp_ind_[j] % cp_w, cp_ind_[j] // cp_w
            x = ct[0] / cp_w * w
            y = ct[1] / cp_h * h
            x_min = (x - cp_wh_[j][0] / 2 + abox[0]) * snake_config.down_ratio
            y_min = (y - cp_wh_[j][1] / 2 + abox[1]) * snake_config.down_ratio
            x_max = (x + cp_wh_[j][0] / 2 + abox[0]) * snake_config.down_ratio
            y_max = (y + cp_wh_[j][1] / 2 + abox[1]) * snake_config.down_ratio
            plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min])

    plt.show()


def visualize_snake_evolution(img, data):
    # img = img_utils.bgr_to_rgb(img)
    # plt.imshow(img)
    # for poly in data['i_it_py']:
    #     poly = poly * 4
    #     poly = np.append(poly, [poly[0]], axis=0)
    #     plt.plot(poly[:, 0], poly[:, 1])
    #     plt.scatter(poly[0, 0], poly[0, 1], edgecolors='w')
    # plt.show()
    img = ((data["inp"].transpose(1, 2, 0) * snake_config.std + snake_config.mean) * 255).astype(
        np.uint8
    )
    for poly in data["i_gt_py"]:
        poly = poly * 4
        poly = np.append(poly, [poly[0]], axis=0)
        cv2.polylines(img, [np.int32(poly)], True, (0, 255, 0), 1)

    # for poly in data['i_gt_py']:
    #     poly = poly * 4
    #     poly = np.append(poly, [poly[0]], axis=0)
    #     cv2.polylines(img, [np.int32(poly)], True, (0, 0, 255), 1)
    # cv2.imshow("Ground Truth - Evolution", img)
    cv2.imwrite("/home/VANDERBILT/liuy99/Desktop/gt_evolution.png", img)
    # cv2.waitKey(0)


def visualize_snake_octagon(img, extreme_points):
    img = img_utils.bgr_to_rgb(img)
    octagons = []
    bboxes = []
    ex_points = []
    for i in range(len(extreme_points)):
        for j in range(len(extreme_points[i])):
            bbox = get_bbox(extreme_points[i][j] * 4)
            octagon = snake_cityscapes_utils.get_octagon(extreme_points[i][j] * 4)
            bboxes.append(bbox)
            octagons.append(octagon)
            ex_points.append(extreme_points[i][j])
    _, ax = plt.subplots(1)
    ax.imshow(img)
    n = len(octagons)
    for i in range(n):
        x, y, x_max, y_max = bboxes[i]
        ax.add_patch(
            patches.Polygon(
                xy=[[x, y], [x, y_max], [x_max, y_max], [x_max, y]],
                fill=False,
                linewidth=1,
                edgecolor="r",
            )
        )
        octagon = np.append(octagons[i], octagons[i][0]).reshape(-1, 2)
        ax.plot(octagon[:, 0], octagon[:, 1])
        ax.scatter(ex_points[i][:, 0] * 4, ex_points[i][:, 1] * 4, edgecolors="w")
    plt.show()


def get_bbox(ex):
    x = ex[:, 0]
    y = ex[:, 1]
    bbox = [np.min(x), np.min(y), np.max(x), np.max(y)]
    return bbox

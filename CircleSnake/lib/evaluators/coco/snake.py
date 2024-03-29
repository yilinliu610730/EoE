import json
import math
import os

import cv2
import numpy as np
import pycocotools.coco as coco
import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval
from PIL import Image
from shapely.geometry import Polygon
from external.cityscapesscripts.evaluation import \
    evalInstanceLevelSemanticLabeling
from lib.config import cfg
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.utils import data_utils
from lib.utils.snake import (snake_cityscapes_utils, snake_config,
                             snake_eval_utils, snake_poly_utils)


class Evaluator:
    def __init__(self, result_dir):
        self.results = []
        self.img_ids = []
        self.aps = []

        self.result_dir = result_dir
        os.system("mkdir -p {}".format(self.result_dir))

        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args["ann_file"]
        self.data_root = args["data_root"]
        self.coco = coco.COCO(self.ann_file)

        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self.contiguous_category_id_to_json_id = {category_id: category_id for category_id in
                                                  self.json_category_id_to_contiguous_id.values()}

        self.iter_num = 0
        self.dice = 0
        self.num_images = 0

        self.mask = []
        self.rotate_mask = []
        self.confusion_matrix = np.zeros((4, 4))
        self.fp = []
        self.fn = []
        self.tp = []
        self.tn = []

    def calculateIOU(self, predictedList, actualList, label):

        for class_id in [0, 1, 2, 3]:
            row = np.where(label == 0)
            col = list(filter(lambda d: d['category_id'] == class_id, actualList))
            match_matrix = np.zeros((int(len(row[0])), int(len(col))))
            for i in range(len(match_matrix)):
                predicted = predictedList[row[0][i]]
                for j in range(len(match_matrix[0])):
                    actual = col[j]
                    actual_post = np.array(actual['segmentation'][0])
                    actual_post = actual_post.reshape((len(actual_post) // 2, 2))
                    polygonP = Polygon(predicted)
                    polygonA = Polygon(actual_post)
                    intersect = polygonP.intersection(polygonA).area
                    union = polygonP.union(polygonA).area
                    iou = intersect / union

                    match_matrix[i, j] = iou  # key: actual annotations' id, value: iou

            index_list = np.where(match_matrix > 0)
            for si in range(len(index_list[0])):
                if np.max(match_matrix) == 0:
                    break

                index = np.where(match_matrix == np.max(match_matrix))
                x = int(index[0][0])
                y = int(index[1][0])

                if (-1 in match_matrix[x, :]) or (-1 in match_matrix[:, y]):
                    match_matrix[x, y] = 0  # calclate the single object
                    continue
                else:
                    match_matrix[x, y] = -1
                    self.confusion_matrix[class_id][0] += 1

            for i in range(len(match_matrix)):
                if len(np.where(match_matrix[i, :] < 0)[0]) == 0:
                    self.confusion_matrix[class_id][2] += 1
            if (len(match_matrix) > 0):
                for j in range(len(match_matrix[0])):
                    if len(np.where(match_matrix[:, j] < 0)[0]) == 0:
                        self.confusion_matrix[class_id][3] += 1
            print("")

    def evaluate(self, output, batch):
        detection = output["detection"]
        score = detection[:, 4].detach().cpu().numpy()
        label = detection[:, 5].detach().cpu().numpy().astype(int)
        py = output["py"][-1].detach().cpu().numpy() * snake_config.down_ratio

        if not cfg.rotate_reproduce and not cfg.debug_test and len(py) == 0:
            return

        img_id = int(batch["meta"]["img_id"][0])
        center = batch["meta"]["center"][0].detach().cpu().numpy()
        scale = batch["meta"]["scale"][0].detach().cpu().numpy()

        h, w = batch["inp"].size(2), batch["inp"].size(3)
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img["height"], img["width"]
        py = [data_utils.affine_transform(py_, trans_output_inv) for py_ in py]
        rles = snake_eval_utils.coco_poly_to_rle(py, ori_h, ori_w)

        img_path = os.path.join(self.data_root, img["file_name"])
        orig_img = cv2.imread(img_path)
        cnt = 0

        if cfg.debug_test:
            path = os.path.join(self.data_root, img["file_name"])
            orig_img = cv2.imread(path)

            # Prediction
            pred_img = orig_img.copy()
            for poly_idx, polys in enumerate(py):
                category = label[cnt]
                color = (0, 255, 0)  # green
                if category == 1:
                    color = (0, 255, 255)  # blue
                elif category == 2:
                    color = (255, 0, 255)  # pink
                elif category == 3:
                    color = (255, 255, 0)  # yellow

                poly_corrected = np.zeros(shape=(128, 2), dtype=np.int32)

                # Limit to border
                for i, (poly_x, poly_y) in enumerate(polys):
                    if poly_x < 0:
                        poly_x = 0
                    elif poly_x > ori_w:
                        poly_x = ori_w
                    if poly_y < 0:
                        poly_y = 0
                    elif poly_y > ori_h:
                        poly_y = ori_h
                    poly_corrected[i] = int(round(poly_x)), int(round(poly_y))
                cv2.polylines(pred_img, [np.int32(poly_corrected)], True, color, 2)
                text_pt_x = min(np.array(poly_corrected)[:, 0])
                text_pt_y = min(np.array(poly_corrected)[:, 1])
                cv2.rectangle(
                    pred_img,
                    (text_pt_x, text_pt_y),
                    (text_pt_x + 40, text_pt_y - 15),
                    color,
                    -1,
                )
                cv2.putText(
                    pred_img,
                    "%.2f" % score[poly_idx],
                    (text_pt_x, text_pt_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )
                cnt += 1

            # Ground truth
            gt_img = orig_img.copy()
            ann_ids = self.coco.getAnnIds(img_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                instance_poly = [
                    np.array(poly, dtype=int).reshape(-1, 2) for poly in ann["segmentation"]
                ]
                color = (0, 255, 0)  # green
                category = ann["category_id"]
                if category == 1:
                    color = (0, 255, 255)  # blue
                elif category == 2:
                    color = (255, 0, 255)  # pink
                elif category == 3:
                    color = (255, 255, 0)  # yellow
                cv2.polylines(gt_img, instance_poly, True, color, 2)
                text_pt_x = min(np.array(instance_poly)[0, :, 0])
                text_pt_y = min(np.array(instance_poly)[0, :, 1])
                cv2.rectangle(
                    gt_img,
                    (text_pt_x, text_pt_y),
                    (text_pt_x + 40, text_pt_y - 15),
                    color,
                    -1,
                )
                cv2.putText(
                    gt_img,
                    "%.2f" % 1,
                    (text_pt_x, text_pt_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

            self.calculateIOU(py, anns, label)
            image_name = os.path.basename(path)

            if cfg.show_images:
                cv2.imshow("Prediction", pred_img)
                cv2.imshow("GT", gt_img)
                cv2.waitKey(0)
            if cfg.save_images:
                path = os.path.join(
                    "/home/VANDERBILT/liuy99/Documents/EOS/CircleSnake3/data/debug_snake/predicted"
                )
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(os.path.join(path, image_name), pred_img)

                path = os.path.join(
                    "/home/VANDERBILT/liuy99/Documents/EOS/CircleSnake3/data/debug_snake/truth"
                )
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(os.path.join(path, image_name), gt_img)

        if cfg.dice:
            # Prediction mask
            pred_mask = np.zeros(orig_img.shape, dtype=np.uint8)

            for polys in py:
                cv2.drawContours(pred_mask, [polys.astype(int)], -1, (255, 255, 255), -1)

            if cfg.debug_test and cfg.show_images:
                cv2.imshow("Pred Mask", pred_mask)

            # GT Mask
            gt_mask = np.zeros(orig_img.shape, dtype=np.uint8)

            ann_ids = self.coco.getAnnIds(img_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                instance_poly = [
                    np.array(poly, dtype=int).reshape(-1, 2) for poly in ann["segmentation"]
                ]
                cv2.drawContours(gt_mask, instance_poly, -1, (255, 255, 255), -1)

            if cfg.debug_test and cfg.show_images:
                cv2.imshow("Truth Mask", gt_mask)

            M = np.float32([[1, 0, 0], [0, 1, 0]])
            pred_mask = cv2.warpAffine(pred_mask, M, (pred_mask.shape[1], pred_mask.shape[0]))

            gt_mask = gt_mask.astype(np.bool)[:, :, 0]
            pred_mask = pred_mask.astype(np.bool)[:, :, 0]

            # pred_mask = gt_mask

            intersection = np.logical_and(gt_mask, pred_mask)
            dice_score = 2 * intersection.sum() / (gt_mask.sum() + pred_mask.sum())

            if math.isnan(dice_score):
                dice_score = 1

            if cfg.debug_test:
                cv2.imshow(
                    "Intersection",
                    intersection.astype(np.uint8) * 125 + gt_mask.astype(np.uint8) * 125,
                )
                print(dice_score)

            self.dice += dice_score
            self.num_images += 1

            if cfg.debug_test and cfg.show_images:
                cv2.waitKey(0)
        coco_dets = []
        for i in range(len(rles)):
            detection = {
                "image_id": img_id,
                "category_id": self.contiguous_category_id_to_json_id[label[i]],
                "segmentation": rles[i],
                "score": float("{:.2f}".format(score[i])),
            }
            coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)
        self.iter_num += 1

    def evaluate_rotate(self, output, batch, rotate=False):
        detection = output["detection"]
        score = detection[:, ].detach().cpu().numpy()
        label = detection[:, 5].detach().cpu().numpy().astype(int)
        py = output["py"][-1].detach().cpu().numpy() * snake_config.down_ratio

        if not cfg.rotate_reproduce and not cfg.debug_test and len(py) == 0:
            return

        img_id = int(batch["meta"]["img_id"][0])
        center = batch["meta"]["center"][0].detach().cpu().numpy()
        scale = batch["meta"]["scale"][0].detach().cpu().numpy()

        h, w = batch["inp"].size(2), batch["inp"].size(3)
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        py = [data_utils.affine_transform(py_, trans_output_inv) for py_ in py]

        path = os.path.join(self.data_root, img["file_name"])
        orig_img = cv2.imread(path)

        # Prediction mask
        pred_mask = np.zeros(orig_img.shape, dtype=np.uint8)

        for polys in py:
            cv2.drawContours(pred_mask, [polys.astype(int)], -1, (255, 255, 255), -1)

        if rotate:
            pred_mask = cv2.rotate(pred_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # cv2.imshow("Pred", pred_mask)
        # cv2.waitKey(0)

        pred_mask = pred_mask.astype(np.bool)[:, :, 0]

        if rotate:
            self.rotate_mask.append(pred_mask)
        else:
            self.mask.append(pred_mask)

    def summarize_rotate(self):
        for i in range(len(self.mask)):
            intersection = np.logical_and(self.mask[i], self.rotate_mask[i])
            dice_score = 2 * intersection.sum() / (self.mask[i].sum() + self.rotate_mask[i].sum())

            # print(dice_score)

            import math

            if math.isnan(dice_score):
                print("nan")
                dice_score = 1
            self.dice += dice_score
        print(self.dice / len(self.mask))

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, "results.json"), "w"))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, "results.json"))
        coco_eval = COCOeval(self.coco, coco_dets, "segm")
        coco_eval.params.maxDets = [1000, 1000, 1000]
        coco_eval.params.catIds = [0, 1, 2, 3]
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.results = []
        self.img_ids = []
        self.aps.append(coco_eval.stats[0])
        if cfg.dice:
            self.dice /= self.num_images
            print("Dice Score:", self.dice)
        print('Confusion Matrix for Class 0: ', self.confusion_matrix[0])
        print('Confusion Matrix for Class 1: ', self.confusion_matrix[1])
        print('Confusion Matrix for Class 2: ', self.confusion_matrix[2])
        print('Confusion Matrix for Class 3: ', self.confusion_matrix[3])

        # Output per-category APs
        per_category_aps = coco_eval.eval['precision']
        category_ids = list(self.coco.cats.keys())

        for category_index, category_id in enumerate(category_ids):
            category_name = self.coco.cats[category_id]['name']
            aps = per_category_aps[:, 0, category_index, 0, 0]
            aps = aps[aps > -1]

            # Compute the mean AP across all custom IoU thresholds for the current category
            ap = np.mean(aps) if per_category_aps.size else float("nan")

            print(f"Mean AP for {category_name}: {ap * 100}")
        return {"ap": coco_eval.stats[0]}


class DetectionEvaluator:
    def __init__(self, result_dir):
        self.results = []
        self.img_ids = []
        self.aps = []

        self.result_dir = result_dir
        os.system("mkdir -p {}".format(self.result_dir))

        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args["ann_file"]
        self.data_root = args["data_root"]
        self.coco = coco.COCO(self.ann_file)

        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def evaluate(self, output, batch):
        detection = output["detection"]
        detection = detection[0] if detection.dim() == 3 else detection
        box = detection[:, :4].detach().cpu().numpy() * snake_config.down_ratio
        score = detection[:, 4].detach().cpu().numpy()
        label = detection[:, 5].detach().cpu().numpy().astype(int)

        img_id = int(batch["meta"]["img_id"][0])
        center = batch["meta"]["center"][0].detach().cpu().numpy()
        scale = batch["meta"]["scale"][0].detach().cpu().numpy()

        if len(box) == 0:
            return

        h, w = batch["inp"].size(2), batch["inp"].size(3)
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img["height"], img["width"]

        coco_dets = []
        for i in range(len(label)):
            box_ = data_utils.affine_transform(box[i].reshape(-1, 2), trans_output_inv).ravel()
            box_[2] -= box_[0]
            box_[3] -= box_[1]
            box_ = list(map(lambda x: float("{:.2f}".format(x)), box_))
            detection = {
                "image_id": img_id,
                "category_id": self.contiguous_category_id_to_json_id[label[i]],
                "bbox": box_,
                "score": float("{:.2f}".format(score[i])),
            }
            coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, "results.json"), "w"))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, "results.json"))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.params.maxDets = [1000, 1000, 1000]
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.results = []
        self.img_ids = []
        self.aps.append(coco_eval.stats[0])
        return {"ap": coco_eval.stats[0]}


Evaluator = Evaluator if cfg.segm_or_bbox == "segm" else DetectionEvaluator

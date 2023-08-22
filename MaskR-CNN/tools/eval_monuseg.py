# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.datasets import register_coco_instances

# Register MoNuSeg dataset
register_coco_instances("monuseg_train", {},
                        "/home/ethan/Documents/detectron2/datasets/monuseg/MoNuSeg_train2021.json",
                        "/home/ethan/Documents/detectron2/datasets/monuseg/train")
register_coco_instances("monuseg_val", {}, "/home/ethan/Documents/detectron2/datasets/monuseg/MoNuSeg_val2021.json",
                        "/home/ethan/Documents/detectron2/datasets/monuseg/val")
register_coco_instances("monuseg_test", {}, "/home/ethan/Documents/detectron2/datasets/monuseg/MoNuSeg_test2021.json",
                        "/home/ethan/Documents/detectron2/datasets/monuseg/test")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("monuseg_train", "monuseg_val")
cfg.DATASETS.TEST = ("monuseg_test",)
cfg.DATALOADER.NUM_WORKERS = 16
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.OUTPUT_DIR = "/home/ethan/Documents/detectron2/projects/MoNuSeg/monuseg_coco/"

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

cfg.MODEL.WEIGHTS = "/home/ethan/Documents/detectron2/projects/MoNuSeg/monuseg_coco/model_final.pth"
cfg.TEST.DETECTIONS_PER_IMAGE = 1000
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
trainer = DefaultTrainer(cfg)
trainer.resume_or_load()
predictor = DefaultPredictor(cfg)

dice = 0
num_images = 0

# # Create side-by-side
dataset_test = DatasetCatalog.get("monuseg_test")
monuseg_test_metadata = MetadataCatalog.get("monuseg_test")
for idx, d in enumerate(dataset_test):
    file_name = d["file_name"]

    img = cv2.imread(file_name)
    pred_img = img.copy()

    vis1 = Visualizer(img[:, :, ::-1], metadata=monuseg_test_metadata, scale=1.0)
    vis2 = Visualizer(img[:, :, ::-1], metadata=monuseg_test_metadata, scale=1.0)
    ground = vis1.draw_dataset_dict(d)
    outputs = predictor(img)
    pred = vis2.draw_instance_predictions(outputs["instances"].to("cpu"),)

    pred_masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
    pred_mask = np.zeros((512, 512), dtype=np.bool)
    from detectron2.utils.visualizer import GenericMask
    pred_contours = []
    for i in range(pred_masks.shape[0]):
        # pred_mask = np.logical_or(pred_mask, pred_masks[i])
        pred_mask = pred_masks[i]
        pred_contour = np.array(GenericMask(pred_mask, 512, 512).polygons[0], dtype=int).reshape(-1, 2)
        pred_contours.append(pred_contour)
    cv2.drawContours(pred_img, pred_contours, -1, (0, 255, 0), 2)

    # cv2.imshow("Pred", pred_img)
    # cv2.waitKey(0)


    # gt_mask = np.zeros(img.shape, dtype=np.uint8)
    # anns = d.get("annotations", None)
    # for ann in anns:
    #     polys = [np.array(poly, dtype=int).reshape(-1, 2) for poly in ann['segmentation']]
    #     cv2.drawContours(gt_mask, polys, -1, (255, 255, 255), -1)
    # gt_mask = gt_mask.astype(np.bool)[:, :, 0]
    #
    # intersection = np.logical_and(gt_mask, pred_mask)
    # dice_score = 2 * intersection.sum() / (gt_mask.sum() + pred_mask.sum())
    #
    # dice += dice_score
    # num_images += 1
    # #
    # cv2.imshow("Prediction Mask", pred_mask.astype(np.uint8) * 255)
    # cv2.imshow("Prediction", pred.get_image()[:, :, ::-1])
    # cv2.imshow("Ground Truth", ground.get_image()[:, :, ::-1])
    # cv2.imshow("Ground Truth Mask", gt_mask.astype(np.uint8) * 255)
    # cv2.waitKey(0)
    #
    # concat = np.concatenate((ground.get_image()[:, :, ::-1], pred.get_image()[:, :, ::-1]), axis=1)

    path = os.path.join("/home/ethan/Documents/CircleSnake/data/debug", str(idx))
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(os.path.join(path, "mask_pred_segm.png"), pred_img)

# for d in dataset_test:
#     file_name = d["file_name"]
#
#     img = cv2.imread(file_name)
#
#     vis1 = Visualizer(img[:, :, ::-1], metadata=monuseg_test_metadata, scale=1.0)
#     vis2 = Visualizer(img[:, :, ::-1], metadata=monuseg_test_metadata, scale=1.0)
#     ground = vis1.draw_dataset_dict(d)
#     outputs = predictor(img)
#     pred = vis2.draw_instance_predictions(outputs["instances"].to("cpu"), )
#
#     pred_masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
#     pred_mask = np.zeros((512, 512), dtype=np.bool)
#     for i in range(pred_masks.shape[0]):
#         pred_mask = np.logical_or(pred_mask, pred_masks[i])
#
#     gt_mask = np.zeros(img.shape, dtype=np.uint8)
#     anns = d.get("annotations", None)
#     for ann in anns:
#         polys = [np.array(poly, dtype=int).reshape(-1, 2) for poly in ann['segmentation']]
#         cv2.drawContours(gt_mask, polys, -1, (255, 255, 255), -1)
#     gt_mask = gt_mask.astype(np.bool)[:, :, 0]
#
#     intersection = np.logical_and(gt_mask, pred_mask)
#     dice_score = 2 * intersection.sum() / (gt_mask.sum() + pred_mask.sum())
#
#     dice += dice_score
#     num_images += 1

    # cv2.imshow("Prediction Mask", pred_mask.astype(np.uint8) * 255)
    # cv2.imshow("Prediction", pred.get_image()[:, :, ::-1])
    # cv2.imshow("Ground Truth", ground.get_image()[:, :, ::-1])
    # cv2.imshow("Ground Truth Mask", gt_mask.astype(np.uint8) * 255)
    # cv2.waitKey(0)

    # concat = np.concatenate((ground.get_image()[:, :, ::-1], pred.get_image()[:, :, ::-1]), axis=1)

    # path = os.path.join("/home/ethan/Documents/CircleSnake/data/debug", str(i))
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # cv2.imwrite(os.path.join(path, "mask_pred_segm.png"), pred.get_image()[:, :, ::-1])

    # concat = np.concatenate((ground.get_image()[:, :, ::-1], pred.get_image()[:, :, ::-1]), axis=1)

    # out_path = "/home/sybbure/CircleNet/detectron2/projects/MoNuSeg/round5/ground_vs_pred/{0}".format(file_name.split("/")[-1])
    # cv2.imwrite(out_path, concat)

def draw_mask(pred_masks):
    # pred_masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
    pred_mask = np.zeros((512, 512), dtype=np.bool)
    for i in range(pred_masks.shape[0]):
        pred_mask = np.logical_or(pred_mask, pred_masks[i])

    return pred_mask


for i, d in enumerate(dataset_test):
    file_name = d["file_name"]

    img = cv2.imread(file_name)
    rot_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    outputs = predictor(img)
    rot_outputs = predictor(rot_img)

    # pred_masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
    # pred_mask = np.zeros((512, 512), dtype=np.bool)
    # for i in range(pred_masks.shape[0]):
    #     pred_mask = np.logical_or(pred_mask, pred_masks[i])

    pred_mask = draw_mask(np.asarray(outputs["instances"].to("cpu").pred_masks))
    rot_mask = draw_mask(np.asarray(rot_outputs["instances"].to("cpu").pred_masks))
    rot_mask = cv2.rotate(rot_mask.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE).astype(np.bool)

    intersection = np.logical_and(rot_mask, pred_mask)
    dice_score = 2 * intersection.sum() / (rot_mask.sum() + pred_mask.sum())

    # print(dice_score)

    import math

    if math.isnan(dice_score):
        dice_score = 1
    dice += dice_score
    num_images += 1

    # cv2.imshow("Prediction Mask", pred_mask.astype(np.uint8) * 255)
    # cv2.imshow("Rotated Prediction Mask", rot_mask.astype(np.uint8) * 255)
    # cv2.imshow("Ground Truth", ground.get_image()[:, :, ::-1])
    # cv2.imshow("Ground Truth Mask", gt_mask.astype(np.uint8) * 255)
    cv2.waitKey(0)

    # concat = np.concatenate((ground.get_image()[:, :, ::-1], pred.get_image()[:, :, ::-1]), axis=1)

    # path = os.path.join("/home/ethan/Documents/CircleSnake/data/debug", str(i))
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # cv2.imwrite(os.path.join(path, "mask_pred_segm.png"), pred.get_image()[:, :, ::-1])

print("Dice", dice / num_images)

# Evaluate using COCO AP metrics
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# evaluator = COCOEvaluator("monuseg_test", ("segm"), False, output_dir=cfg.OUTPUT_DIR, use_fast_impl=False)
evaluator = detectron2.evaluation.COCOEvaluator("monuseg_test", tasks=("segm",), distributed=False,
                                                output_dir=cfg.OUTPUT_DIR, use_fast_impl=False)
val_loader = build_detection_test_loader(cfg, "monuseg_test")
inference_on_dataset(trainer.model, val_loader, evaluator)

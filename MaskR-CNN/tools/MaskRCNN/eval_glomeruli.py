# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import math
from tqdm import tqdm

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import GenericMask

ROTATE_CONSISTENCY_EVAL = False
DICE_EVAL = True
VISUALISE = False
SAVE = False

# Register MoNuSeg dataset
def register_datasets():
    register_coco_instances("train", {}, "/home/ethan/Documents/CircleSnake/data/kidpath_coco/train_circle.json", "/home/ethan/Documents/CircleSnake/data/kidpath_coco/train")
    register_coco_instances("val", {}, "/home/ethan/Documents/CircleSnake/data/kidpath_coco/validate_circle.json", "/home/ethan/Documents/CircleSnake/data/kidpath_coco/validate")
    register_coco_instances("test", {}, "/home/ethan/Documents/CircleSnake/data/kidpath_coco/test_circle.json", "/home/ethan/Documents/CircleSnake/data/kidpath_coco/test")


def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("test",)
    cfg.OUTPUT_DIR = "/home/ethan/Documents/detectron2/projects/MoNuSeg/mask_r50_glomeruli_pretrain_v2"
    cfg.MODEL.WEIGHTS = "/home/ethan/Documents/detectron2/projects/MoNuSeg/mask_r50_glomeruli_pretrain_v2/model_0002999.pth"
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.90
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    return cfg

def _draw_mask(pred_masks):
    # pred_masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
    pred_mask = np.zeros((512, 512), dtype=np.bool)
    for i in range(pred_masks.shape[0]):
        pred_mask = np.logical_or(pred_mask, pred_masks[i])
    return pred_mask

if __name__ == "__main__":
    register_datasets()
    cfg = setup_cfg()

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    predictor = DefaultPredictor(cfg)
    dataset_test = DatasetCatalog.get("test")
    monuseg_test_metadata = MetadataCatalog.get("test")

    if DICE_EVAL:
        dice = 0
        num_images = 0

        for idx, d in tqdm(enumerate(dataset_test)):
            file_name = d["file_name"]

            img = cv2.imread(file_name)
            pred_img = img.copy()

            outputs = predictor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # if VISUALISE:
                # # vis1 = Visualizer(img[:, :, ::-1], metadata=monuseg_test_metadata, scale=1.0)
                # # vis2 = Visualizer(img[:, :, ::-1], metadata=monuseg_test_metadata, scale=1.0)
                # ground = vis1.draw_dataset_dict(d)
                # pred = vis2.draw_instance_predictions(outputs["instances"].to("cpu"),)

            pred_masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
            pred_mask = np.zeros((512, 512), dtype=np.bool)
            pred_contours = []
            for i in range(pred_masks.shape[0]):
                pred_mask = np.logical_or(pred_mask, pred_masks[i])
                # pred_mask = pred_masks[i]
                pred_contour = np.array(GenericMask(pred_masks[i], 512, 512).polygons[0], dtype=int).reshape(-1, 2)
                pred_contours.append(pred_contour)
            cv2.drawContours(pred_img, pred_contours, -1, (0, 255, 0), 2)


            gt_img = img.copy()
            gt_mask = np.zeros(img.shape, dtype=np.uint8)
            anns = d.get("annotations", None)
            for ann in anns:
                polys = [np.array(poly, dtype=int).reshape(-1, 2) for poly in ann['segmentation']]
                cv2.drawContours(gt_mask, polys, -1, (255, 255, 255), -1)
                cv2.drawContours(gt_img, polys, -1, (0, 255, 0), 2)
            gt_mask = gt_mask.astype(np.bool)[:, :, 0]

            intersection = np.logical_and(gt_mask, pred_mask)
            dice_score = 2 * intersection.sum() / (gt_mask.sum() + pred_mask.sum())

            dice += dice_score
            num_images += 1

            if VISUALISE:
                if not SAVE:
                    cv2.imshow("Prediction Mask", pred_mask.astype(np.uint8) * 255)
                    cv2.imshow("Prediction", pred_img)
                    cv2.imshow("Ground Truth", gt_img)
                    cv2.imshow("Ground Truth Mask", gt_mask.astype(np.uint8) * 255)
                    cv2.waitKey(0)

                # concat = np.concatenate((ground.get_image()[:, :, ::-1], pred.get_image()[:, :, ::-1]), axis=1)
                else:
                    path = os.path.join("/home/ethan/Documents/CircleSnake/data/debug_grey", str(idx))
                    if not os.path.exists(path):
                        os.makedirs(path)
                    cv2.imwrite(os.path.join(path, "mask_pred_segm.png"), pred_img)

        print("Dice Segmentation", dice / num_images)

    if ROTATE_CONSISTENCY_EVAL:
        dice = 0
        num_images = 0
        for i, d in tqdm(enumerate(dataset_test)):
            file_name = d["file_name"]

            img = cv2.imread(file_name)
            rot_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            outputs = predictor(img)
            rot_outputs = predictor(rot_img)

            pred_masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
            pred_mask = np.zeros((512, 512), dtype=np.bool)
            for i in range(pred_masks.shape[0]):
                pred_mask = np.logical_or(pred_mask, pred_masks[i])

            pred_mask = _draw_mask(np.asarray(outputs["instances"].to("cpu").pred_masks))
            rot_mask = _draw_mask(np.asarray(rot_outputs["instances"].to("cpu").pred_masks))
            rot_mask = cv2.rotate(rot_mask.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE).astype(np.bool)

            intersection = np.logical_and(rot_mask, pred_mask)
            if rot_mask.sum() + pred_mask.sum() == 0:
                dice_score = 1
            else:
                dice_score = 2 * intersection.sum() / (rot_mask.sum() + pred_mask.sum())

            # print(dice_score)

            dice += dice_score
            num_images += 1

            if VISUALISE:
                cv2.imshow("Prediction Mask", pred_mask.astype(np.uint8) * 255)
                cv2.imshow("Rotated Prediction Mask", rot_mask.astype(np.uint8) * 255)
                cv2.waitKey(0)

        print("Rotate Consistency Dice", dice / num_images)


    # Evaluate using COCO AP metrics
    evaluator = detectron2.evaluation.COCOEvaluator("test", tasks=("segm",), distributed=False,
                                                    output_dir=cfg.OUTPUT_DIR, use_fast_impl=False)
    val_loader = build_detection_test_loader(cfg, "test")
    inference_on_dataset(trainer.model, val_loader, evaluator)
#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import pycocotools
import logging
import os
from collections import OrderedDict
import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators,  inference_on_dataset
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.engine import DefaultPredictor

# Register Eos dataset
CLASS_NAMES =["eos","papilla","rbc", "cluster"]
# 数据集路径
DATASET_ROOT = '/home/VANDERBILT/liuy99/Documents/EOS/CircleSnake3/data/EoE'
TRAIN_PATH = os.path.join(DATASET_ROOT, 'train')
VAL_PATH = os.path.join(DATASET_ROOT, 'val')
TEST_PATH = os.path.join(DATASET_ROOT, 'test')
TRAIN_JSON = os.path.join(DATASET_ROOT, 'EoE_train2022.json')
VAL_JSON = os.path.join(DATASET_ROOT, 'EoE_val2022.json')
TEST_JSON = os.path.join(DATASET_ROOT, 'EoE_test2022.json')

PREDEFINED_SPLITS_DATASET = {
    "train": (TRAIN_PATH, TRAIN_JSON),
    "val": (VAL_PATH, VAL_JSON),
    "test": (TEST_PATH, TEST_JSON),
}
#
DatasetCatalog.register("train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
DatasetCatalog.register("val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
DatasetCatalog.register("test", lambda: load_coco_json(TEST_JSON, TEST_PATH))


monuseg_train_metadata = MetadataCatalog.get("train").set(thing_classes=CLASS_NAMES,
                                                    evaluator_type='coco',
                                                    json_file=TRAIN_JSON,
                                                    image_root=TRAIN_PATH)

monuseg_val_metadata = MetadataCatalog.get("val").set(thing_classes=CLASS_NAMES,
                                                    evaluator_type='coco',
                                                    json_file=VAL_JSON,
                                                    image_root=VAL_PATH)
monuseg_test_metadata = MetadataCatalog.get("test").set(thing_classes=CLASS_NAMES,
                                                    evaluator_type='coco',
                                                    json_file=TEST_JSON,
                                                    image_root=TEST_PATH)
dataset_dicts = DatasetCatalog.get("test")


#  Output ground truth:
import random
from detectron2.utils.visualizer import Visualizer
for i, d in enumerate(dataset_dicts):
    file_name = d["file_name"]
    filename = os.path.basename(file_name)
    filename_without_extension = os.path.splitext(filename)[0]
    img = cv2.imread(file_name)
    visualizer = Visualizer(img[:, :, ::-1], metadata=monuseg_test_metadata, scale=1.0)
    vis = visualizer.draw_dataset_dict(d)
    out_path = "/home/VANDERBILT/liuy99/Documents/detectron2/output/ground_truth/{0}.png".format(filename_without_extension)
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    cv2.imwrite(out_path, vis.get_image()[:, :, ::-1])

# implement `build_evaluator` method.
class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

#
# class MyTrainer(DefaultTrainer):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         self.checkpointer = DetectionCheckpointer(
#             self.model,
#             self.cfg.OUTPUT_DIR,
#             save_to_disk=True,
#             **self.checkpointer_args
#         )
#         self.periodic_checkpointer = PeriodicCheckpointer(
#             self.checkpointer,
#             self.cfg.SOLVER.CHECKPOINT_PERIOD,
#             **self.checkpointer_args
#         )
#
#     def build_hooks(self):
#         hooks = super().build_hooks()
#         hooks.insert(-1, MyEvalHook(self.cfg.TEST.EVAL_PERIOD, self.periodic_checkpointer))
#         return hooks
#
# class MyEvalHook:
#     def __init__(self, eval_period, periodic_checkpointer):
#         self.eval_period = eval_period
#         self.periodic_checkpointer = periodic_checkpointer
#         self.best_val_loss = float("inf")
#
#     def after_step(self):
#         iteration = self.trainer.iter
#         if iteration % self.eval_period == 0:
#             losses = self.trainer.evaluator.evaluate()
#             loss = losses["total_loss"]
#             if loss < self.best_val_loss:
#                 self.best_val_loss = loss
#                 self.periodic_checkpointer.save("best_model")

#
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("train")
cfg.DATASETS.TEST = ("test",)
cfg.DATALOADER.NUM_WORKERS = 16
# Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 16
# cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.TEST.DETECTIONS_PER_IMAGE = 100
cfg.TEST.EVAL_PERIOD = 100
cfg.OUTPUT_DIR = "/home/VANDERBILT/liuy99/Documents/detectron2/output"
#
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()

from detectron2.evaluation import COCOEvaluator
model = trainer.build_model(cfg)
from detectron2.checkpoint import DetectionCheckpointer
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
evaluator = COCOEvaluator("test", ("bbox", "segm"), False, output_dir=cfg.OUTPUT_DIR, use_fast_impl=False)
trainer.test(cfg, model, evaluators=evaluator)

# Run inference on the test dataset
data_loader = build_detection_test_loader(cfg, "test")
results = inference_on_dataset(trainer.model, data_loader, evaluator)
# Print the Average Precision (AP) for each class
ap = results["bbox"]
for class_name, ap_value in ap.items():
    print(f"{class_name}: {ap_value}")

# visualize the data
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set the testing threshold for this model
cfg.MODEL.WEIGHTS = "/home/VANDERBILT/liuy99/Documents/detectron2/output/model_final.pth"
test_metadata = MetadataCatalog.get("test")
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer, VisImage
import glob
import re
from detectron2.structures import Instances

img_list = glob.glob('/home/VANDERBILT/liuy99/Documents/EOS/CircleSnake3/data/EoE/test/*.jpg')
import cv2
import numpy as np

for i in range(len(img_list)):
    image_file = img_list[i]
    filename = os.path.basename(image_file)
    filename_without_extension = os.path.splitext(filename)[0]
    img: np.ndarray = cv2.imread(image_file)
    output = predictor(img)["instances"]

    v = Visualizer(img[:, :, ::-1],
                   metadata=test_metadata,
                   scale=1.0)

    # Draw polygons around the predicted instances and display the confidence level
    result_image = img.copy()
    for mask, category, score in zip(output.pred_masks, output.pred_classes, output.scores):
        mask = mask.cpu().numpy()

        # Convert the mask to polygons
        contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Set custom colors based on the category
        if category == 0:
            color = (0, 255, 0)  # green
        elif category == 1:
            color = (0, 255, 255)  # blue
        elif category == 2:
            color = (255, 0, 255)  # pink
        elif category == 3:
            color = (255, 255, 0)  # yellow

        # Draw the polygons
        cv2.polylines(result_image, contours, True, color, 2)

        # Draw the confidence level
        text = f"{score:.2f}"
        x1, y1 = contours[0][0, 0]
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(result_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1 - 5), color, -1)
        cv2.putText(result_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    out_path = os.path.join(cfg.OUTPUT_DIR, "predicted/{0}.png".format(filename_without_extension))
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    cv2.imwrite(out_path, result_image)

# for i in range(len(img_list)):
#     image_file = img_list[i]
#     filename = os.path.basename(image_file)
#     filename_without_extension = os.path.splitext(filename)[0]
#     img: np.ndarray = cv2.imread(image_file)
#     output: Instances = predictor(img)["instances"]
#
#     v = Visualizer(img[:, :, ::-1],
#                    metadata=test_metadata,
#                    scale=1.0)
#     result: VisImage = v.draw_instance_predictions(output.to("cpu"))
#     result_image: np.ndarray = result.get_image()[:, :, ::-1]
#
#     out_path = os.path.join(cfg.OUTPUT_DIR, "predicted/{0}.png".format(filename_without_extension))
#     if not os.path.exists(os.path.dirname(out_path)):
#         os.makedirs(os.path.dirname(out_path))
#     cv2.imwrite(out_path, result_image)


# import json
# def load_json_arr(json_path):
#     lines = []
#     with open(json_path, 'r') as f:
#         for line in f:
#             lines.append(json.loads(line))
#     return lines
#
# experiment_metrics = load_json_arr('/home/VANDERBILT/liuy99/Documents/detectron2/output/metrics.json')
# import matplotlib.pyplot as plt
# plt.plot(
#     [x['iteration'] for x in experiment_metrics],
#     [x['total_loss'] for x in experiment_metrics])
# plt.plot(
#     [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
#     [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
# plt.legend(['total_loss', 'validation_loss'], loc='upper left')
# plt.show()

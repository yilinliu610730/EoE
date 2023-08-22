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

# Register MoNuSeg dataset
from detectron2.data.datasets import register_coco_instances
register_coco_instances("train", {}, "/home/ethan/Documents/CircleSnake/data/kidpath_coco/train_circle.json", "/home/ethan/Documents/CircleSnake/data/kidpath_coco/train")
register_coco_instances("val", {}, "/home/ethan/Documents/CircleSnake/data/kidpath_coco/validate_circle.json", "/home/ethan/Documents/CircleSnake/data/kidpath_coco/validate")
register_coco_instances("test", {}, "/home/ethan/Documents/CircleSnake/data/kidpath_coco/test_circle.json", "/home/ethan/Documents/CircleSnake/data/kidpath_coco/test")

# # visualize training data
monuseg_train_metadata = MetadataCatalog.get("test")
dataset_dicts = DatasetCatalog.get("test")

# import random
# from detectron2.utils.visualizer import Visualizer
# for i, d in enumerate(dataset_dicts):
#     file_name = d["file_name"]
#     img = cv2.imread(file_name)
#     visualizer = Visualizer(img[:, :, ::-1], metadata=monuseg_train_metadata, scale=1.0)
#     vis = visualizer.draw_dataset_dict(d)
#     out_path = "/home/ethan/Documents/detectron2/datasets/test/{0}.png".format(i)
#     cv2.imwrite(out_path, vis.get_image()[:, :, ::-1])

from detectron2.engine import DefaultTrainer
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train", "val")
cfg.DATASETS.TEST = ("test",)
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 50000
cfg.SOLVER.CHECKPOINT_PERIOD = 500
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.TEST.DETECTIONS_PER_IMAGE = 100
cfg.OUTPUT_DIR = "/home/ethan/Documents/detectron2/projects/MoNuSeg/glomeruli_clean"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()

from detectron2.evaluation import COCOEvaluator
model = trainer.build_model(cfg)
evaluator = COCOEvaluator("test", ("bbox", "segm"), False, output_dir=cfg.OUTPUT_DIR, use_fast_impl=False)
trainer.test(cfg, model, evaluators=evaluator)
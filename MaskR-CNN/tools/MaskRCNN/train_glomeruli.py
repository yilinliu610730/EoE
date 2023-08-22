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
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer

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
    cfg.DATALOADER.NUM_WORKERS = 16
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = "/home/ethan/Downloads/model_final_a3ec72.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 20000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.OUTPUT_DIR = "/home/ethan/Documents/detectron2/projects/MoNuSeg/mask_r101_glomeruli_pretrain"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg

if __name__ == "__main__":
    # Register datasets
    register_datasets()

    # Setup cfg
    cfg = setup_cfg()

    # Train
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

    # # Test
    # model = trainer.build_model(cfg)
    # evaluator = COCOEvaluator("test", ("bbox", "segm"), False, output_dir=cfg.OUTPUT_DIR, use_fast_impl=False)
    # trainer.test(cfg, model, evaluators=evaluator)
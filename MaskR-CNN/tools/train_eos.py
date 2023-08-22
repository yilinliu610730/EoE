#引入以下注释
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import pycocotools
import logging
import cv2
import numpy as np
import os, json, cv2, random
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer, VisImage
import glob
import re
from detectron2.structures import Instances

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
import os
from collections import OrderedDict
import torch
from detectron2 import model_zoo
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


#声明类别，尽量保持
CLASS_NAMES =["eos","papilla","rbc", "cluster"]
# 数据集路径
DATASET_ROOT = '/home/VANDERBILT/liuy99/Documents/EOS/CircleSnake3/data/EoE'
# ANN_ROOT = os.path.join(DATASET_ROOT, 'COCOformat')

TRAIN_PATH = os.path.join(DATASET_ROOT, 'train')
VAL_PATH = os.path.join(DATASET_ROOT, 'val')
TEST_PATH = os.path.join(DATASET_ROOT, 'test')

TRAIN_JSON = os.path.join(DATASET_ROOT, 'EoE_train2022.json')
VAL_JSON = os.path.join(DATASET_ROOT, 'EoE_val2022.json')
TEST_JSON = os.path.join(DATASET_ROOT, 'EoE_test2022.json')

# 声明数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "coco_my_train": (TRAIN_PATH, TRAIN_JSON),
    "coco_my_val": (TEST_PATH, TEST_JSON),
}

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



def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   json_file=json_file,
                                   image_root=image_root)



def register_dataset_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco")


def plain_register_dataset():

    DatasetCatalog.register("coco_my_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("coco_my_train").set(thing_classes=CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                    evaluator_type='coco', # 指定评估方式
                                                    json_file=TRAIN_JSON,
                                                    image_root=TRAIN_PATH)


    DatasetCatalog.register("coco_my_val", lambda: load_coco_json(TEST_JSON, TEST_PATH))
    MetadataCatalog.get("coco_my_val").set(thing_classes=CLASS_NAMES, # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                evaluator_type='coco', # 指定评估方式
                                                json_file=TEST_JSON,
                                                image_root=TEST_PATH)

def checkout_dataset_annotation(name="coco_my_val"):
    #dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
    # dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH)
    dataset_dicts = DatasetCatalog.get(name)
    print(len(dataset_dicts))
    for i, d in enumerate(dataset_dicts):
        #print(d)
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.0)
        vis = visualizer.draw_dataset_dict(d)
        #cv2.imshow('show', vis.get_image()[:, :, ::-1])
        # cv2.imwrite('out/'+str(i) + '.jpg',vis.get_image()[:, :, ::-1])
        #cv2.waitKey(0)
        out_path = "/home/VANDERBILT/liuy99/Documents/detectron2/output1/{0}.png".format(i)
        cv2.imwrite(out_path, vis.get_image()[:, :, ::-1])
        # cv2.imshow('show', vis.get_image()[:, :, ::-1])


def checkout_dataset_annotation1(cfg,name="coco_my_val"):
    # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get(name)
    print(len(dataset_dicts))
    for i, d in enumerate(dataset_dicts):
        img: np.darray = cv2.imread(d["file_name"])
        output: Instances = predictor(img)["instances"]
        v = Visualizer(img[:, :, ::-1],
                       metadata=MetadataCatalog.get(name),
                       scale=0.8)
        result: VisImage = v.draw_instance_predictions(output.to("cpu"))
        result_image: np.ndarray = result.get_image()[:, :, ::-1]

        out_path = "/home/VANDERBILT/liuy99/Documents/detectron2/output1/{0}.png".format(i)
        cv2.imwrite(out_path, result_image)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.merge_from_list(args.opts)          # 从CLI参数 覆盖配置

    # 更改配置参数
    cfg.DATASETS.TRAIN = ("coco_my_train",) # 训练数据集名称
    cfg.DATASETS.TEST = ("coco_my_val",)
    cfg.DATALOADER.NUM_WORKERS = 4  # 单线程

    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.MAX_SIZE_TRAIN = 256 # 训练图片输入的最大尺寸
    cfg.INPUT.MAX_SIZE_TEST = 256 # 测试数据输入的最大尺寸
    cfg.INPUT.MIN_SIZE_TRAIN = (256, 256) # 训练图片输入的最小尺寸，可以设定为多尺度训练
    cfg.INPUT.MIN_SIZE_TEST = 256
    #cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING，其存在两种配置，分别为 choice 与 range ：
    # range 让图像的短边从 512-768随机选择
    #choice ： 把输入图像转化为指定的，有限的几种图片大小进行训练，即短边只能为 512或者768
    # cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'
#  本句一定要看下注释！！！！！！！！
    cfg.MODEL.RETINANET.NUM_CLASSES = 4  # 类别数+1（因为有background，也就是你的 cate id 从 1 开始，如果您的数据集Json下标从 0 开始，这个改为您对应的类别就行，不用再加背景类！！！！！）
    #cfg.MODEL.WEIGHTS="/home/yourstorePath/.pth"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 4  # batch_size=2; iters_in_one_epoch = dataset_imgs/batch_size
    cfg.OUTPUT_DIR = "/output1"
    # 根据训练数据总数目以及batch_size，计算出每个epoch需要的迭代次数
    #9000为你的训练数据的总数目，可自定义
    ITERS_IN_ONE_EPOCH = int(9000 / cfg.SOLVER.IMS_PER_BATCH)

    # 指定最大迭代次数
    cfg.SOLVER.MAX_ITER = 200 # 12 epochs，
    # 初始学习率
    cfg.SOLVER.BASE_LR = 0.002
    # 优化器动能
    cfg.SOLVER.MOMENTUM = 0.9
    #权重衰减
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    # 学习率衰减倍数
    cfg.SOLVER.GAMMA = 0.1
    # 迭代到指定次数，学习率进行衰减
    cfg.SOLVER.STEPS = (7000,)
    # 在训练之前，会做一个热身运动，学习率慢慢增加初始学习率
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    # 热身迭代次数
    cfg.SOLVER.WARMUP_ITERS = 1000

    cfg.SOLVER.WARMUP_METHOD = "linear"
    # 保存模型文件的命名数据减1
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    # 迭代到指定次数，进行一次评估
    cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH
    #cfg.TEST.EVAL_PERIOD = 100

    #cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    plain_register_dataset()
    # checkout_dataset_annotation()
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        checkout_dataset_annotation1(cfg)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


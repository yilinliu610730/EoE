#!/usr/bin/env python3

import datetime
import json
import os
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from utils import filter_for_annotations, filter_for_jpeg, walklevel

ROOT_DIR = "/home/VANDERBILT/liuy99/Documents/EOS/train"
IMAGE_DIR = os.path.join(ROOT_DIR, "tiles")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "tiles")

INFO = {
    "description": "EoE Dataset",
    "url": "https://github.com/yilinliu610730/EOS",
    "version": "0.1.0",
    "year": 2022,
    "contributor": "Yilin Liu",
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}

LICENSES = [
    {"id": 1, "name": "Vanderbilt University - HRLB Lab - Dr. Yuankai Huo", "url": "NA"}
]

CATEGORIES = [
    {"id": 1, "name": "eos", "supercategory": "Class",},
    {"id": 2, "name": "papilla", "supercategory": "Class",},
    {"id": 3, "name": "rbc", "supercategory": "Class",},
    {"id": 4, "name": "cluster", "supercategory": "Class",},
]


def main():
    sublist = {}
    sublist["train"] = ["P18_1908_S2"]
    sublist["val"] = []
    sublist["test"] = []
    types = ["train", "val", "test"]

    for type in types:
        image_id = 1
        segmentation_id = 1

        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": [],
        }

        # filter for jpeg images
        for root, folders, files in os.walk(IMAGE_DIR, 0):
            json_file = os.path.join(ROOT_DIR, "json/Eos_%s2022.json" % type)

            # go through each scn image
            for image_folder in folders:
                images_path = os.path.join(IMAGE_DIR, image_folder)
                for _, _, images in os.walk(images_path, 0):
                    # image_files = filter_for_jpeg(image_path, files)
                    # image_path = os.path.join(IMAGE_DIR, image_folder, os.path.basename(folder_path)+'.png')
                    for filename in images:
                        if filename.endswith('.jpg') and filename.split(' ')[0] in sublist[type]:

                            image_path = os.path.join(images_path, filename)
                            image = Image.open(image_path)
                            image_info = pycococreatortools.create_image_info(
                                image_id, filename, image.size
                            )

                            coco_output["images"].append(image_info)
                            print(image_info)

                            ANNOTATION_DIR = os.path.join(IMAGE_DIR, image_folder)
                            # filter for associated png annotations
                            for root, _, files in os.walk(ANNOTATION_DIR):
                                annotation_files = filter_for_annotations(root, files, filename)
                                print(annotation_files)

                                # go through each associated annotation
                                for annotation_filename in annotation_files:
                                    if "Eos" in annotation_filename:
                                        class_id = 1
                                    elif "Papilla" in annotation_filename:
                                        class_id = 2
                                    elif "RBC" in annotation_filename:
                                        class_id = 3
                                    elif "Cluster" in annotation_filename:
                                        class_id = 4
                                    else:
                                        raise RuntimeError

                                    category_info = {
                                        "id": class_id,
                                        "is_crowd": "crowd" in annotation_filename,
                                    }
                                    binary_mask = np.asarray(
                                        Image.open(os.path.join(ANNOTATION_DIR, annotation_filename)).convert("1")
                                    ).astype(np.uint8)

                                    annotation_info = pycococreatortools.create_annotation_info(
                                        segmentation_id,
                                        image_id,
                                        category_info,
                                        binary_mask,
                                        image.size,
                                        tolerance=2,
                                    )
                                    if annotation_info is not None:
                                        coco_output["annotations"].append(annotation_info)

                                    segmentation_id = segmentation_id + 1

                            image_id = image_id + 1

        # print(coco_output)
        # with open(
        #     os.path.join(ROOT_DIR, "eos_coco.json"), "w"
        # ) as output_json_file:
        #     json.dump(coco_output, output_json_file)
        with open(json_file.format(ROOT_DIR), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)

        print(f"SUCCESS: created coco file in {ROOT_DIR}")


if __name__ == "__main__":
    main()

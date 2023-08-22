#!/usr/bin/env python3

import datetime
import json
import os
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from utils import filter_for_annotations, filter_for_jpeg, walklevel

ROOT_DIR = "YOUR_ROOT_DIRECTORY"
IMAGE_DIR = ROOT_DIR
ANNOTATION_DIR = ROOT_DIR
JSON_DIR = "YOUR_JSON_DIRECTROY"

INFO = {
    "description": "EoE Dataset",
    "url": "YOUR_URL",
    "version": "0.1.0",
    "year": 2022,
    "contributor": "Yilin Liu",
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}

LICENSES = [
    {"id": 1, "name": "YOUR_LICENSE"}
]

CATEGORIES = [
    {"id": 0, "name": "eos", "supercategory": "Class",},
    {"id": 1, "name": "papilla", "supercategory": "Class",},
    {"id": 2, "name": "rbc", "supercategory": "Class",},
    {"id": 3, "name": "cluster", "supercategory": "Class",},
]


def main():
    sublist = {}
    sublist["train"] = ["P18_1340_S2", "..."]
    sublist["val"] = ["P17_5219_S6", "..."]
    sublist["test"] = ["P17_8000_S2", "..."]

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
        for image_folder in sublist[type]:
            json_file = os.path.join(JSON_DIR, "EoE_%s2022.json" % type)
            images_path = os.path.join(IMAGE_DIR, image_folder)
            for _, _, images in os.walk(images_path, 0):
                # image_files = filter_for_jpeg(image_path, files)
                # image_path = os.path.join(IMAGE_DIR, image_folder, os.path.basename(folder_path)+'.png')
                for filename in images:
                    if filename.endswith('.jpg'):
                        image_path = os.path.join(images_path, filename)
                        image = Image.open(image_path)
                        image_info = pycococreatortools.create_image_info(
                            image_id, filename, image.size
                        )

                        # coco_output["images"].append(image_info)

                        ANNOTATION_DIR = os.path.join(IMAGE_DIR, image_folder)
                        annotation_files = []
                        # filter for associated png annotations
                        for root, _, files in os.walk(ANNOTATION_DIR):
                            annotation_files = filter_for_annotations(root, files, filename)
                            print(annotation_files)

                            # go through each associated annotation
                            for annotation_filename in annotation_files:
                                if "Papilla" in annotation_filename:
                                    class_id = 1
                                elif "Eos" in annotation_filename:
                                    class_id = 0
                                elif "Cluster" in annotation_filename:
                                    class_id = 3
                                elif "RBC" in annotation_filename:
                                    class_id = 2
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
                        if annotation_files:
                            coco_output["images"].append(image_info)
                            image_id = image_id + 1

        # print(coco_output)
        # with open(
        #     os.path.join(ROOT_DIR, "eos_coco.json"), "w"
        # ) as output_json_file:
        #     json.dump(coco_output, output_json_file)
        with open(json_file.format(JSON_DIR), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)

        print(f"SUCCESS: created coco file in {json_file.format(ROOT_DIR)}")


if __name__ == "__main__":
    main()

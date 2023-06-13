# EoE

## Overview

Eosinophilic Esophagitis (EoE) is a  chronic immune-mediated inflammatory disease of the esophagus identified by an unusual amount of eosinophils(eosinophilia) in the esophageal tissue. We present a computer solution for the identification and analysis of Eosinophilic Esophagitis (EoE) -- CircleSnake -- that achieved multi-class identification of the presence and numbers of eosinophils in entire biopsy Whole-Slide Images (WSI). From the results, the CircleSnake model yields higher average precision score than the traditional train-from scratch counterpart (Mask R-CNN model). The trained also achieved to distinguish four different types of cells similar to EoE.

Contact: yilin.liu@vanderbilt.edu.

## Installation
Initial setup: _Clone the repository, create your own branch for development, and install dependencies_

  ```sh
  git clone <EoE>
  
  git checkout -b <branch_name>
  
  ???INSTALL.md???
  ```

## Data Preparation Pipeline

   1. Image Annotations

      Annotations are done using the **QuPath** Software.

   2. Image Segmentation
      - **json_to_contour.py**
        > this will converted the json file above to contour and cut contour into 512*512 patches
      
      
      
      - **class_to_coco.py**
        > this will read in the contour patches we just create and convert it to a COCO file
      
      
      
      - **coco_to_circlenet.py**  
        > this will convert a COCO file to the CircleSnake compatible version

  3. Dataset Splitting
  
  4. Model Training and Testing

     - [CircleSnake](https://github.com/hrlblab/CircleSnake)
     
     - [Mask R-CNN](https://github.com/facebookresearch/detectron2)


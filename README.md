# EoE

## Steps For Data Preparation Pipeline:
- **json_to_contour.py**
  > this will converted the json file above to contour and cut contour into 512*512 patches



- **class_to_coco.py**
  > this will read in the contour patches we just create and convert it to a COCO file



- **coco_to_circlenet.py**  
  > this will convert a COCO file to the CircleNet compatible version



- **Run the CircleSnake model**

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2 # import after setting OPENCV_IO_MAX_IMAGE_PIXELS
import numpy as np
import json
from matplotlib import pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def json_to_contour():
    image_dir = '/home/VANDERBILT/liuy99/Documents/EOS/train/images'
    json_dir = '/home/VANDERBILT/liuy99/Documents/EOS/train/converted_json'
    output_dir = '/home/VANDERBILT/liuy99/Documents/EOS/train/annotation_patches'
    cnt = 1

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            print(os.path.join(image_dir, filename))
            image = Image.open(os.path.join(image_dir, filename))

            # Convert image to array
            image = np.array(image)
            # print(image.shape)
            # image = np.zeros(image.shape, np.uint8)
            # image = (image * 255).astype(np.uint8)

            name = filename.split('.')[0]
            print(name)
            print(os.path.join(json_dir, name + '.json'))

            # Opening JSON file
            f = open(os.path.join(json_dir, name + '.json'))

            # returns JSON object as
            # a dictionary
            data = json.load(f)

            # Iterating through the json
            # list
            for i in range(0, len(data)):
                print(i)
                for coord in data[i]['geometry']['coordinates']:
                    # reset image
                    mask = np.zeros(image.shape, np.uint8)
                    mask = (mask * 255).astype(np.uint8)
                    classification = data[i]['properties']['classification']['name']

                    # print(coord)
                    contours = np.array(coord)
                    mincoord = (np.amin(coord, axis = 0) + np.amax(coord, axis = 0)) / 2
                    # print(np.amin(coord, axis = 0))
                    # print(np.amax(coord, axis=0))
                    # print(mincoord)

                    print(contours)
                    # draw contours on both mask and image
                    max = np.max(mask)
                    revise = contours.reshape(-1, 1, 2).astype(np.int32)
                    mask = cv2.drawContours(mask, [contours.reshape(-1, 1, 2).astype(np.int32)], -1, (255, 255, 255), -1)
                    image = cv2.drawContours(image, [contours.reshape(-1, 1, 2).astype(np.int32)], -1, (255, 255, 255), -1)

                    # find where the contour is
                    index = np.where(image==255)

                    # cut the contour into patches
                    cut_mask(mask, mincoord[0], mincoord[1], cnt, classification)
                    cut_image(image, mincoord[0], mincoord[1], cnt, classification)

                    cnt += 1

                # Closing file
                f.close()


def cut_image(contour, xmin, ymin, cnt, classification, width=512, height=512):
    # Load image

    directory = '/home/VANDERBILT/liuy99/Documents/EOS/train/tiles'
    output_dir = '/home/VANDERBILT/liuy99/Documents/EOS/train/tiles_'
    categories = ['']

    for (root, dirs, _) in os.walk(directory, topdown=True):
        for scn in dirs:
            dir = os.path.join(root, scn)
            for filename in os.listdir(dir):
                # if filename.endswith(".jpg"):
                patch_x = int(filename.split(',')[0].split('=')[1])
                patch_y = int(filename.split(',')[1].split('=')[1])

                if not(xmin > patch_x and xmin < patch_x + width and ymin > patch_y and ymin < patch_y + height):
                    continue

                # Crop image
                contour_patch = contour[patch_y:patch_y+height, patch_x:patch_x+width]

                # Convert array to image
                contour_patch = Image.fromarray(contour_patch)

                # Save the cropped image
                if len(filename.split('_')) == 3:
                    name = filename.split('.')[0]
                    contour_patch.save(os.path.join(dir, f'{name}.jpg'))

                # # Display image
                # image.imshow()

                cnt = cnt + 1

def cut_mask(contour, xmin, ymin, cnt, classification, width=512, height=512):
    # Load image

    directory = '/home/VANDERBILT/liuy99/Documents/EOS/train/tiles'
    output_dir = '/home/VANDERBILT/liuy99/Documents/EOS/train/tiles'
    categories = ['']

    for (root, dirs, _) in os.walk(directory, topdown=True):
        for scn in dirs:
            dir = os.path.join(root, scn)
            for filename in os.listdir(dir):
                # if filename.endswith(".jpg"):
                patch_x = int(filename.split(',')[0].split('=')[1])
                patch_y = int(filename.split(',')[1].split('=')[1])

                if not(xmin > patch_x and xmin < patch_x + width and ymin > patch_y and ymin < patch_y + height):
                    continue

                # Crop image
                contour_patch = contour[patch_y:patch_y+height, patch_x:patch_x+width]

                # Convert array to image
                contour_patch = Image.fromarray(contour_patch)

                # Save the cropped image
                if len(filename.split('_')) == 3:
                    name = filename.split('.')[0]
                    contour_patch.save(os.path.join(dir, f'{name}_{cnt}_{classification}.png'))

                # # Display image
                # image.imshow()

                cnt = cnt + 1

if __name__ == "__main__":
    json_to_contour()
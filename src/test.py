import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data.plant import PlantDataset, PlantConfig
import argparse

# Root directory of the project
ROOT_DIR = os.path.abspath("../../Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import skimage


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--img_path')
    parser.add_argument('--data_dir', default='/Tim/Projects/plant-instance-seg/data')
    args = parser.parse_args()
    return args


class InferenceConfig(PlantConfig):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def detect(img_path, model, config):
    image = skimage.io.imread(img_path)
    image, _, _, _, _ = utils.resize_image(
        image, config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM, config.IMAGE_MIN_SCALE, mode=config.IMAGE_RESIZE_MODE)
    results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], ax=ax,
                                title="Predictions")
    plt.show()


if __name__ == '__main__':
    args = parse_args()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    config = InferenceConfig()
    config.display()

    dataset = PlantDataset()
    dataset.load_coco(args.data_dir, "val")
    dataset.prepare()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(args.model, by_name=True)

    for i in range(5):
        detect(args.img_path, model, config)



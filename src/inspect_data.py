import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.extend(['/Tim/Projects/Mask_RCNN', '/Tim/Projects/plant-instance-seg'])

from data.plant import PlantDataset, PlantConfig
from mrcnn import utils, visualize
from mrcnn.model import log

DATA_DIR = '/Tim/Projects/plant-instance-seg/data'

if __name__ == '__main__':
    dataset_config = PlantConfig()
    dataset = PlantDataset()
    dataset.load_coco(DATA_DIR, 'train')
    dataset.prepare()

    print("Image Count: {}".format(len(dataset.image_ids)))
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    # image_ids = np.random.choice(dataset.image_ids, 4)
    # for image_id in image_ids:
    #     image = dataset.load_image(image_id)
    #     mask, class_ids = dataset.load_mask(image_id)
    #
    #     print(mask.shape)
    #     print(class_ids.shape)
    #     print(dataset.image_info[0])
    #     visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

    # Load random image and mask.
    image_id = np.random.choice(dataset.image_ids, 1)[0]
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    # Resize
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=dataset_config.IMAGE_MIN_DIM,
        max_dim=dataset_config.IMAGE_MAX_DIM,
        mode=dataset_config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop=crop)
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)

    # Display image and additional stats
    print("image_id: ", image_id, dataset.image_reference(image_id))
    print("Original shape: ", original_shape)
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
    # Display image and instances
    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)


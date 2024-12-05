import os
import pathlib

if "models" in pathlib.Path.cwd().parts:
    while "models" in pathlib.Path.cwd().parts:
        os.chdir('..')
else:
	os.system('git clone https://github.com/tensorflow/models.git')

os.chdir('models/research')
os.system('protoc object_detection/protos/*.proto --python_out=.')
os.chdir('../..')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def load_model(model_name):
    model_dir = pathlib.Path(model_name) / "saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'training/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

model_name = 'model'
detection_model = load_model(model_name)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, image_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

    display(Image.fromarray(image_np))


import cv2


def bluring(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    output = run_inference_for_single_image(model, img)
    classes = output['detection_classes']
    boxes = output['detection_boxes']


    chats_idx = np.where(classes == 1)[0]

    im_width = img.shape[0]
    im_height = img.shape[1]
    for idx in chats_idx:
        x, y, w, h = boxes[idx]
        x *= im_width
        w *= im_width
        y *= im_height
        h *= im_height
        x, w, y, h = int(x), int(w), int(y), int(h)
        img[int(x):int(w), int(y):int(h)] = cv2.blur(img[int(x):int(w), int(y):int(h)], (50, 50))

    return img

def process_image(path_to_img_with_extension, path_to_save_img_with_extension):
    PATH_TO_IMAGE = path_to_img_with_extension
    blured_image = bluring(detection_model, PATH_TO_IMAGE)
    cv2.imwrite(path_to_save_img_with_extension, cv2.cvtColor(blured_image, cv2.COLOR_BGR2RGB))

process_image('test3.jpg', 'blured1.png')

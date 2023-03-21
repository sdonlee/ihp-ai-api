
import numpy as np
import os
import sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from matplotlib import pyplot as plt
from PIL import Image
# from IPython.display import display

import pathlib

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import pandas as pd
import glob
import xml.etree.ElementTree as ET

import copy
import typing

import cv2

# import os
# path_torch = '/root/ENService/ai/obj_det/pytorch'
# if path_torch not in os.environ['PATH'].split(':'):
#     os.environ['PATH'] += f':{path_torch}'
# path_tmp = '/opt/hpcx/ompi/bin'
# if path_tmp not in os.environ['PATH'].split(':'):
#     os.environ['PATH'] += f':{path_tmp}'
# path_tmp = '/opt/hpcx/ompi/lib'
# if path_tmp not in os.environ['LD_LIBRARY_PATH'].split(':'):
#     os.environ['LD_LIBRARY_PATH'] += f':{path_tmp}'

# exec env LD_LIBRARY_PATH=/opt/hpcx/ompi/lib python -x "$0" "$@"


#########################################################################################################################
#########################################################################################################################

def load_model(model_path: str = 'yolov7.pt'):
    
    model = tf.saved_model.load(model_path)
    model = model.signatures['serving_default']
        
    return model


def load_label(path_to_label:str, model_fn=''):
    
    category_index = label_map_util.create_category_index_from_labelmap(path_to_label, use_display_name=True)
    
    return category_index


def det_single_frm(image:np.array, model_fn, conf_thres:float=.7, iou_thres:float=0.45):
    
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    output_dict = model_fn(input_tensor)
    
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
        for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    
    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict


def draw_box( img_in:np.array, det_dict, category_index, conf_thres, labelcolors, line_thickness:int=1):
    
    vis_util.visualize_boxes_and_labels_on_image_array(
        img_in,
        det_dict['detection_boxes'],
        det_dict['detection_classes'],
        det_dict['detection_scores'],
        category_index,
        min_score_thresh = conf_thres,
        instance_masks = det_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=line_thickness)

    return img_in





#########################################################################################################################
#########################################################################################################################



if __name__ =="__main__":
    
    # List of the strings that is used to add correct label for each box.
    PATH_TO_MODELS = f'/root/ENService/ai/med/obj_det/tf/models/fasterrcnn/saved_model'
    PATH_TO_LABELS = f'/root/ENService/ai/med/obj_det/tf/models/fasterrcnn/label/label_map.pbtxt'
    PATH_TO_VIDEOS = f'/root/ENService/video_test/test_videos'
    
    category_index = load_label(path_to_label=PATH_TO_LABELS)
    
    print([category_index[key]['name'] for key in list(category_index.keys())])

    ########################################
    ### MODEL LOAD
    ########################################

    detection_model = load_model(PATH_TO_MODELS)
    len(detection_model.inputs)
    detection_model.inputs[0].shape
    detection_model._output_shapes
    
    videolist = glob.glob(PATH_TO_VIDEOS+'/*.mp4')
    
    for pathtmp in ['hand_main', 'hand_sub', 'no_medi']:
        vid_path = pathlib.Path(f'/root/src/videos/{pathtmp}')
        # img_path = pathlib.Path('/root/src/images')
        # lbl_path = pathlib.Path('/root/src/labels')

        ########################################
        ### DATA LOAD
        ########################################

        # data_names = load_imgs(img_path, lbl_path, nDataSmp=100)
        data_names = load_vids(vid_path)


        # print(detection_model.signatures['serving_default'].inputs)
        # detection_model.signatures['serving_default'].output_dtypes
        # detection_model.signatures['serving_default'].output_shapes

        ########################################
        ### INFERENCE
        ########################################

        for video_name in data_names:
            show_inference_vid(detection_model, vid_path, video_name, category_index, save=True)
        
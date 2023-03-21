import argparse
import glob
import numpy as np
import cv2
# import torch.backends.cudnn as cudnn

import sys
path_torch = '/'.join(__file__.split('/')[:-1] + ['lib'])
if path_torch not in sys.path:
    sys.path.insert(0,path_torch)

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

import torch

from ai.med.obj_det.pytorch.lib.models.experimental import model_load
from ai.med.obj_det.pytorch.lib.utils.general import check_img_size, non_max_suppression, scale_coords 
from ai.med.obj_det.pytorch.lib.utils.plots import plot_one_box
from ai.med.obj_det.pytorch.lib.utils.torch_utils import select_device, time_synchronized

fraction =0.9
torch.cuda.set_per_process_memory_fraction(fraction, device=None)

#########################################################################################################################
#########################################################################################################################

def load_model(model_path: str = 'yolov7.pt'):

    device = select_device()
    model = model_load(model_path, map_location=device)
    model.half()  # to FP16
        
    return model


def load_label(path_to_label:str, model_fn=''):
    
    category_index = dict()
    for ilabel, label in enumerate(model_fn.module.names if hasattr(model_fn, 'module') else list(model_fn.names.values())):
        category_index[ilabel+1] = dict()
        category_index[ilabel+1] = {'id':ilabel+1, 'name':label }
    
    return category_index


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border


    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # GPU
    device = select_device()
    img = torch.from_numpy(img).to(device)
    img = img.half() # uint8 to fp16
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img, ratio, (dw, dh)



def det_single_frm(image:np.array, model_fn, conf_thres:float=.7, iou_thres:float=0.45):
    
    # device = select_device()
    new_shape = 320
    stride = int(model_fn.stride.max())
    model_input, ratio, (dw, dh) = letterbox(image, new_shape, stride=stride)

    # model_input = frm_trans(image)
    
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model_fn(model_input, augment=True)[0]
    
    # conf_thres = 0.25
    # iou_thres = 0.45
    
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
    
    hw1 = [model_input.shape[2],model_input.shape[3]]
    hw0 = [image.shape[0],image.shape[1]]
    
    boxes_ = scale_coords(hw1, pred[0][:,:4], hw0).round().detach().cpu().numpy()
    score_ = pred[0][:,4].detach().cpu().numpy()
    class_ = pred[0][:,5].detach().cpu().numpy()
    
    output_dict = {'detection_boxes':boxes_, 'detection_classes':class_,'detection_scores':score_}
    
    # Process detections
    # for i, det in enumerate(pred):  # detections per image
    #     im0s = gen_result(frm, output_size, det, names, colors)
    #     vid_writer.write(im0s)
    
    return output_dict


def draw_box( img_in:np.array, det_dict, category_index, conf_thres, labelcolors, line_thickness:int=1):
     
    if len(det_dict['detection_boxes']):
    
        # Write results
        for ibox in range(len(det_dict['detection_boxes'])):
            box_ = det_dict['detection_boxes'][ibox,:]
            cls_ = det_dict['detection_classes'][ibox]
            conf = det_dict['detection_scores'][ibox]
            
            label = f'{category_index[int(cls_+1)]["name"]} {conf:.2f}'
            
            plot_one_box(box_, img_in, label=label, color=labelcolors[int(cls_)], line_thickness=line_thickness)
    
    return img_in

#########################################################################################################################
#########################################################################################################################

if __name__ =="__main__":
    
    PATH_TO_MODELS = f'/root/ENService/ai/med/obj_det/pytorch/models/yolov7/saved_model/ihp009inv_yolov7_best.pt'
    PATH_TO_LABELS = f'/root/ENService/ai/med/obj_det/tf/models/fasterrcnn/label/label_map.pbtxt'
    PATH_TO_VIDEOS = f'/root/ENService/video_test/test_videos'
    # category_index = load_label(path_to_label=PATH_TO_LABELS)
    
    source = '/yolov7/inference/videos'
    imgsz = 320

    conf_thres = 0.25
    iou_thres = 0.45
    
    # Directories
    save_dir = f'/yolov7/runs/detect/exp'
    
    # Load model
    # weights = f'/root/ENService/ai/med/obj_det/pytorch/models/ihp009inv_yolov7_best.pt' #'yolov7.pt'
    model_fn = load_model(PATH_TO_MODELS)
    
    stride = int(model_fn.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    category_index = load_label('', model_fn)
    colors = get_colors(category_index)

    # Get names and colors
    # names = model_fn.module.names if hasattr(model_fn, 'module') else model_fn.names
    # colors = get_colors(names)
    videolist = glob.glob(PATH_TO_VIDEOS+'/*.mp4')

    t0 = time_synchronized()
    
    # vid_path = '/yolov7/inference/videos/*'
    # vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
    # files = sorted(glob.glob(vid_path, recursive=True))
    # video_list = [x for x in files if x.split('.')[-1].lower() in vid_formats]
    
    
    for ivid, videoname in enumerate(video_list):
        
        videoname = f'/yolov7/inference/videos/videoplayback.mp4'
        print(videoname)
        
        vid, fps = vid_load(videoname)
        nfrm, h0, w0, ch = vid.shape
        
        
        for ifrm, frm in enumerate(vid):
            
            img_in, ratio, (dw, dh) = letterbox(frm, imgsz, stride=stride)
            # model_input = frm_trans(img_in)
            output_dict = det_single_frm(img_in, model_fn)
            
            im0s = frm
            im0s = draw_box(img_in, output_dict, model_fn.names)
            
            if ifrm==0: 
                save_path = save_dir + '/inference_' + video_list[0].split('/')[-1]
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (im0s.shape[1], im0s.shape[0]))    
            
            vid_writer.write(im0s)
            
            str_to_say = f'\r - Video {ivid:03d}/{len(video_list):03d} (frm {ifrm:03d}/{vid.shape[0]:03d}).' # TOTAL_TIME: ({t4-t0:0.1f}), ImgTrans ({(1E3 * (t2 - t1)):.1f}ms), Inference ({(1E3 * (t3 - t2)):.1f}ms), NMS ({(1E3 * (t4 - t3)):.1f}ms)'
            print(f'{str_to_say:200s}', end='')

        vid_writer.release() 
        print(save_path)
        break
    
    print(f'Done. ({time_synchronized() - t0:.3f}s)')



# if __name__ == '__main__':


#     # List of the strings that is used to add correct label for each box.
#     PATH_TO_LABELS = '/root/src/medication/object_detection/graph/label_map.pbtxt'
#     category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
#     print([category_index[key]['name'] for key in list(category_index.keys())])

#     for pathtmp in ['hand_main', 'hand_sub', 'no_medi']:
#         vid_path = pathlib.Path(f'/root/src/videos/{pathtmp}')
#         # img_path = pathlib.Path('/root/src/images')
#         # lbl_path = pathlib.Path('/root/src/labels')

#         ########################################
#         ### DATA LOAD
#         ########################################

#         # data_names = load_imgs(img_path, lbl_path, nDataSmp=100)
#         data_names = load_vids(vid_path)

#         ########################################
#         ### MODEL LOAD
#         ########################################

#         detection_model = load_model()
#         print(detection_model.signatures['serving_default'].inputs)
#         detection_model.signatures['serving_default'].output_dtypes
#         detection_model.signatures['serving_default'].output_shapes

#         ########################################
#         ### INFERENCE
#         ########################################

#         for video_name in data_names:
#             show_inference_vid(detection_model, vid_path, video_name, category_index, save=True)
        

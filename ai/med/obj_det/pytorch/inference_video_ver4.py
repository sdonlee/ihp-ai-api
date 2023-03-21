import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import os
import json
import copy
import numpy as np
import datetime

from collections import deque
from collections import Counter

medication_video_list = []
non_medication_video_list = []

def frame_to_timecode(frame,fps): #시간 측정
    total_time_s = frame / fps

    s = total_time_s % 60
    f = frame % fps

    return ("%02d:%02d" % (s, f))

#슬라이딩 윈도우 적용
#target_list : 프레임 결과 리스트. 윈도우 사이즈가 7이면, 7개 프레임의 결과가 저장되어 있는 리스트
#               해당 리스트는 [[클래스, confidence score]] 로 구성되어 있다.
#               len(target_list) == 7 로 되어 있음
#window_size : 윈도우 사이즈
def Sliding_window(target_list, window_size):
    flag  = False

    non_detected_frame_count = 0
    detected_frame_count = 0

    for k in range(len(target_list)): # 리스트 요소 순차적 확인
        if not target_list[k]: # 리스트 k번째 요소가 빈값이면 non_detected_frmae_count +=1  
            non_detected_frame_count += 1
        else:
            detected_frame_count += 1 # 값이 있으면 detected_frame_count += 1
    
    if non_detected_frame_count <= detected_frame_count: # 검출된 객체가 있는 프레임의 개수가 더 많으면 flag true
        flag = True
    else:
        pass

    return flag #flag return

def model_load(): #모델 로드
    weights, view_img, save_txt, imgsz, trace = opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace #파라미터들 제공
   
    # Directories
    save_dir = Path(increment_path(Path('./inference'), exist_ok=opt.exist_ok))  # increment run 저장 디렉토리
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device) #gpu 셋팅
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    model = TracedModel(model, device, opt.img_size)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    return device, model, stride, imgsz, trace, half, classify

def detect(sample, device, model, stride, imgsz, classify, trace, half, video_name,save_img=False): 
    source, save_txt = sample, opt.save_txt
    print(source)
    # source, weights, view_img, save_txt, imgsz, trace = sample, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace #파라미터들 제공
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path('./inference'), exist_ok=opt.exist_ok))  # increment run 저장 디렉토리
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    #json dic 구조
    each_bbox = {
                    "labels": [
                        {
                            "description": "None",
                            "score":0.0
                        }
                    ],
                    "positions":{
                                "x":0.0,
                                "y":0.0,
                                "w":0.0,
                                "h":0.0
                    }
    }

    JSON_STRUCTURE = {
        "results": {
            "frame_results" : [
            
            ]
        }
    }

    current_frame = 0
    target_list = deque()
    medication_clip_count = 0

    med_check = False #medication video classification flag

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        box_coordinates = []
        preds_and_score = []
        object_detection = []

        flag = False #obejct detection flag

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(opt.save_path + '/')   # img.jpg
            # save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            print('current frame: ', current_frame)

            if len(det) != 0:
                flag = True
            elif len(det) == 0:
                flag = False

            if flag == True: #현재 프레임에서 검출 결과가 있을때
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class #class number
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # 박스 좌표 변환
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    box_coordinates.append(xyxy)
                    preds_and_score.append([names[int(cls)], conf]) #프레임 별 결과 저장. [클래스, score]

            elif flag == False: # 현재 프레임에서 검출 결과가 없을때
                box_coordinates = []
                preds_and_score = [] 

        target_list.append(preds_and_score) #프레임 별 검출 결과를 리스트에 저장

        if len(target_list) >= int(opt.window_size): #프레임 결과 리스트 길이가 윈도우 사이즈 이상일떄
            check = Sliding_window(target_list, opt.window_size) #슬라이딩 윈도우 적용. check는 Sliding_window에서 반환된 flag값
            target_list.popleft() #다음 프레임 결과를 추가하기 위한 pop

            if check == True: #check가 True면, 복약 클립 개수 증가
                medication_clip_count += 1
            elif check == False:
                pass

        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path + 'inference_image.jpg', im0)
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(str(opt.save_path) + '/'+ video_name +'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                
                vid_writer.write(im0)

        #json 구조 내용 저장
        frame_result = {
                "video_name" : "None",
                "frame_number" : 0,
                "timestamp" : "None",
                "object_detection" : object_detection
        }
        frame_result['video_name'] = video_name #영상 이름
        frame_result['frame_number'] = current_frame #프레임 수
        frame_result['timestamp'] = frame_to_timecode(current_frame , fps) #시간

        current_frame += 1

        
        for j, bbox in enumerate(preds_and_score):
            each_bbox['labels'][0]['description'] = bbox[0] # 클래스
            each_bbox['labels'][0]['score'] = bbox[1].cpu().numpy().tolist() # confidence score

            # 박스 좌표값
            each_bbox['positions']['x'] = box_coordinates[j][0].cpu().numpy().tolist()
            each_bbox['positions']['y'] = box_coordinates[j][1].cpu().numpy().tolist()
            each_bbox['positions']['w'] = box_coordinates[j][2].cpu().numpy().tolist() - box_coordinates[j][0].cpu().numpy().tolist()
            each_bbox['positions']['h'] = box_coordinates[j][3].cpu().numpy().tolist() - box_coordinates[j][1].cpu().numpy().tolist()

            finished_bbox = copy.deepcopy(each_bbox)
            object_detection.append(finished_bbox)

        finished_object_detectoin = copy.deepcopy(object_detection)
        frame_result['object_detection'] = finished_object_detectoin
        finished_frame_result = copy.deepcopy(frame_result)
        JSON_STRUCTURE['results']['frame_results'].append(finished_frame_result)

    if medication_clip_count >= int(opt.Sliding_window_threshold): #현재 영상에서 복약 클립 개수가 슬라이딩 윈도우 임계값보다 크면 복약 영상으로 처리. 그 결과를 json 저장
        med_check = True
        print(video_name + '_is medication video')
        JSON_STRUCTURE['results']['medication'] = True
    else:
        med_check = False
        JSON_STRUCTURE['results']['medication'] = False

    print(f'Done. ({time.time() - t0:.3f}s)')

    #json 파일 저장
    json_name = video_name +'.json'
    with open(opt.save_path + '/' + json_name,'w') as outfile:
        json.dump(JSON_STRUCTURE, outfile, indent=4)

    return med_check #inference 결과 flag

def inference_video(video_path):

    processing_start = time.time() 
    path = video_path
    file_list = os.listdir(path)
    print(file_list)

    #저장 경로 확인
    try:
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
    except OSError:
        print('Error: Creating Directory. ' + opt.save_path)

    file_list_filtering = [file for file in file_list if file.startswith('2') or file.startswith('1') or file.startswith('n')] #임시 파일 필터링
    print(len(file_list_filtering))

    model_load_start = time.time()
    device, model, stride, imgsz, trace, half, classify = model_load() #모델 로드
    model_load_end = time.time()

    for i in file_list_filtering:
        sample = path +'/'+ i
        print(i)
        print(sample)

        start_each_video = time.time()
        med = detect(sample, device, model, stride, imgsz, trace, half, classify, i) #inference flag

        if med == True:
            medication_video_list.append(i)
        elif med == False:
            non_medication_video_list.append(i)
        end_each_video = time.time()

        sec = (end_each_video - start_each_video)
        inference_time = str(datetime.timedelta(seconds=sec)).split(".")
        print('inference_time: ', inference_time)

    processing_finished = time.time()
    print('model load time: ', str(datetime.timedelta(seconds = (model_load_end - model_load_start))).split("."))
    print('processing time: ', str(datetime.timedelta(seconds = (processing_finished - processing_start))).split("."))

    print('medication video list : ', medication_video_list, len(medication_video_list))
    print('non medication video list: ', non_medication_video_list, len(non_medication_video_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    # parser.add_argument('--video_path', type=str, default='ihpdata/ihp_video', help='video_path'ㅌㄴ)
    parser.add_argument('--save_path', type=str, default='./inference', help='inference_save_path')
    parser.add_argument('--window_size', type=int, default=7, help='Sliding window size')
    parser.add_argument('--Sliding_window_threshold', type=float, default='24', help='Sliding window method threshold')
    opt = parser.parse_args()

    source_path = opt.source

    print('inference start.....')
    inference_start = time.time()
    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov7.pt']:
                inference_video(source_path)
                strip_optimizer(opt.weights)
        else:
            inference_video(source_path)
    inference_end = time.time()
    print('inference finished.... ',str(datetime.timedelta(seconds= inference_end - inference_start)).split("."))
from typing import Tuple

import cv2
import numpy as np
import glob
import time

import os
from pathlib import Path
# import shutil

# from multiprocessing import Pool

# def vid_load(filein:str=""):
    
#     frames = []
#     cap = cv2.VideoCapture(filein)
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     ret = True
#     while ret:
#         ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
#         if ret:
#             frames.append(img)
#     video = np.stack(frames, axis=0) # dimensions (T, H, W, C)
#     cap.release()
    
#     return video, fps


def vid_resize(video:np.array, target_size:tuple):
    frames = []
    for img in video:
        resize_img = cv2.resize(img, target_size)
        frames.append(resize_img)
    video_out = np.stack(frames, axis=0) # dimensions (T, H, W, C)
    
    return video_out

def vid_gry(video:np.array):
    frames = []
    for img in video:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.stack((img,) * 3, axis=-1)
        
        frames.append(img)
    video_out = np.stack(frames, axis=0) # dimensions (T, H, W, C)
    
    return video_out
    


def video_read(filein: str, ifrm: int = -1, _gray:bool=False, target_size: tuple = (0, 0)) -> Tuple[np.ndarray, float]:
    """load video with opencv
    return: video(np.ndarray[0~255 uint8]) & fps(float)
    """
    filepath = filein

    cap = cv2.VideoCapture(filepath)
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_ = cap.get(cv2.CAP_PROP_FPS)
    nfrm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vid_ = np.zeros((nfrm, frame_height, frame_width, 3)).astype("uint8")
    for ifrm in range(nfrm):    
        ret, vid_[ifrm, :] = frm = cap.read()
    cap.release()       

    # Additional Process
    if target_size[0]==0:
        vid_out = vid_
    else:
        vid_out = np.zeros((nfrm, target_size[1], target_size[0], 3)).astype("uint8")
        for ifrm, frm in enumerate(vid_):
            if not target_size[0]==0: # RESIZE
                frm = cv2.resize(frm, target_size)
            if _gray: # GRAY
                frm = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                frm = np.stack((frm,) * 3, axis=-1)
            
            vid_out[ifrm, :] = frm

    return vid_out, fps_


def video_write(
    vid: np.ndarray,
    fileout: str,
    fps: float = 25,
    target_size: tuple = (320, 240),
):
    """Video writer

    Args:
        vid (np.ndarray): 4d numpy array (nfrm x height x width x ch)
        fileout (str): path for file saving
        fps (float, optional): Defaults to 25.
        target_size (tuple, optional): Defaults to (320, 240).
    """
    nfrm, height, width, depth = vid.shape

    target_size = (width, height)
    filepath = fileout

    filepath_tmp = filepath.split('/')
    filepath_tmp[-1] = filepath_tmp[-1].split('.')[0]+'_mp4v.mp4'
    filepath_tmp = '/'.join(filepath_tmp)
    
    ####
    out = cv2.VideoWriter(
        filepath_tmp,
        cv2.VideoWriter_fourcc(*"mp4v"),#"avc1"), #*"h264"),  # MPEG"),  # "M", "J", "P", "G"),
        fps,
        target_size,
    )

    for ifrm in range(nfrm):
        img = vid[ifrm, :]
        out.write(img)

    out.release()

    # os.system(f"ffmpeg -i {filepath_tmp} -vcodec libx264 {filepath}")
    # os.remove(filepath_tmp)

# def feat2vid(feat_: np.ndarray) -> np.ndarray:
#     vid_ = np.stack((feat_,) * 3, axis=-1)
#     return vid_





if __name__ == "__main__":

    file_in = "../video_test/test_videos/test/videoplayback.mp4"
    
    aiapi = VideoIO(video_path = video_, DEBUG=True)
    
    path_in_ = "video/test/"
    path_out = "result/test"
    filepath_list = glob.glob(f"{path_in_}/**/*.mp4", recursive=True)

    for ifile, filepath_in_ in enumerate(filepath_list):
        filepath_out = f"{path_out}/{'/'.join(filepath_in_.split('/')[2:])}"
        if not os.path.exists(filepath_out):
            print(f'{ifile:5d}/{len(filepath_list):5d} - Loads & Saves... {filepath_in_:<50s}')
            
            Path('/'.join(filepath_out.split('/')[:-1])).mkdir( parents=True, exist_ok=True )

            vid, fps = video_write(filepath_in_,_gray=True,target_size=(320, 240))
            video_write(vid, filepath_out, fps, target_size=(320, 240))
        else:
            print(f'{ifile:5d}/{len(filepath_list):5d} - Already exists... {filepath_in_:<50s}')

    # filename_list = [
    #     path_.split("/")[-1] for path_ in glob.glob(f"data/video/gachon/*.mp4")
    # ]
    # filename_list = filename_list[:10]

    # def video_write_mp(filename):
    #     print(filename)
    #     filepath_in = f"data/video/gachon/{filename}"
    #     filepath_out = f"data/video/easy_result/gachon/vid_gry/{filename}"
    #     # filepath_out = f"{path_data}/easy_result/vid_gry/{filepath_.split('/')[-1]}"

    #     if not os.path.exists(filepath_out):
    #         vid, fps = video_load(filepath_in)
    #         vid_g = video_preprocess(vid, target_size=(320, 240))
    #         vid_3d = np.stack((vid_g * 255.0,) * 3, axis=-1).astype("uint8")
    #         video_write(vid_3d, filepath_out, fps, target_size=(320, 240))

    # nProcess = 8
    # with Pool(nProcess) as p:
    #     p.map(video_write_mp, filename_list)

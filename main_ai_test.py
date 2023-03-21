import glob, time
import numpy as np
import os

from ai.aiapi_ import VideoIO, EasyNeg, ObjDet, MedDet, AIAPI
# from ai.aiapi import AIAPI
# from ai.aiapi_easyneg import EasyNeg

def test_common(video_):

    aiapi = VideoIO(video_path = video_, DEBUG=True)
    aiapi.video_read()
    aiapi.video_resize()
    

def test_easyneg(video_):
    # aiapi = AIAPI(video_path = video_, DEBUG=True)
    # aiapi.analyze_easyng()
    aiapi = EasyNeg(video_path = video_, DEBUG=True)
    aiapi.analyze_neg()
    
    print(f"{aiapi.scr_drk:.2f}, {aiapi.scr_mtl:.2f}")
    print([f"{key_}: {val_:.2f}," for key_, val_ in zip(aiapi.time.keys(), aiapi.time.values())])
    
    
def test_objdet(video_):
    # aiapi = ObjDet(video_path = video_, det_lib='tf', model_path=f'/root/ENService/ai/med/obj_det/tf/models/fasterrcnn', DEBUG=True)
    aiapi = ObjDet(video_path = video_, det_lib='pytorch', model_path=f'/root/ENService/ai/med/obj_det/pytorch/models/yolov7', DEBUG=True)
    aiapi.analyze_det()
    # aiapi.video_read()
    # aiapi.detect()
    # aiapi.save_det()
    
def test_meddet(video_):
    
    # aiapi = MedDet(video_path = video_, det_lib='tf', model_path=f'/root/ENService/ai/med/obj_det/tf/models/fasterrcnn', DEBUG=True)
    aiapi = MedDet(video_path = video_, prj_name='IHP009INV', DEBUG=True)
    # aiapi = MedDet(video_path = video_, prj_name='TST001DMO', DEBUG=True)
    aiapi.analyze_med()
    
    t1 = time.time()
    os.system(f"ffmpeg -i {filepath_tmp} -vcodec libx264 {filepath}")
    os.remove(filepath_tmp)
    t2 = time.time()
    print(f' --> Encoding: {t2-t1:.2f}')
    
    
    

if __name__ == '__main__':
    
    # path_ = f'/home/ihp/don/server/ihp-ai-server-micro'
    path_ = f'/root/ENService'
    videolist = glob.glob(path_+f'/video_test/test_videos/*.mp4')
    print(f' VIDEO LOAD # {len(videolist):03d}')
    time_array = np.zeros((len(videolist),4))

    for ivid, video_ in enumerate(videolist):
        
        # video_ = '/root/ENService/ai/video_test/videoplayback.mp4'
        print("==========================================================")
        print(f" - {ivid:03d} - {video_}")
        t0 = time.time()
        test_common(video_)
        # test_easyneg(video_)
        # test_objdet(video_)
        # test_meddet(video_)
        t1 = time.time()
        
        time_array[ivid,0] = t1-t0
        # break
        # time_array[ivid,1] = t2-t1
        # aiapi.reset()
#         aiapi.analyze_neg()
#         t2 = time.time()
#         aiapi.analyze_det()
#         t3 = time.time()
        
#         print(aiapi.scr_med, video_)
#         print(f' - Time LoadAll: {t1-t0:02.2f}')
#         print(f' - Time EasyNeg: {t2-t1:02.2f}')
#         print(f' - Time AnalDet: {t3-t2:02.2f}')

        # time_list.append([t2-t1,t3-t2])

#         ivid = ivid+1
    # print(time_array.mean(axis=0).round(2))
import numpy as np
import pathlib
import os
import glob
import time

import json

from ai.common.videoio import video_read, video_write, vid_resize, vid_gry
from ai.common.objdet import get_colors
from ai.neg.analysis import feat_raw, feat_frm, feat_vid


class VideoIO:
    def __init__(self, video_path:str="../video_test/test_videos/test/videoplayback.mp4", target_size:tuple=(320,240), DEBUG:bool=False):
        
        self.DEBUG = DEBUG
        
        self.time = dict()
        self.time["video_rd_"] = 0.
        self.time["video_rsz"] = 0.
        self.time["video_wrt"] = 0.
        
        self.path_vid_ori = video_path
        self.path_vid_det = self.__gen_path_vid_det()
        
        self.vid_ori = np.zeros((1,1,1))
        self.vid_rsz = np.array((1,1,1))
        self.vid_out = np.array((1,1,1))
        
        self.fps = 0
        self.nfrm = 0
        self.size_ori = (0,0) 
        self.size_out = target_size
        
    def __gen_path_vid_det(self):
        video_path = self.path_vid_ori 
        return f"../result/{video_path.split('/')[-1]}"
        
    def print_time(self):
        for item in list(self.time.keys()):
            print(f" - PROCESS TIME for {item} : {self.time[item]:.3f} sec ")
        
    def video_read(self):
        
        if self.DEBUG:
            t1 = time.time()
        
        self.vid_ori, self.fps = video_read(self.path_vid_ori)
        self.nfrm = self.vid_ori.shape[0]
        self.size_ori = self.vid_ori.shape[1:3]
        # self.target_size = (320,240)
    
        if self.DEBUG:
            t2 = time.time()
        
        if self.DEBUG:
            print(f" - VIDEO Info:")
            print(f" - >>> {self.path_vid_ori}") 
            print(f" - >>> {self.vid_ori.shape}")
            print()
            print(f" - >>> {t2-t1:.2f} secs - Time for Video Load")
            self.time["video_rd_"] = t2-t1
        
    
    def video_resize(self):
        
        if self.DEBUG:
            t1 = time.time()
        if self.size_out[0]==0:
            self.vid_rsz = self.vid_ori
        else:
            self.vid_rsz = vid_resize(self.vid_ori, self.size_out)
        
        if self.DEBUG:
            t2 = time.time()
            
            print(f" - VIDEO Resize:")
            print(f" - >>> {self.vid_ori.shape} =>  {self.vid_rsz.shape}")
            print()
            print(f" - >>> {t2-t1:.2f} secs - Time for Video Resize")
            self.time["video_rsz"] = t2-t1
        
    
    def video_write(self):
        
        if self.DEBUG:
            t1 = time.time()
        
        video_write(
            self.vid_out,
            self.path_vid_det,
            self.fps,
            self.size_out,
        )
    
        if self.DEBUG:
            t2 = time.time()
        
        if self.DEBUG:
            print(f" - VIDEO Write:")
            print(f" - >>> {self.path_vid_det}") 
            print(f" - >>> {self.vid_out.shape}")
            print()
            print(f" - >>> {t2-t1:.2f} secs - Time for Video Write")
            self.time["video_wrt"] = t2-t1
            
    # def delete_video_in(self):
    #     if os.path.exists(self.path_vid_ori):
    #         os.remove(self.path_vid_ori)
    
    # def delete_video_out(self):
    #     if os.path.exists(self.path_vid_det):
    #         os.remove(self.path_vid_det)
    

class EasyNeg(VideoIO):
    def __init__(self, video_path:str='', target_size:tuple = (320,240), DEBUG:bool=False):
        super().__init__(video_path)
        
        self.DEBUG = DEBUG
        self.time["easyneg_drk"] = 0.
        self.time["easyneg_mtl"] = 0.
        
        self.scr_drk = 0
        self.scr_mtl = 0
        
        self.lbl_drk = False
        self.lbl_mtl = False
        self.__get_info()
        
    def __get_info(self):
        
        config_path = '/'.join(__file__.split('/')[:-1]) + '/neg/config_neg.json'
        with open(config_path,'r') as f:
            config = json.load(f)
        
        self.time_range = config['NEG']['TIME_RANGE']
        self.thr_drk = config['NEG']['THR_DRK']
        self.thr_mtl = config['NEG']['THR_MTL']
        
        # self.video_read()
        # self.video_resize()

    def analyze_drk(self):
        
        if self.DEBUG:
            t1 = time.time()
            
        f_loc = feat_raw(self.vid_rsz, "brt")
        f_frm = feat_frm(f_loc, "avr", thr_ = 0.0)
        f_vid = feat_vid(f_frm, "avr", val_ = self.time_range)

        if self.DEBUG:
            t2 = time.time()
            self.time['easyneg_drk'] = t2-t1
            print(f' --> easyneg - drk: {t2-t1:.2f}')
            
        self.scr_drk = f_vid
        self.lbl_drk = bool(self.scr_drk > self.thr_drk)

    def analyze_mtl(self):
        
        if self.DEBUG:
            t1 = time.time()
            
        f_loc = feat_raw(self.vid_ori, "ofg")
        f_frm = feat_frm(f_loc, "avr", thr_=0.0)
        f_vid = feat_vid(f_frm, "avr", val_=self.time_range)
        
        if self.DEBUG:
            t2 = time.time()
            self.time['easyneg_mtl'] = t2-t1
            print(f' --> easyneg - mtl: {t2-t1:.2f}')
            
        self.scr_mtl = f_vid
        self.lbl_mtl = bool(self.scr_mtl > self.thr_mtl)
    
    def analyze_neg(self):

        self.video_read()
        self.video_resize()
        self.analyze_drk()
        # self.analyze_mtl()


class ObjDet(VideoIO):
    def __init__(self, video_path:str, prj_name:str="TST001DMO", label_fake:bool=True, save_video:bool=True, DEBUG:bool=False):
        super().__init__(video_path, DEBUG=DEBUG)
        
        self.DEBUG = DEBUG
        self.time["objdet_det"] = 0.
        self.time["objdet_drw"] = 0.
        self.time["objdet_wrt"] = 0.
        
        self.__get_info(prj_name)
        
        if self.det_lib=='tf':
            from ai.med.obj_det.tf.analysis import load_label, load_model
            
            self.model_path = f"{self.default_path}/obj_det/{self.det_lib}/models/{self.modelname}/saved_model"
            self.label_path = f"{self.default_path}/obj_det/{self.det_lib}/models/{self.modelname}/label/label_map.pbtxt"
            #'label_map_fake.pbtxt'
            
        elif self.det_lib=='pytorch':
            from ai.med.obj_det.pytorch.analysis import load_label, load_model

            self.model_path = f"{self.default_path}/obj_det/{self.det_lib}/models/{self.modelname}/saved_model/best.pt"
            # self.model_path = model_path+'/saved_model/best.pt'
            self.label_path = ''
        
        self.save_video = save_video
        
        self.path_out = f"{self.default_path}/../../result" #'/'.join(model_path.split('/')[:-6] +['result'])
        videoname = self.path_vid_ori.split('/')[-1]
        self.path_vid_det= self.path_out +'/'+ 'det_'+videoname
        
    def __get_info(self, prj_name:str="TST001DMO"):
        config_path = '/'.join(__file__.split('/')[:-1]) + '/med/config_med.json'
        with open(config_path,'r') as f:
            config = json.load(f)
        
        self.default_path = config['MED']['DEFAULT_PATH']
        
        self.modelname = config['MED']['OBJDET'][prj_name]['MODEL']
        
        self.det_lib = config['MED']['OBJDET'][prj_name]['LIB']
        self.conf_thres = config['MED']['OBJDET'][prj_name]['THR_DET']
        self.iou_thres = config['MED']['OBJDET'][prj_name]['THR_IOU']
        # self.win_sz = config['MED'][{prj_name}]['CLP_SIZE']
    
    def load_model(self):
    
        if self.DEBUG:
            t1 = time.time()
    
        self.model_path = f"{self.default_path}/obj_det/{self.det_lib}/models/{self.modelname}/saved_model"
        self.label_path = f"{self.default_path}/obj_det/{self.det_lib}/models/{self.modelname}/label/label_map.pbtxt"
        if self.det_lib=='pytorch':
            self.model_path = f"{self.model_path}/best.pt"
            self.label_path = ''
            
        if self.det_lib=='tf':
            from ai.med.obj_det.tf.analysis import load_label, load_model
        elif self.det_lib=='pytorch':
            from ai.med.obj_det.pytorch.analysis import load_label, load_model
                    
        self.model_fn = load_model(model_path=self.model_path)
        self.category_index = load_label(self.label_path, self.model_fn)
        
        if self.DEBUG:
            t2 = time.time()
            self.time["objdet_mdl"] = t2-t1
            print(f' --> objdet - mdl: {t2-t1:.2f}')
        
    def detect(self):
        
        if self.DEBUG:
            t1 = time.time()
            
        if self.det_lib=='tf':
            from ai.med.obj_det.tf.analysis import det_single_frm
        elif self.det_lib=='pytorch':
            from ai.med.obj_det.pytorch.analysis import det_single_frm
        
        for ifrm, frm in enumerate(self.vid_ori):
            if ifrm ==0:
                det_box = []
            det_dict = det_single_frm(frm, self.model_fn, self.conf_thres, self.iou_thres)
            det_box.append(det_dict)
            ifrm = ifrm+1
        
        self.det_box = det_box
        
        if self.DEBUG:
            t2 = time.time()
            self.time["objdet_det"] = t2-t1
            print(f' --> objdet - det: {t2-t1:.2f}')
    
    def draw_box(self):
        
        if self.DEBUG:
            t1 = time.time()
        
        if self.det_lib=='tf':
            from ai.med.obj_det.tf.analysis import draw_box
        elif self.det_lib=='pytorch':
            from ai.med.obj_det.pytorch.analysis import draw_box
            
        det_box = self.det_box
            
        labelcolors = get_colors(self.category_index)
        for ifrm, frm in enumerate(self.vid_ori):
            if ifrm==0:
                frames = []
            img = draw_box(frm, det_box[ifrm], self.category_index, self.conf_thres, labelcolors)
            frames.append(img)
        
        # self.vid_out = vid_resize(np.stack(frames, axis=0), target_size=(1280,720)) # dimensions (T, H, W, C)
        self.vid_out = np.stack(frames, axis=0)
        if self.DEBUG:
            t2 = time.time()
            self.time["objdet_drw"] = t2-t1
            print(f' --> objdet - drw: {t2-t1:.2f}')
                
    def save_vid(self):
        
        if self.DEBUG:
            t1 = time.time()
            
        video_write(self.vid_out, self.path_vid_det)
        if self.DEBUG:
            t2 = time.time()
            self.time["objdet_wrt"] = t2-t1
            print(f' --> objdet - wrt: {t2-t1:.2f}')
    
    def analyze_det(self):

        self.video_read()
        self.load_model()
        self.detect()
        self.draw_box()
        self.save_vid()
        
                        
class MedDet(ObjDet):
    def __init__(self, video_path:str, prj_name:str="TST001DMO", label_fake:bool=True, save_video:bool=True, DEBUG:bool=False):
        super().__init__(video_path, prj_name, label_fake=label_fake, save_video=save_video, DEBUG=DEBUG)
        
        self.nBoxInFrm = 0
        self.nEnoughBoxedClip = 0
        self.get_info(prj_name)
    
    def get_info(self, prj_name):
        config_path = '/'.join(__file__.split('/')[:-1]) + '/med/config_med.json'
        with open(config_path,'r') as f:
            config = json.load(f)
            
        self.default_path = config['MED']['DEFAULT_PATH']
        
        self.modelname = config['MED']['OBJDET'][prj_name]['MODEL']
        
        self.det_lib = config['MED']['OBJDET'][prj_name]['LIB']
        self.conf_thres = config['MED']['OBJDET'][prj_name]['THR_DET']
        self.iou_thres = config['MED']['OBJDET'][prj_name]['THR_IOU']
        
        self.clp_thr = config['MED']['OBJDET'][prj_name]['THR_nCLP']

    def count_box(self):
        nfrm = self.vid_ori.shape[0]
        nlabel = len(self.category_index)
    
        nBoxInFrm = np.zeros((nfrm, nlabel))
        for ifrm in range(nfrm):
            for ilabel in range(nlabel):
                nBoxInFrm[ifrm, ilabel] += np.sum([self.det_box[ifrm]['detection_classes']==ilabel])
        
        self.nBoxInFrm = nBoxInFrm
    
    def count_clp(self , win_sz:int = 7):
        
        nBoxInFrm = self.nBoxInFrm
        
        existBoxInFrm = np.sum(nBoxInFrm,axis=1)>0
        nBoxedFrmCumsum = existBoxInFrm.astype('int').cumsum()
        nEnoughBoxedClip = np.sum(nBoxedFrmCumsum[win_sz:] - nBoxedFrmCumsum[:-win_sz] >= win_sz/2)
        
        self.nEnoughBoxedClip = nEnoughBoxedClip
        
    def analyze_med(self):
        
        if self.DEBUG:
            t1 = time.time()
        
        self.video_read()
        self.load_model()
        self.detect()
        self.draw_box()
        self.save_vid()
        self.count_box()
        self.count_clp()
        
        self.scr_med = self.nEnoughBoxedClip
        self.lbl_med = self.scr_med > self.clp_thr
        
        if self.DEBUG:
            t2 = time.time()
            self.time["med"] = t2-t1
            print(f' --> med: {t2-t1:.2f}')
        

class AIAPI(MedDet, EasyNeg):
    def __init__(self, prj_name:str="TST001DMO", video_path:str='', DEBUG:bool=False):
        
        config_path = '/'.join(__file__.split('/')[:-1]) + '/common/config.json'
        with open(config_path,'r') as f:
            config = json.load(f)
        
        time_range = config['NEG']['TIME_RANGE']
        thr_drk = config['NEG']['THR_DRK']
        thr_mtl = config['NEG']['THR_MTL']
        #EasyNeg: video_path:str='', target_size:tuple=(320,240), thr_drk:float=0.9, thr_mtl:float=0.95, DEBUG:bool=False
        EasyNeg.__init__(self, video_path=video_path, time_range=time_range, thr_drk=thr_drk, thr_mtl=thr_mtl)
        
        
        modelname = config['MED']['OBJDET'][prj_name]['MODEL']
        
        det_lib = config['MED']['OBJDET'][prj_name]['LIB']
        thr_cnf = config['MED']['OBJDET'][prj_name]['THR_DET']
        thr_iou = config['MED']['OBJDET'][prj_name]['THR_IOU']
        # self.win_sz = config['MED'][{prj_name}]['CLP_SIZE']
        thr_clp = config['MED']['OBJDET'][prj_name]['THR_nCLP']
        
        self.default_path = config['MED']['DEFAULT_PATH']
        self.modelname = modelname
        # ObjDet: video_path:str, det_lib:str='tf', model_path:str='', label_fake:bool=True, save_video:bool=True, conf_thres:float=0.7, iou_thres:float=0.45, DEBUG:bool=False
        # MedDet: video_path:str, det_lib:str='tf', model_path:str='', label_fake:bool=True, save_video:bool=True, conf_thres:float=0.7, iou_thres:float=0.45, clp_thr:float=24, DEBUG:bool=False
        MedDet.__init__(self, video_path=video_path, det_lib=det_lib, conf_thres=thr_cnf, iou_thres=thr_iou, clp_thr=thr_clp)
        
        # super().__init__(video_path=video_path, model_path=path_tmp, DEBUG=DEBUG)
        
    def analyze_easyng(self):
        
        self.video_read()
        self.analyze_neg()

    def analyze_meddet(self):

        self.video_read()
        self.detect()
        self.save_det()
        self.analyze_med()

        

import cv2
import numpy as np
import glob
import copy

import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec

from multiprocessing import Pool

# from common.videoio import video_load, video_write
# from ai.common.videoio import video_read, video_write
# if __name__ == '__main__':
#     from neg.videoio import video_load, video_write
# else:
#     from ai.neg.videoio import video_load, video_write

def feat_vid(feat_frm:np.ndarray, featname_vid:str, val_:float):
    
    nfrm = feat_frm.shape[0]
    feat_frm_rvs = np.flip(feat_frm,axis=0)
    
    feat_frm_cumsum = feat_frm_rvs.cumsum()
    if featname_vid == 'avr':
        nfrm_target = np.floor(nfrm*val_).astype(np.int32)
        f_vid = 1.-feat_frm_cumsum[nfrm_target]/nfrm_target
        # negativeness = 1-motionness or 1-brightness
    elif featname_vid == 'frm':
        try:
            f_vid = np.where(feat_frm_cumsum>val_)[0][0]/nfrm
            # negativeness = nfrm_enoughmotion or nfrm_enoughbrightness
        except:
            f_vid = 1.
    else:
        f_vid = 0.

    return f_vid

def feat_frm(feat_loc: np.ndarray, featname_frm: str, thr_: float = 0.0) -> np.ndarray:

    nfrm, h, w = feat_loc.shape
    # feat_name: avr, cnt_thr,
    if not nfrm == 0:
        
        if featname_frm == "avr":
            feat_loc = feat_loc.clip(thr_,1.)
            feat_ = feat_loc.mean(axis=(1,2))
        elif featname_frm == "cnt":
            feat_ = (feat_loc > thr_).mean(axis=(1,2))
        else:
            feat_ = np.zeros((1, 1))
    else:
        feat_ = np.zeros((1, 1))

    return feat_

def feat_raw(vid:np.ndarray, feat_name:str) -> np.ndarray:
    """
    args: ndarray-4d, nfrm x H x W (0.~1.)
    returns: ndarray-3d, nfrm x H x W (0~1 float)
    """

    if not vid.shape[0] == 0:
        if feat_name == "brt":
            vid = vid.astype(np.float32)/255.
            feat_ = vid[:,:,:,0]
        elif feat_name == "ofg":
            vid = vid[:,:,:,0]
            feat_ = feat_optflow_gf(vid)
        else:
            feat_ = np.zeros((1, 1))
    else:
        feat_ = np.zeros((1,1))

    return feat_

def feat_optflow_gf(vid_gry: np.ndarray, max_val: float = 32.) -> np.ndarray:

    prev = None  # 이전 프레임 저장 변수

    feat_of = np.zeros_like(vid_gry).astype(np.float32)

    for ifrm in range(vid_gry.shape[0]):

        gray = vid_gry[ifrm, :]
        if prev is None:
            prev = gray
        else:
            frm1_gpu = cv2.cuda_GpuMat(prev)
            frm2_gpu = cv2.cuda_GpuMat(gray)

            # create optical flow instance
            gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(
                3,  # pyr_level
                0.5,  # pyr_scale
                False,
                15,  # win_size
                3,  # iter
                5,  # poly_n
                1.1,  # poly_sigma
                0,
            )

            # calculate optical flow
            gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc(
                gpu_flow,
                frm1_gpu,
                frm2_gpu,
                None,
            )
            # prev, next, flow(output), pyr_scale, levels, winsize=15, iternations=3, poly_n=5, poly_sigma=1.1, flag
            flow = gpu_flow.download()
            flow_mag = np.sqrt(np.sum(flow ** 2, axis=2))

            feat_of[ifrm, :] = flow_mag  # / ofg_thre

            prev = gray

    feat_of.clip(0.,max_val)

    return feat_of

def get_gt_gsheet():

    service_gsheet = gapi_connect()
    sheetid = "1nuI80IcDmZPRA1VRH8OeJbS7kEuh_72XDAApC1bJWQE"

    sheet_contents = read_sheet(
        service_gsheet,
        sheetid,
    )
    table_ = np.array(sheet_contents['label_zero'])

    file_list = [str_.split('/')[-1] for str_ in table_[2:,0]]
    label_dict = dict()
    idx_column = table_[0,:]=='pos'
    label_dict['pos'] = (table_[2:,idx_column].astype(int).sum(axis=1)>0).astype(int)
    nLabelNeg = np.sum( table_[0,:]=='neg') 
    for i in range(nLabelNeg):
        label_dict[table_[1,-nLabelNeg+i]] = [int(str_) for str_ in table_[2:,-nLabelNeg+i]]
    return file_list, label_dict

def get_dataset():
    '''
    filename
    filepath
    label_gt
    feature
    label_predict
    '''
    
    file_list, label_dict = get_gt_gsheet()
    
    path_list = glob.glob('result/test/**/*.mp4')
    filefrompath = [path_.split('/')[-1] for path_ in path_list]

    data_dict = dict()
    for ifile, filename in enumerate(file_list):
        data_dict[filename] = dict()
        
        idx_tmp = filefrompath.index(filename)
        data_dict[filename]['filepath'] = path_list[idx_tmp]
        data_dict[filename]['label_gt'] = dict()
        for key_ in list(label_dict.keys()):
            data_dict[filename]['label_gt'][key_] = label_dict[key_][ifile]
        data_dict[filename]['feat'] = dict()
        for key_ in list(label_dict.keys()):
            data_dict[filename]['feat'][key_] = 0
        data_dict[filename]['labelprd'] = dict()
        for key_ in list(label_dict.keys()):
            data_dict[filename]['labelprd'][key_] = 0
        

    return data_dict



def set_thre(data_dict, target_label):

    feat_list = list()
    label_list = list()
    idx_dict = dict()
    idx_dict['pos'] = list()
    idx_dict['neg'] = list()
    idx_dict['non'] = list()
    
    for ifile, filename in enumerate(data_dict.keys()):
        datum_dict = data_dict[filename]
        # print(data_dict[filename]['feat'])
        feat_list.append(datum_dict['feat'])
        if datum_dict['label_gt']['pos']==1:
            label_list.append(0.)
        elif datum_dict['label_gt'][target_label]==1:
            label_list.append(1.)
        else:
            label_list.append(.5)
        
    acc_dict = dict()
    acc_dict['pos'] = list()
    acc_dict['neg'] = list()
    acc_dict['all'] = list()

    thr_list = np.linspace(0,1,21)
    for thr_ in  thr_list:
        label_prd = np.array(feat_list)<thr_
        label_pos = np.array(label_list)==1
        label_neg = np.array(label_list)==0
        label_non = np.array(label_list)==.5

        nCrtPos = np.sum(label_prd*label_pos)
        nCrtNeg = np.sum((1-label_prd)*label_neg)
        
        acc_pos = nCrtPos/np.sum(label_pos)
        acc_neg = nCrtNeg/np.sum(label_neg)
        acc_all = (nCrtPos+nCrtNeg)/(np.sum(label_pos)+np.sum(label_neg))
        
        acc_dict['pos'].append(acc_pos)
        acc_dict['neg'].append(acc_neg)
        acc_dict['all'].append(acc_all)

    thr_ = thr_list[acc_dict['all'].index(max(acc_dict['all']))]

    return thr_, acc_dict

def visualize_data(data_dict, acc_dict, experiment_title, thr_):
    
    acc_max = max(acc_dict['all'])
    thr_opt = thr_

    experiment_title = f'{experiment_title}-{thr_opt:.3f}-ACC{acc_max:.3f}'
    featname_loc = experiment_title.split('-')[1]
    if featname_loc =='brt':
        target_label = 'dark'
    elif featname_loc == 'ofg':
        target_label = 'take-off'
    else:
        print(f'Not a Proper featname_loc: {featname_loc}')
    
    
    clr_dict = dict()
    clr_dict['pos'] = 'blue'
    clr_dict['neg'] = 'red'
    clr_dict['non'] = 'lightgray'
    clr_dict['all'] = 'lightgray'
    
    idx_dict = dict()
    idx_dict['pos'] = list()
    idx_dict['neg'] = list()
    idx_dict['non'] = list()
    
    feat_list = list()
    for ifile, filename in enumerate(data_dict.keys()):
        datum_dict = data_dict[filename]
        # print(data_dict[filename]['feat'])
        feat_list.append(datum_dict['feat'])
        if datum_dict['label_gt']['pos']==1:
            idx_dict['pos'].append(ifile)
        elif datum_dict['label_gt'][target_label]==1:
            idx_dict['neg'].append(ifile)
        else:
            idx_dict['non'].append(ifile)

    fig = plt.figure(figsize=(24, 12))
    plt.subplot(1,2,1)
    for label_ in idx_dict.keys():
        plt.scatter(idx_dict[label_], np.array(feat_list)[idx_dict[label_]], c=clr_dict[label_], label=label_,)
    plt.axhline(
        thr_opt,
        color=clr_dict['all'],
        linestyle="--",
    )
    plt.legend()
    plt.subplot(1,2,2)
    for label_ in acc_dict.keys():
        plt.plot(acc_dict[label_], '.-', label=label_, c =clr_dict[label_])
    plt.legend()
    plt.suptitle(experiment_title)
    plt.savefig(f'result/test/{experiment_title}.png')
    plt.close()


# if __name__ == "__main__":
#     '''
#     Video Load
#     Feature Generation (local->frame->video)
#     Save Result (feature, label)
#     Visualize
#     '''
    
#     data_dict = get_dataset()
#     for featname_loc in ['brt', 'ofg']:
#         if featname_loc=='brt':
#             labelname = 'dark'
#         elif featname_loc=='ofg':
#             labelname = 'take-off'
#         else:
#             labelname = 'nothing...'


#         for featname_frm in ['avr']:#['cnt', 'avr']:
#             for featname_vid in ['frm','avr']:

#                 data_dict_tmp = copy.copy(data_dict)
#                 experiment_title = f' Experiment-{featname_loc}-{featname_frm}-{featname_vid}'
#                 f_vid_list = list()
#                 for ifile, filename in enumerate(data_dict_tmp.keys()):
                    
#                     print(f'\r{experiment_title}-{ifile:04d}', end='')
#                     datum_dict = data_dict_tmp[filename]
                    
#                     # Video Load
#                     vid, _ = video_read(datum_dict['filepath'])
#                     # Local-Frame-Video Feature Generation
#                     f_loc = feat_raw(vid, featname_loc)
                    
#                     f_frm = feat_frm(f_loc, featname_frm, thr_=0.)
#                     f_vid = feat_vid(f_frm, featname_vid, val_=0.5)
                    
#                     # save result
#                     data_dict_tmp[filename]['feat'] = f_vid
                
#                 thr_, acc_dict = set_thre(data_dict, target_label=labelname)
#                 acc_show = max(acc_dict['all'])*100
#                 print(f'\n -> acc: {acc_show:2.1f}%%')
#                 visualize_data(data_dict_tmp, experiment_title=experiment_title, acc_dict=acc_dict, thr_=thr_)


import os
import glob

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

if __name__ == "__main__":
    from videoio import video_load
    from feat_loc import (
        feat_optflow_gf,
        get_feat_vid,
        get_feat_frm,
        get_feat_loc,
        get_feat_norm,
    )
    import pickle
    import matplotlib.pyplot as plt
    import pandas as pd
else:
    from imgproc.videoio import video_load
    from imgproc.feat_loc import (
        feat_optflow_gf,
        get_feat_vid,
        get_feat_frm,
        get_feat_loc,
        get_feat_norm,
    )


class VidNeg:
    def __init__(
        self,
        norm_thr={
            "loc": {"drk": 255.0, "mtl": 16.0},
            "frm": {"avr": 0.0},
            "vid": {"avr": 0.5},
        },
        videoname: str = "",
    ):

        self.thr_ = norm_thr

        self.vid_address = videoname
        self.vid = np.zeros(0)

        self.feat_loc = dict()
        for featname in list(self.thr_["loc"].keys()):
            self.feat_loc[featname] = 0.0
        self.feat_frm = dict()
        for featname in list(self.thr_["loc"].keys()):
            self.feat_frm[featname] = 0.0
        self.feat_vid = dict()
        for featname in list(self.thr_["loc"].keys()):
            self.feat_vid[featname] = 0.0

    def load_video(self, target_size=(320, 240)):
        self.vid, _ = video_load(self.vid_address, target_size=(320, 240))

    def compute_neg(
        self,
    ):

        for featname_loc in list(self.thr_["loc"].keys()):
            f_loc = self.get_feat_loc(
                self.vid,
                feat_name=featname_loc,
                feat_thr=self.thr_["loc"][featname_loc],
            )
            self.feat_loc[featname_loc] = self.get_feat_norm(
                f_loc, 0.0, self.thr_["loc"][featname_loc]
            )
            for featname_frm in list(self.thr_["frm"].keys()):
                f_frm = self.get_feat_frm(
                    self.feat_loc[featname_loc],
                    featname_frm=featname_frm,
                    feat_thr=self.thr_["frm"][featname_frm],
                )
                self.feat_frm[featname_loc] = f_frm
                for featname_vid in list(self.thr_["vid"].keys()):
                    f_vid = self.get_feat_vid(
                        self.feat_frm[featname_loc],
                        featname_vid=featname_vid,
                        feat_thr=self.thr_["vid"][featname_vid],
                    )
                    self.feat_vid[featname_loc] = f_vid

    def get_feat_vid(self, feat_frm: np.ndarray, featname_vid: str, feat_thr: float):
        """
        compute negativeness of the video (noiseness)
        """
        f_vid = get_feat_vid(feat_frm, featname_vid=featname_vid, feat_thr=feat_thr)

        return f_vid

    def get_feat_frm(self, feat_loc, featname_frm: str, feat_thr: float = 0.0):

        # feat_name: avr, cnt_thr,
        feat_ = get_feat_frm(feat_loc, featname_frm=featname_frm, feat_thr=feat_thr)

        return feat_

    def get_feat_loc(self, vid, feat_name: str, feat_thr: float = 255.0):
        """
        args: ndarray-4d, nfrm x H x W (0.~1.)
        returns: ndarray-3d, nfrm x H x W (0~1 float)
        """
        feat_ = get_feat_loc(vid, feat_name=feat_name, feat_thr=feat_thr)

        return feat_

    def get_feat_norm(self, feat, min_val: float = 0.0, max_val: float = 1.0):

        feat_ = get_feat_norm(feat, min_val=min_val, max_val=max_val)

        return feat_


def dataload(
    path_data: str,
    sheetid,
    label_dict,
):

    label_pos = label_dict['pos']
    label_neg = label_dict['neg']
    
    if os.path.exists(path_data + "/database.pickle"):
        with open(path_data + "/database.pickle", "rb") as f:
            db_pd_load = pickle.load(f)
    else:
        db_pd, db_pd_pn, db_pd_detail = load_gsheet(sheetid, label_dict)
        with open(path_data + "/database.pickle", "wb") as f:
            pickle.dump(db_pd, f, pickle.HIGHEST_PROTOCOL)
        with open(path_data + "/database.pickle", "wb") as f:
            db_pd_load = pickle.load(f)

    label_pos = [
        columnname for columnname in db_pd_load.columns if columnname in label_pos
    ]
    label_neg = [
        columnname for columnname in db_pd_load.columns if columnname in label_neg
    ]

    db_pd = db_pd_load
    # db_pd = pd.DataFrame(columns=["pos", "neg"])

    # db_pd["pos"] = (db_pd_load[label_pos] == "1").sum(axis=1) > 0
    # db_pd["neg"] = (db_pd_load[label_neg] == "1").sum(axis=1) > 0

    return db_pd


def savefeat(
    path_data: str,
    thr_loc_list: list = [16.0, 32.0],
    thr_tim_list: list = [150, 300, 450],
):
    videonames = glob.glob(f"{path_data}/**.mp4")

    nvid = len(videonames)
    nfrm = 1000
    of_max = 400
    
    if not os.path.exists(path_data + "/feat_vid.pickle"):
        # feat_frm_all = dict()
        # feat_frm_all["drk"] = np.zeros((len(thr_loc_list), nvid, nfrm))
        # feat_frm_all["mtl"] = np.zeros((len(thr_loc_list), nvid, nfrm))
        feat_vid_all = dict()
        feat_vid_all["drk"] = np.zeros((len(thr_loc_list), len(thr_tim_list), nvid))
        feat_vid_all["mtl"] = np.zeros((len(thr_loc_list), len(thr_tim_list), nvid))

        for ivid, videoname in enumerate(videonames):

            feat_filename = videoname.split(".")[0] + ".pickle"

            if not os.path.exists(feat_filename):
                print(
                    f"\r{ivid:03d}/{len(videonames):03d} - video processing... {videoname}",
                    end="",
                )
                video = VidNeg(videoname=videoname)
                video.load_video(target_size=(320, 240))
                feat_loc = dict()
                feat_loc["mtl"] = get_feat_loc(
                    video.vid[:, :, :, 0], feat_name="mtl", feat_thr=1000.0
                )
                feat_loc["drk"] = get_feat_loc(
                    video.vid[:, :, :, 0], feat_name="drk", feat_thr=1000.0
                )
                with open(feat_filename, "wb") as f:
                    pickle.dump(feat_loc, f, pickle.HIGHEST_PROTOCOL)

            else:
                print(
                    f"\r{ivid:03d}/{len(videonames):03d} - feature loading... {videoname}",
                    end="",
                )
                with open(feat_filename, "rb") as f:
                    feat_loc = pickle.load(f)

            print(
                f"\r{ivid:03d}/{len(videonames):03d} - computing... {videoname}",
                end="",
            )
            for ithrloc, feat_thr_loc in enumerate(thr_loc_list):
                for featname_loc in ["drk", "mtl"]:
                    
                    feat_loc_nrm = get_feat_norm(
                        feat_loc[featname_loc], 0.0, feat_thr_loc
                    )
                    
                    feat_frm = get_feat_frm(
                        feat_loc_nrm, featname_frm="avr", feat_thr=0.0
                    )

                    for ithrtim, feat_thr_time in enumerate(thr_tim_list):

                        feat_vid = get_feat_vid(
                            feat_frm, featname_vid="avr", feat_thr=feat_thr_time
                        )

                        print(
                            f"\r{ivid:03d}/{len(videonames):03d} - done {videoname}",
                            end="",
                        )
                        feat_vid_all[featname_loc][ithrloc, ithrtim, ivid] = feat_vid

        with open(path_data + "/feat_vid.pickle", "wb") as f:
            pickle.dump(feat_vid_all, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(path_data + "/feat_vid.pickle", "rb") as f:
            feat_vid_all = pickle.load(f)
    feat_frm_all = []
    return feat_frm_all, feat_vid_all


def train(path_data: str, sheetid:str):

    neglist = ["Drk", "MotionLess", "TimeLimit", "Etc"]
    poslist = ["Blister", "Bottle", "Sachet"]
    
    label_dict = dict()
    label_dict['pos'] = poslist
    label_dict['neg'] = neglist
    # neglist = ["Drk", "MotionLess", "TimeLimit", ""]
    db_pd = dataload(path_data, sheetid, label_dict)

    thr_loc_list = [8.0, 16.0, 32.0, 64.0]
    thr_tim_list = [0, 150, 300, 450, 600]
    # feat_thr_loc = 16.0
    # feat_thr_time = 100
    feat_frm_all, feat_vid_all = savefeat(path_data, thr_loc_list, thr_tim_list)

    # irow_ = np.where((db_pd[poslist]=="1").astype('float').sum(axis=1)>1)[0]
    # for irow in irow_:
    #     db_pd.iloc[irow][poslist[0]]="1"
    #     for label in poslist[1:]:
    #         db_pd.iloc[irow][label]=""
    # irow_ = np.where((db_pd[neglist]=="1").astype('float').sum(axis=1)>1)[0]
    # for irow in irow_:
    #     db_pd.iloc[irow][neglist[0]]="1"
    #     for label in neglist[1:]:
    #         db_pd.iloc[irow][label]=""
    
    irow_ = np.where((db_pd['TimeLimit']=="1").astype(float))[0]
    for irow in irow_:
        db_pd.iloc[irow]["Etc"]="1"
        db_pd.iloc[irow]["TimeLimit"]=""
    
    db_pd_tmp = pd.DataFrame(columns=['pos','Drk','Mtl','Etc'])
    db_pd_tmp['pos'] = ((db_pd[poslist]=="1").sum(axis=1)>0).astype(float)
    db_pd_tmp['Drk'] = ((db_pd['Drk']=="1")>0).astype(float)
    db_pd_tmp['Mtl'] = ((db_pd['MotionLess']=="1")>0).astype(float)
    db_pd_tmp['Etc'] = ((db_pd['Etc']=="1")>0).astype(float)
    
    
    gt_label = np.zeros(db_pd_tmp.shape[0])
    for ilabel, labelname in enumerate(db_pd_tmp.columns):
        gt_label = gt_label + (db_pd_tmp[labelname]==1.).astype('float')*ilabel
    
    gt_label_pn = db_pd_tmp['Drk'] + db_pd_tmp['Mtl'] + db_pd_tmp['Etc']*2
    
    scrthrlist = 0.05 * np.arange(10, 21)
    
    for ithrloc, feat_thr_loc in enumerate(thr_loc_list):
        for ithrtim, feat_thr_time in enumerate(thr_tim_list):
            plt.figure(figsize=(20, 20))
            plt.scatter(
                feat_vid_all['drk'][ithrloc, ithrtim, gt_label==0.], feat_vid_all['mtl'][ithrloc, ithrtim, gt_label==0.], color="blue"
            )
            plt.scatter(
                feat_vid_all['drk'][ithrloc, ithrtim, gt_label==1.], feat_vid_all['mtl'][ithrloc, ithrtim, gt_label==1.], color="red"
            )
            plt.scatter(
                feat_vid_all['drk'][ithrloc, ithrtim, gt_label==2.], feat_vid_all['mtl'][ithrloc, ithrtim, gt_label==2.], color="red"
            )
            plt.scatter(
                feat_vid_all['drk'][ithrloc, ithrtim, gt_label==3.], feat_vid_all['mtl'][ithrloc, ithrtim, gt_label==3.],
                color="lightgray",
            )
            plt.xlabel('drk')
            plt.ylabel('mtl')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.savefig(
                path_data
                + f"/featvid_scatter_{int(feat_thr_loc):02d}_{feat_thr_time:03d}.png"
            )
            plt.close("all")
            print("============================================================")
            for thr_scr_drk in scrthrlist:
                for thr_scr_mtl in scrthrlist:
                    prd_drk = feat_vid_all["drk"][ithrloc, ithrtim, :] >= thr_scr_drk
                    prd_mtl = feat_vid_all["mtl"][ithrloc, ithrtim, :] >= thr_scr_mtl
                    prd_pos = 1-(prd_mtl+prd_drk)
                    prd_label = prd_drk+prd_mtl*2
                    prd_label_pn = 1-prd_drk*prd_mtl

                    confmat = confusion_matrix(gt_label, prd_label)
                    
                    acc = confmat[:, :].diagonal().sum() / confmat[:, :].sum() * 100
                    pre = confmat[0, 0] / confmat[:, 0].sum() * 100
                    rec = confmat[0, 0] / confmat[0, :].sum() * 100
                    
                    confmat_pn = confusion_matrix(gt_label_pn, prd_label_pn)
                    acc_pn = confmat_pn[:2, :2].diagonal().sum() / confmat_pn[:, :].sum() * 100
                    pre_pn = confmat_pn[0, 0] / (confmat_pn[:2, 0].sum()+np.finfo(float).eps) * 100
                    rec_pn = confmat_pn[0, 0] / (confmat_pn[0, :2].sum()+np.finfo(float).eps) * 100
                    
                    print(
                        f"{feat_thr_loc}, {feat_thr_time}, {thr_scr_drk:.2f}, {thr_scr_mtl:.2f} | alllabel {acc:03.1f}%, {pre:03.1f}%, {rec:03.1f}% | posneg {acc_pn:03.1f}%, {pre_pn:03.1f}%, {rec_pn:03.1f}%"
                    )

    a = 1
    # plt.figure()
    # plt.scatter(
    #     np.where(db_pd["pos"] == "1"),
    #     feat_vid[np.array(db_pd["pos"] == "1")],
    #     color="blue",
    # )
    # plt.scatter(
    #     np.where(db_pd["neg"] == "1"),
    #     feat_vid[np.array(db_pd["neg"] == "1")],
    #     color="red",
    # )
    # plt.ylim([0, 1])
    # plt.savefig(f"feat-vid.png")
    # plt.close("all")


if __name__ == "__main__":
    
    testid = 1
    
    if testid == 1:
        title_exp = "EN-EXP-001"
        sheetid = "1nuI80IcDmZPRA1VRH8OeJbS7kEuh_72XDAApC1bJWQE"
        path_data = "/root/video/etc/EN_data_001_combined"
        colinfo = [i for i in range(1, 10)]
    elif testid == 2:
        title_exp = "EN-EXP-002"
        sheetid = "1rHc_FGxAK6IBoYesQdfBsbb1yRjG03vA8XxAGUuw9KE"
        path_data = "/root/video/etc/EN_data_002_SNU"
        colinfo = [i for i in range(1, 8)]
    else:
        raise ()

    
    train(path_data, sheetid)
    # video_list = glob.glob("/root/video/etc/EN_data_001_combined/21*.mp4")
    # for videoname in video_list:
    #     video = VidNeg(videoname=videoname)
    #     video.load_video()
    #     video.compute_neg()

    #     print(video.vid_address.split("/")[-1], video.feat_vid)

import cv2
import numpy as np


nDevice = cv2.cuda.getCudaEnabledDeviceCount()
print(nDevice)


def feat_optflow_gf(vid_gry):

    prev = None
    feat_of = np.zeros_like(vid_gry).astype(np.float32)
    try:
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
                flow_mag = np.sqrt(np.sum(flow**2, axis=2))

                feat_of[ifrm, :] = flow_mag  # / ofg_thre

                prev = gray
    except:
        print(f"CANNOT CALCULATE OPTICALFLOW: {featname_frm}; type must be avr/cnt")
        raise NotImplementedError

    return feat_of


def get_feat_vid(feat_frm: np.ndarray, featname_vid: str, feat_thr: float):
    """
    compute negativeness of the video (noiseness)
    """
    if feat_thr < 1:
        feat_thr = 100
        # nfrm = feat_frm.shape[0]
        # feat_thr = np.floor(nfrm * feat_thr).astype(np.int32)
    if feat_frm.shape[0] <= feat_thr:
        f_vid = 0.0
    else:
        f_vid = 1.0 - (feat_frm[feat_thr:].mean())

    # nfrm = feat_frm.shape[0]
    # feat_frm_rvs = np.flip(feat_frm, axis=0)
    # feat_frm_cumsum = feat_frm_rvs.cumsum()

    # if featname_vid == "avr":
    #     nfrm_target = np.floor(nfrm * feat_thr).astype(np.int32)
    #     f_vid = 1.0 - (feat_frm_cumsum[nfrm_target] / nfrm_target)
    #     # negativeness = 1-motionness or 1-brightness
    # elif featname_vid == "frm":
    #     ifrm_over_thr = np.where(feat_frm_cumsum > feat_thr)
    #     # negativeness = nfrm_enough_motion or nfrm_enough_brightness
    #     try:
    #         f_vid = ifrm_over_thr[0][0] / nfrm
    #     except:
    #         f_vid = 1.0
    # else:
    #     print(f"WRONG TYPE VIDEO FEATURE: {featname_frm}; type must be avr/cnt")
    #     raise NotImplementedError

    return f_vid


def get_feat_frm(feat_loc, featname_frm: str, feat_thr: float = 0.0):

    nfrm, h, w = feat_loc.shape
    # feat_name: avr, cnt_thr,

    try:
        if featname_frm == "avr":
            """
            Average the pixel values over Thr
            """
            # feat_loc = get_feat_norm(feat_loc, min_val=feat_thr, max_val=1.0)
            feat_ = feat_loc.mean(axis=(1, 2))
        elif featname_frm == "cnt":
            """
            Count the pixels over Thr
            """
            feat_ = (feat_loc > feat_thr).mean(axis=(1, 2))
        else:
            print(f"WRONG TYPE FRAME FEATURE: {featname_frm}; type must be avr/cnt")
            raise NotImplementedError
    except:
        print(f"CANNOT GENREATE FRAME FEATURE: {featname_frm}, {feat_thr}")
        raise NotImplementedError

    return feat_


def get_feat_loc(vid, feat_name: str, feat_thr: float = 255.0):
    """
    args: ndarray-4d, nfrm x H x W (0.~1.)
    returns: ndarray-3d, nfrm x H x W (0~1 float)
    """
    try:
        if feat_name == "drk":
            vid = vid.astype(np.float32)
            feat_ = vid[:, :, :, 0]
            # feat_ = self.get_feat_norm(feat_, 0.0, 255.0)
        elif feat_name == "mtl":
            vid = vid[:, :, :, 0]
            feat_ = feat_optflow_gf(vid)
            # feat_ = self.get_feat_norm(feat_, 0.0, feat_thr)
        else:
            print(f"WRONG TYPE LOCAL FEATURE: {feat_name}; type must be brt/ofg")
            raise NotImplementedError
    except:
        print(f"CANNOT GENREATE FRAME FEATURE: {feat_name}, {feat_thr}")
        raise NotImplementedError

    return feat_


def get_feat_norm(feat, min_val: float = 0.0, max_val: float = 1.0):

    feat_clipped = feat.clip(min_val, max_val)
    feat_ = (feat_clipped - min_val) / (max_val - min_val)

    return feat_

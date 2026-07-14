import torch
import math
import numpy as np
import cv2
import random
import os

def low_pass_filter(delta_time, log_new_frame, lp_log_frame0, inten01, cutoff_hz=0):
    if cutoff_hz <= 0:
        # unchange
        return log_new_frame, log_new_frame

    tau = 1/(math.pi*2*cutoff_hz)
    eps = inten01*(delta_time/tau)
    eps = torch.clamp(eps, min=0, max=1)

    new_lp_log_frame0 = (1-eps)*lp_log_frame0+eps*log_new_frame
    new_lp_log_frame1 = lp_log_frame0

    return new_lp_log_frame0, new_lp_log_frame1

def lowpass_learn_driver(all_gray, model, exp, cutoff_hz):
    ret_all_gray = []
    lp_log_frame0 = all_gray[0]
    for i in range(1, all_gray.shape[0]):
        inten01 = model.crf_evs(all_gray[i].unsqueeze(-1)).squeeze(-1)
        lp_log_frame0, lp_log_frame1 = low_pass_filter(exp, log_new_frame=all_gray[i], lp_log_frame0=lp_log_frame0, inten01=inten01, cutoff_hz=cutoff_hz)
        ret_all_gray.append(lp_log_frame1)
    ret_all_gray.append(lp_log_frame0)
    return torch.stack(ret_all_gray)

def event_loss_call(all_rgb, event_data, color_mask, bin_num, model, exp, pre=False, cutoff_hz=30):
    loss = []
    color_mask = color_mask.repeat(bin_num + 1, 1)
    all_gray = (all_rgb.view(-1, 3) * color_mask).sum(dim=1).view((bin_num + 1, -1))
    all_gray = lowpass_learn_driver(all_gray, model, exp, cutoff_hz=cutoff_hz)
    if pre:
        all_gray = all_gray[1:]

    for its in range(bin_num):
        start = its
        end = its + 1

        thres = (all_gray[end] - all_gray[start]) / 0.3

        event_cur = event_data[start]
        pos = event_cur > 0
        neg = event_cur < 0
        noe = event_cur == 0

        loss_pos = ((thres * pos) - ((event_cur) * pos)) ** 2
        loss_neg = ((thres * neg) - ((event_cur) * neg)) ** 2
        loss_noe = ((thres * noe) - (event_cur * noe)) ** 2

        loss.append(torch.mean(loss_pos + loss_neg + loss_noe))
    event_loss = torch.mean(torch.stack(loss, dim=0), dim=0)
    return event_loss

def low_pass_filter_fix(delta_time, log_new_frame, lp_log_frame0, inten01):
    eps = inten01*delta_time*1000
    eps = torch.clamp(eps, min=0, max=1)

    new_lp_log_frame0 = (1-eps)*lp_log_frame0+eps*log_new_frame
    new_lp_log_frame1 = lp_log_frame0

    return new_lp_log_frame0, new_lp_log_frame1

def lowpass_learn_driver_fix(all_gray, model, exp):
    ret_all_gray = []
    lp_log_frame0 = all_gray[0]
    for i in range(1, all_gray.shape[0]):
        inten01 = model.crf_evs(all_gray[i].unsqueeze(-1)).squeeze(-1)
        lp_log_frame0, lp_log_frame1 = low_pass_filter_fix(exp, log_new_frame=all_gray[i], lp_log_frame0=lp_log_frame0, inten01=inten01)
        ret_all_gray.append(lp_log_frame1)
    ret_all_gray.append(lp_log_frame0)
    return torch.stack(ret_all_gray)

def event_loss_call_fix(all_rgb, event_data, color_mask, bin_num, model, exp, pre=False):
    loss = []
    color_mask = color_mask.repeat(bin_num + 1, 1)
    all_gray = (all_rgb.view(-1, 3) * color_mask).sum(dim=1).view((bin_num + 1, -1))
    all_gray = lowpass_learn_driver_fix(all_gray, model, exp)
    if pre:
        all_gray = all_gray[1:]

    for its in range(bin_num):
        start = its
        end = its + 1

        thres = (all_gray[end] - all_gray[start]) / 0.3

        event_cur = event_data[start]
        pos = event_cur > 0
        neg = event_cur < 0
        noe = event_cur == 0

        loss_pos = ((thres * pos) - ((event_cur) * pos)) ** 2
        loss_neg = ((thres * neg) - ((event_cur) * neg)) ** 2
        loss_noe = ((thres * noe) - (event_cur * noe)) ** 2

        loss.append(torch.mean(loss_pos + loss_neg + loss_noe))
    event_loss = torch.mean(torch.stack(loss, dim=0), dim=0)
    return event_loss

class bin_num_eval(object):
    '''

    '''

    def __init__(self, views_num, dir, parent=None):
        self.path = dir
        if os.path.exists(dir + "/bin_num_np.npy"):
            self.bin_num_np = np.load(dir + "/bin_num_np.npy")
            self.bin_num_np_counter = np.load(dir + "/bin_num_np_counter.npy")
            self.bin_num_np_ave = np.load(dir + "/bin_num_np_ave.npy")
        else:
            self.bin_num_np = np.zeros((views_num))
            self.bin_num_np_counter = np.zeros((views_num))
            self.bin_num_np_ave = np.zeros((views_num))

    def update(self, img_i, offset, s):
        self.bin_num_np[img_i] += offset
        self.bin_num_np_counter[img_i] += s
        self.bin_num_np_ave[img_i] = self.bin_num_np[img_i] / self.bin_num_np_counter[img_i]
        #print(img_i, offset / s)

    def get_bin_flag(self, img_i):
        if self.bin_num_np_ave[img_i] <= 10:
            return 0
        elif self.bin_num_np_ave[img_i] <= 15:
            return 1
        else:
            return 2

    def get_bin_flag_ellff(self, img_i):
        if self.bin_num_np_ave[img_i] <= 5:
            return 0
        elif self.bin_num_np_ave[img_i] <= 10:
            return 1
        else:
            return 2

    def save(self):
        np.save(self.path + "/bin_num_np.npy", self.bin_num_np)
        np.save(self.path + "/bin_num_np_counter.npy", self.bin_num_np_counter)
        np.save(self.path + "/bin_num_np_ave.npy", self.bin_num_np_ave)

    def mean_pixels_offset_cal(self, pose_w2c, K, depthes, select_blur_coords, rays_os, rays_ds, img_i, near, far):

        flag = (depthes[0] >= near) & (depthes[0] <= far)
        for i in range(1, len(depthes) - 1):
            flag = (depthes[i] >= near) & (depthes[i] <= far) & flag
        s = flag.sum().item()
        flag = flag.view(-1)

        offsets = 0
        x0, y0 = select_blur_coords[:, 0], select_blur_coords[:, 1]
        for i in range(len(depthes) - 1):
            x, y = self.pixels_offset_estimate(
                rays_os[i][select_blur_coords[:, 0], select_blur_coords[:, 1]],
                rays_ds[i][select_blur_coords[:, 0], select_blur_coords[:, 1]],
                rays_os[i + 1][select_blur_coords[:, 0], select_blur_coords[:, 1]],
                torch.Tensor(pose_w2c[i + 1]), torch.Tensor(K), depthes[i])
            #print(x[0] - x0[0], y[0] - y0[0])
            offset = torch.sum(torch.sqrt((x - x0) ** 2 + (y - y0) ** 2) * flag)
            offsets += offset.item()
        self.update(img_i, offsets / len(depthes), s)
        return

    # 两帧间像素偏移值计算
    def pixels_offset_estimate(self, o0, d0, o1, w2c, K, depth):
        '''
        输入
        给定基准相机位姿： o0, d0
        给定偏移相机位姿： o1
        给定偏移相机w2c： w2c
        给定相机内参： K
        给定深度值： depth

        输出
        下一帧 x 坐标
        下一帧 y 坐标
        '''
        depth = depth.view(-1)
        p = o0 + torch.stack([d0[:, 0] * depth, d0[:, 1] * depth, d0[:, 2] * depth], dim=-1)
        xxx = torch.norm((p - o1), p=2, dim=1)
        zzz = torch.t(xxx.repeat(3, 1))
        #zz = (p - o1)[:,2]
        #zzz = torch.t(zz.repeat(3,1))
        new_d = (p - o1).div(zzz)
        c_d = torch.matmul(new_d, torch.t(w2c))
        #c_d_new = torch.t(torch.matmul(w2c, torch.t(new_d)))
        yyy = torch.t(c_d[:,2].expand(3,-1))
        c_d = -c_d / yyy
        y, x = c_d[:, 0] * K[0][0] + K[0][2], -(c_d[:,1] * K[1][1]) + K[1][2]
        return x, y

class bin_num_eval_frame(object):
    '''

    '''

    def __init__(self, views_num, dir, parent=None):
        self.path = dir
        if os.path.exists(dir + "/bin_num_frame.pt"):
            self.bin_num_frame = torch.load(dir + "/bin_num_frame.pt")
            self.bin_num = torch.mean((self.bin_num_frame / 5).view(100, 640000), dim=1)
            print("exist!")
        else:
            self.bin_num_frame = torch.zeros((views_num, 800, 800))

    def get_bin_flag(self, img_i):
        if self.bin_num[img_i] <= 10:
            return 0
        elif self.bin_num[img_i] <= 15:
            return 1
        else:
            return 2

    def get_bin_flag_ellff(self, img_i):
        if self.bin_num[img_i] <= 5:
            return 0
        elif self.bin_num[img_i] <= 10:
            return 1
        else:
            return 2

    def save(self):
        torch.save(self.path + "/bin_num_frame.pt", self.bin_num_frame)

    def mean_pixels_offset_cal(self, pose_w2c, K, depthes, select_blur_coords, rays_os, rays_ds, img_i, near, far, select):

        flag = (depthes[0] >= near) & (depthes[0] <= far)
        for i in range(1, len(depthes) - 1):
            flag = (depthes[i] >= near) & (depthes[i] <= far) & flag
        s = flag.sum().item()

        x0, y0 = select_blur_coords[:, 0], select_blur_coords[:, 1]
        frame = torch.zeros((640000))
        for i in range(len(depthes) - 1):

            x, y = self.pixels_offset_estimate(
                rays_os[i][select_blur_coords[:, 0], select_blur_coords[:, 1]],
                rays_ds[i][select_blur_coords[:, 0], select_blur_coords[:, 1]],
                rays_os[i + 1][select_blur_coords[:, 0], select_blur_coords[:, 1]],
                torch.Tensor(pose_w2c[i + 1]), torch.Tensor(K), depthes[i])
            #print(x[0] - x0[0], y[0] - y0[0]
            frame.scatter_add_(dim=0, index=torch.tensor(select)[0],
                               src=torch.sqrt((x - x0) ** 2 + (y - y0) ** 2) * flag)
            print("ok")

        self.bin_num_frame[img_i] = frame.view(800, 800)
        cv2.imwrite(self.path + "/event_bin_frame/{}.png".format(img_i), (frame.view(800, 800) * 2).cpu().numpy())
        return

    # 两帧间像素偏移值计算
    def pixels_offset_estimate(self, o0, d0, o1, w2c, K, depth):
        '''
        输入
        给定基准相机位姿： o0, d0
        给定偏移相机位姿： o1
        给定偏移相机w2c： w2c
        给定相机内参： K
        给定深度值： depth

        输出
        下一帧 x 坐标
        下一帧 y 坐标
        '''
        p = o0 + torch.stack([d0[:, 0] * depth, d0[:, 1] * depth, d0[:, 2] * depth], dim=-1)
        xxx = torch.norm((p - o1), p=2, dim=1)
        zzz = torch.t(xxx.repeat(3, 1))
        new_d = (p - o1).div(zzz)
        c_d = torch.matmul(new_d, torch.t(w2c))
        #c_d_new = torch.t(torch.matmul(w2c, torch.t(new_d)))
        yyy = torch.t(c_d[:,2].expand(3,-1))
        c_d = -c_d / yyy
        y, x = c_d[:, 0] * K[0][0] + K[0][2], -(c_d[:,1] * K[1][1]) + K[1][2]
        return x, y


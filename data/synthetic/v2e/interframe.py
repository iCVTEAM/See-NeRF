import cv2

from v2ecore.slomo import SuperSloMo
import torch
import numpy as np

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def interframe(indir, outdir):
    batch_size = 16
    preview = False

    slomo = SuperSloMo(
        model="./input/SuperSloMo39.ckpt",
        auto_upsample=False,
        upsampling_factor=3,
        video_path=None,
        vid_orig=None,
        vid_slomo=None,
        preview=preview, batch_size=batch_size)

    interpTimes, avgUpsamplingFactor = slomo.interpolate(indir, outdir, (346, 260))


outdir = "G:/NeRF/3-ERGB-NeRF/0-NeRF_Data/my_data/event_box/r_1/"
indir = "G:/NeRF/3-ERGB-NeRF/0-NeRF_Data/my_data/in_box/r_0/"
interframe(indir, outdir)
'''
for i in range(5):
    img = cv2.imread("G:/NeRF/3-ERGB-NeRF/0-NeRF_Data/my_data/in_box/r_0/{0:03d}".format(i))
    np.save("G:/NeRF/3-ERGB-NeRF/0-NeRF_Data/my_data/in_box/r_0/{0:03d}.npy".format(i),img)
'''
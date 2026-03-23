import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

tonemap = lambda x : (np.log(np.clip(x, 0, 1) * 5000 + 1 ) / np.log(5000 + 1)).astype(np.float32)

def tonemap_exp(img, exp_time):
    img = img[:, :, :3]
    img_new = ((exp_time * img) / (exp_time * img + 1)) ** (1 / 2.2)
    return img_new

scenes = ["BathRoom", "CatRoom", "DinningRoom", "DogRoom", "Sofa", "Sponza", "ToyRoom", "WarmRoom"]


# Generate the images for training, testing, and event simulation
print("------Stage 1: Training, Testing, and Event Simulation Image Folders Generation------")

for scene in scenes:
    print("------Processing {} Scene------".format(scene))

    # dir settings
    base_path = "./{}/".format(scene)
    os.makedirs(base_path + "train_blurry", exist_ok=True)
    os.makedirs(base_path + "train_sharp", exist_ok=True)
    os.makedirs(base_path + "train_event", exist_ok=True)
    os.makedirs(base_path + "test_ldr", exist_ok=True)
    os.makedirs(base_path + "test_hdr", exist_ok=True)

    # training view generation
    print("Training Data Processing")
    for i in range(18):
        print("View: {}".format(i))

        # blurry synthesis in HDR raw domain
        img_hdr_blur = np.zeros((400, 400, 3))
        for j in range(17):
            filename = base_path + "train/{0:04d}/HDRImg_{1:03d}.exr".format(i, j)
            img_hdr = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            # rgba to rgb, 22 is brightness factor of white bg
            img_hdr = img_hdr[..., :3] * img_hdr[..., -1:] + (1. - img_hdr[..., -1:]) * 22
            img_hdr = cv2.resize(img_hdr, (400, 400))
            img_hdr_blur = img_hdr_blur + img_hdr
        img_hdr_blur = img_hdr_blur / 17
        # generate LDR blurry training images with exposure time t_0 ~ t_4
        for t in range(5):
            img_ldr_blur = tonemap_exp(img_hdr_blur, 2 ** (2 * (t - 2)))
            cv2.imwrite(base_path + "train_blurry/{0:03d}_t_{1}.png".format(i, t), img_ldr_blur * 255)

        # generate LDR sharp training images with exposure time t_0 ~ t_4
        # for image-based HDR NVS methods w/o Deblurring effect
        filename = base_path + "train/{0:04d}/HDRImg_{1:03d}.exr".format(i, 8)  # select the middle frame for training
        img_hdr_sharp = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        # rgba to rgb, 22 is brightness factor of white bg
        img_hdr_sharp = img_hdr_sharp[..., :3] * img_hdr_sharp[..., -1:] + (1. - img_hdr_sharp[..., -1:]) * 22
        img_hdr_sharp = cv2.resize(img_hdr_sharp, (400, 400))
        for t in range(5):
            img_ldr_sharp = tonemap_exp(img_hdr_sharp, 2 ** (2 * (t - 2)))
            cv2.imwrite(base_path + "train_sharp/{0:03d}_t_{1}.png".format(i, t), img_ldr_sharp * 255)

        # generate HDR raw images for event simulation in .npy format as input of v2e
        os.makedirs(base_path + "train_event/{0:03d}".format(i), exist_ok=True)
        for j in range(18):
            filename = base_path + "train/{0:04d}/HDRImg_{1:03d}.exr".format(i, j)
            img_hdr = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            # rgba to rgb, 22 is brightness factor of white bg
            img_hdr = img_hdr[..., :3] * img_hdr[..., -1:] + (1. - img_hdr[..., -1:]) * 22
            img_hdr = cv2.resize(img_hdr, (400, 400)) + 1e-6
            np.save(base_path + "train_event/{0:03d}/{1:03d}.npy".format(i, j), img_hdr * 255)



    # testing view data generation
    print("Testing Data Processing")
    for i in range(17):
        print("View: {}".format(i))

        # generate LDR sharp test images with exposure time t_0 ~ t_4
        for t in range(5):
            filename = base_path + "test/{0:04d}/HDRImg.exr".format(i)
            img_hdr_test = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            # rgba to rgb, 22 is brightness factor of white bg
            img_hdr_test = img_hdr_test[..., :3] * img_hdr_test[..., -1:] + (1. - img_hdr_test[..., -1:]) * 22
            img_hdr_test = cv2.resize(img_hdr_test, (400, 400))
            img_ldr_test = tonemap_exp(img_hdr_test, 2 ** (2 * (t - 2)))
            cv2.imwrite(base_path + "test_ldr/{0:03d}_l_{1}.png".format(i, t), img_ldr_test * 255)

        # generate HDR sharp test images
        filename = base_path + "test/{0:04d}/HDRImg.exr".format(i)
        img_hdr_test = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        # rgba to rgb, 22 is brightness factor of white bg
        img_hdr_test = img_hdr_test[..., :3] * img_hdr_test[..., -1:] + (1. - img_hdr_test[..., -1:]) * 22
        img_hdr_test = cv2.resize(img_hdr_test, (400, 400))
        cv2.imwrite(base_path + "test_hdr/{0:03d}.exr".format(i), img_hdr_test)

    print("Finished!")



# Call v2e_color.py for event generation
print("------Stage 2: Event Generation with v2e------")

for scene in scenes:
    print("------Processing {} Scene------".format(scene))

    base_dir = "./{}/train_event/".format(scene)

    for i in range(18):
        print("View: {}".format(i))

        inname = base_dir + "{:03d}/".format(i)
        outname = base_dir + "{:03d}/".format(i)

        if os.path.isdir(outname) is False:
            try:
                os.makedirs(outname, exist_ok=True)
            except:
                print("CREATE: " + outname + "FAILED")

        command = "python ./v2e/v2e_color.py --ignore-gooey --output_folder=" + outname \
                + " --unique_output_folder=False --overwrite --disable_slomo --output_height=400 --output_width=400 --input=" \
                + inname + " --input_frame_rate=1000 --no_preview --dvs_aedat2=None --auto_timestamp_resolution=False --dvs_params=noisy"

        os.system(command)
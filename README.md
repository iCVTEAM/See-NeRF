# Seeing through Light and Darkness: Sensor-Physics Grounded Deblurring HDR NeRF from Single-Exposure Images and Events
This is an official PyTorch implementation of the See-NeRF. Click [here](https://icvteam.github.io/See-NeRF.html) to see the video and supplementary materials in our project website.

# 1. Method Overview

# 2. Installation 

# 3. Code
The code will be released soon.

# 4. Data

## 4.1 Synthetic Data
We provide the **Blender Data**, **Raw & Processed Data**, and the **Train & Test Data** of the synthetic scenes. Please down load them [here](https://drive.google.com/drive/folders/1kF0WjsWkyTM-fhHqKdUwJKtpxYXSlJWw?usp=sharing).

### 4.1.1 Blender Data
We provide the original blender files to generate the **Raw Data** for further processing. The data is generated with [Blender 3.4](https://www.blender.org/download/releases/3-4/). Note that unlike [E<sup>2</sup>NeRF](https://github.com/iCVTEAM/E2NeRF), we didn't use the [Camera Shakify Plugin](https://github.com/EatTheFuture/camera_shakify?tab=readme-ov-file) to simulate camera shake here. Instead, we directly quantized and randomly generated camera shake trajectories as in [EBAD-NeRF](https://github.com/iCVTEAM/EBAD-NeRF). Therefore, if you try to generate the **Raw Data** using these blender files, it might differ slightly from the data we provided. The blender data here is for facilitating the in-depth future related research. If you simply want to use this dataset, please refer to the **Processed Data** and the **Train & Test Data**.

### 4.1.2 Raw Data
The **Raw Data** include two folders ("train" and "test") in each scene's folder. There are 18 and 17 folder in the "train" and "test" folders seperately, representing 18 training views and 17 testing views. There are 18 **raw .exr images** in each "train" folder, synthesized during the camera shaking. We also provide the corresponding depth maps for further evaluation. Note that we also provide the ground truth **.json pose data** generated from blender for further evaluation, but we didn't use these data for training and testing in See-NeRF.

### 4.1.3 Processed Data
The **Processed Data** include five folders ("train_event", "train_blurry", "train_sharp", "test_ldr", "test_hdr") in each scene's folder. If you want to generate the **Processed Data** from **Raw Data** by yourself, you can move the scene folders of **Raw Data** in to the "./data/synthetic/" and use the command below under the environment of [v2e](https://github.com/SensorsINI/v2e).

```
python ./data/synthetic/1-raw2processed.py
```

**"train_event":** There are 18 folders in this folder, representing 18 training views. Each folder contains 18 **raw .npy** image for event simulation and the raw events "v2e-dvs-events.txt" generated from the modified "color_v2e" simulater. Since the event simulation include random noise events, if you try to generate the raw events using our code, it might differ slightly from the raw events we provided.

**"train_blurry" & "train_sharp":** There are 90 (18 training views * 5 different exposure times) synthsized LDR blurry images and LDR sharp images in the two folder, seperately. Note that our See-NeRF only use the blurry images with exposure time *t<sub>2</sub>* for training. The rest images are for the compared methods and further research.

**"test_" & "test_hdr":** There are 85 (17 testing views * 5 different exposure times) sharp images and 17 HDR sharp images in these two folders, seperately. Note that we only use the LDR sharp images with exposure time *t<sub>1</sub>* and *t<sub>3</sub>* for the novel view novel exposure test. The rest images are for the further research.

### 4.1.4 Train & Test Data

The **Train & Test Data** can be generated from the **Processed Data** using the command below with [COLMAP](https://colmap.github.io/install.html) installed and you can train and test our See-NeRF directly with the **Train & Test Data**.
```
python ./data/synthetic/2-processed2train.py
```
The data include three folders ("images", "images_pose_etimation", "GT") and three files (events.pt, poses_bounds.npy, exp_times_test.npy) in each scene's folder.

**"images":** There are 18 single-exposure blurry LDR images with exposure time *t<sub>2</sub>* for training in this folder.

**"images_pose_estimation":** There are 90 (18 training views * 5 in each view) LDR pre-deblurred images with EDI model and 17 GT sharp LDR test images in this folder for the pose estimation with [COLMAP](https://colmap.github.io/install.html). Although *2-processed2train.py* include calling COLMAP for pose estimation , we recommand using its GUI Automatic Reconstruction manually with **Shared Intrinsics option activated**.

**"GT":** This folder contains 17 testing views of both LDR and HDR GT images. Note that for HDR evaluation, we use the [Photomatix Pro 7.1.2 (64-bit)](https://www.hdrsoft.com/download/photomatix-pro.html) with the "*Enhancer*" option to tonemap the HDR GT images and save them into the "/GT/hdr/PhotomatixConversions01" folder, and for LDR novel exposure evaluation, we only use the LDR images with exposure time *t<sub>1</sub>* and *t<sub>3</sub>*.

**"events.pt":** The processed event data to facilitate See-NeRF training.

**"poses_bounds.npy":** The pose data generated from the COLMAP output.

**"exp_times_test.npy":** The specific values of *t<sub>0</sub>*, *t<sub>1</sub>*, *t<sub>2</sub>*, *t<sub>3</sub>*, *t<sub>4</sub>*.

## 4.2 Real Data
We provide the **Raw Data** and the **Train & Test Data** of the Real scenes. Please down load them [here](https://drive.google.com/drive/folders/1ym68udecg_TJ5qgBAbM54-1gj6znqYVC?usp=sharing).

### 4.2.1 Raw Data
The **Raw Data** include the handheld DAVIS 346 captured blurry images and corresponding raw events (**.txt format**) and tripod-fixed DAVIS 346 captured sharp ground truth LDR images with exposure times *t<sub>0</sub>*-*t<sub>4</sub>*. This data is provided to facilitate the future related research. If you simply want train and test our See-NeRF, please refer to the **Train & Test Data**.

**"images_blurry" & "events":** In each scene's folder, there are "blurry_images" folders and corresponding raw "events" folders captured with exposure times *t<sub>0</sub>*-*t<sub>4</sub>* at 16 training views. Note that the blurry images with different exposure time at each training view are not strictly aligned because they are caputured with handheld camera. Therefore, we placed the blurred images and corresponding events with different exposure times into different folders.

**"images_gt_ldr":** This folder contains the ground truth LDR images with exposure times *t<sub>0</sub>*-*t<sub>4</sub>* at 28 testing views, 

**"exp_times_train.npy" & "exp_times_test.npy":** The specific values of *t<sub>0</sub>*, *t<sub>1</sub>*, *t<sub>2</sub>*, *t<sub>3</sub>*, *t<sub>4</sub>*.

### 4.2.2 Train & Test Data
The **Train & Test Data** can be generated from the **Raw Data**  using the command below with [COLMAP](https://colmap.github.io/install.html) installed. You can train and test our See-NeRF directly with the **Train & Test Data**.
```
python ./data/real/1-raw2train.py
```
The data include three folders ("images", "images_pose_etimation", "GT") and three files (events.pt, poses_bounds.npy, exp_times_test.npy) in each scene's folder.

**"images":** There are 16 single-exposure blurry LDR images with exposure time *t<sub>2</sub>* for training in this folder.

**"images_pose_estimation":** There are 80 (16 training views * 5 in each view) LDR pre-deblurred images with EDI model and 28 GT sharp LDR test images in this folder for the pose estimation using [COLMAP](https://colmap.github.io/install.html). Although *1-raw2train.py* include calling COLMAP for pose estimation , we recommand using its GUI Automatic Reconstruction manually with **Shared Intrinsics option activated**.

**"GT":** This folder contains 28 testing views of both LDR and HDR GT images. Since we cannot take the HDR image directly with the DAVIS 346 event camera, we use the Debevec algorithm to merge 5 LDR GT images with exposure times *t<sub>0</sub>*-*t<sub>4</sub>* to hdr GT image for each view, and we use the [Photomatix Pro 7.1.2 (64-bit)](https://www.hdrsoft.com/download/photomatix-pro.html) with the "*Compressor*" option to tonemap the HDR GT images and save them into the "/GT/hdr/PhotomatixConversions01" folder like the operation in synthetic data. Note that we only use views [1, 3, 5, 8, 10, 12, 15, 17, 19, 22, 24, 26] for the novel view synthesis evaluation and exposure time *t<sub>1</sub>* and *t<sub>3</sub>* for LDR novel exposure evaluation, because views [0, 2, 4, 6, 7, 9, 11, 13, 14, 16, 18, 20, 21, 23, 25, 27] are close to the training views.

**"events_offset.pt":** The processed event data to facilitate See-NeRF training with **event temporal blur prior strategy** of [E<sup>3</sup>NeRF](https://github.com/iCVTEAM/E3NeRF) adopted. Note that we also incorporate the **Photometric Quantity Calibration** in Sec. 4.3 of See-NeRF into the event data preprocessing.

**"frames_weights.npy":** The weights of the virtual sharp frames used during the blur synthesis, generated by adopting **event temporal prior blur strategy** of [E<sup>3</sup>NeRF](https://github.com/iCVTEAM/E3NeRF).

**"poses_bounds.npy":** The pose data generated from the COLMAP output.

**"exp_times_test.npy":** The specific values of *t<sub>0</sub>*, *t<sub>1</sub>*, *t<sub>2</sub>*, *t<sub>3</sub>*, *t<sub>4</sub>*.



## Citation

If you find this useful, please consider citing our papers and starring this repository:

```bibtex
@inproceedings{qi2023e2nerf,
  title={E2NeRF: Event enhanced neural radiance fields from blurry images},
  author={Qi, Yunshan and Zhu, Lin and Zhang, Yu and Li, Jia},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13254--13264},
  year={2023}
}
@inproceedings{qi2024deblurring,
  title={Deblurring neural radiance fields with event-driven bundle adjustment},
  author={Qi, Yunshan and Zhu, Lin and Zhao, Yifan and Bao, Nan and Li, Jia},
  booktitle={Proceedings of the 32nd ACM international conference on multimedia},
  pages={9262--9270},
  year={2024}
}
@article{qi2024e3nerf,
  title={E3NeRF: Efficient Event-Enhanced Neural Radiance Fields from Blurry Images},
  author={Qi, Yunshan and Li, Jia and Zhao, Yifan and Zhang, Yu and Zhu, Lin},
  journal={arXiv preprint arXiv:2408.01840},
  year={2024}
}
@article{qi2026seeing,
  title={Seeing through Light and Darkness: Sensor-Physics Grounded Deblurring HDR NeRF from Single-Exposure Images and Events},
  author={Qi, Yunshan and Zhu, Lin and Bao, Nan and Zhao, Yifan and Li, Jia},
  journal={arXiv preprint arXiv:2601.15475},
  year={2026}
}
```

## Acknowledgment

The overall framework are derived from [E<sup>3</sup>NeRF](https://github.com/iCVTEAM/E3NeRF) and the synthetic data generation is inspired by [HDR-NeRF](https://github.com/xhuangcv/hdr-nerf). We appreciate the effort of the contributors to these repositories.

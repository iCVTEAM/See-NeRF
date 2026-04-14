import os
import numpy as np
import cv2
import shutil
import torch
from utils.poses.pose_utils import gen_poses



def events_info_reader(file_dir):
    with open(file_dir, 'r') as file:
        lines = file.readlines()
        events_num = len(lines)
        exposure_start = int(lines[0].split()[0])
        exposure_end = int(lines[events_num - 1].split()[0])
        file.close()
        return events_num, exposure_start, exposure_end

def load_events_txt(basedir, views_num, bin_num):

    # Pre-definition
    event_map = np.zeros((views_num, bin_num, 260, 346), dtype=np.float64) # The final event_map
    frames_weights = []  # The final weights of frames in each view

    # Processing each view
    for view_num in range(views_num):
        print("------View: {0}------".format(view_num))

        file_dir = basedir + '/{0:03d}.txt'.format((view_num))  # .txt format events file

        # Read basic infomation of events (total event number, start time, and end time of the events in current view)
        events_num, exposure_start, exposure_end = events_info_reader(file_dir)
        print("Events number: {0}; Start at: {1}, End at: {2}".format(events_num, exposure_start, exposure_end))

        # Pre-definition of current view
        event_counter = -1      # The event counter of current view
        frames_time = []        # The splitting time points of event bins
        bin_counter_pre = 0   # The frame counter buffer of current view
        event_t_pre = np.zeros((bin_num, 260, 346), dtype=np.int64)  # offset flag of the first event
        event_t_end = np.zeros((bin_num, 260, 346), dtype=np.int64)  # offset flag of the last flag
        event_t_pre_polar = np.zeros((bin_num, 260, 346))
        event_t_end_polar = np.zeros((bin_num, 260, 346))

        # Record the first splitting time points
        frames_time.insert(0, exposure_start)

        file = open(file_dir, "r")
        # event-by-event processing
        while True:
            line = file.readline()
            if not line:
                break

            # read x, y, p, t of each event
            info = line.split()
            t = float(info[0])
            x = int(info[1])
            y = int(info[2])
            p = int(info[3])
            event_counter = event_counter + 1

            # Update the bin_counter according to event counter (Splitting events evenly by number of events)
            bin_counter = int(event_counter * bin_num / events_num)
            if bin_counter != bin_counter_pre:
                # Record the splitting time points
                frames_time.append(t)
                bin_counter_pre = bin_counter

            if p==1:
                # Record positive event
                event_map[view_num][bin_counter][y][x] += 1
            else:
                # Record negtive event
                event_map[view_num][bin_counter][y][x] -= 1

            # Record the trigger time of the first event in current pixel and current bin
            if event_t_pre[bin_counter][y][x] == 0:
                # Calculate the time difference between the first event and the start time of the bin
                event_t_pre[bin_counter][y][x] = int(t - frames_time[bin_counter] + 1)

            # Record the trigger time of the last event in current pixel and current bin
            event_t_end[bin_counter][y][x] = t - 1


            if event_t_pre_polar[bin_counter][y][x] == 0:
                if p == 1:
                    event_t_pre_polar[bin_counter][y][x] = 1
                else:
                    event_t_pre_polar[bin_counter][y][x] = -1
            if p == 1:
                event_t_end_polar[bin_counter][y][x] = 1
            else:
                event_t_end_polar[bin_counter][y][x] = -1

        # Record the last splitting time points
        frames_time.append(exposure_end)
        print("The splitting time points: ", frames_time)



        # Photometric Quantity Calibration
        for i in range(bin_num):
            # Calculate the time difference between the last event and the end time of the bin
            event_t_end[i][event_t_end[i] != 0] = frames_time[i + 1] - event_t_end[i][event_t_end[i] != 0] + 1

        for i in range(bin_num - 1):

            # Case 1: Two consecutive positive or negative events occurs
            offset = np.zeros((260, 346), dtype=np.float64)
            index = (event_t_pre[i + 1] != 0) & (event_t_end[i] != 0)
            index_pos_old = (event_t_pre[i + 1] != 0) & (event_t_end[i] != 0) & (event_t_pre_polar[i + 1] > 0) & (event_t_end_polar[i] > 0)
            index_neg_old = (event_t_pre[i + 1] != 0) & (event_t_end[i] != 0) & (event_t_pre_polar[i + 1] < 0) & (event_t_end_polar[i] < 0)
            index_pos = (event_t_pre_polar[i + 1] > 0) & (event_t_end_polar[i] > 0)
            index_neg = (event_t_pre_polar[i + 1] < 0) & (event_t_end_polar[i] < 0)

            if not (np.array_equal(index_pos_old, index_pos) and np.array_equal(index_neg_old, index_neg)):
                print("Warning !!!", view_num, i)
                return

            # Calculate the Quantity Calibration
            offset[index] = event_t_end[i][index] / (event_t_end[i][index] + event_t_pre[i + 1][index])
            # Calibration for the pre-bin (i)
            event_map[view_num][i][index_pos] = event_map[view_num][i][index_pos] + offset[index_pos]
            event_map[view_num][i][index_neg] = event_map[view_num][i][index_neg] - offset[index_neg]
            # Calibration for the post-bin (i + 1)
            event_map[view_num][i + 1][index_pos] = event_map[view_num][i + 1][index_pos] - offset[index_pos]
            event_map[view_num][i + 1][index_neg] = event_map[view_num][i + 1][index_neg] + offset[index_neg]

            # Case 2: No following event triggered
            index_pos_sub = (event_t_pre_polar[i + 1] == 0) & (event_t_end_polar[i] > 0)
            index_neg_sub = (event_t_pre_polar[i + 1] == 0) & (event_t_end_polar[i] < 0)
            # Calibration for the pre-bin (i)
            # Assuming the light intensity variation follows a uniform distribution (0, 1),
            # then setting the offset to 0.5 is the optimal solution.
            event_map[view_num][i][index_pos_sub] = event_map[view_num][i][index_pos_sub] + 0.5
            event_map[view_num][i][index_neg_sub] = event_map[view_num][i][index_neg_sub] - 0.5

        # Calibration for the last-bin in Case 2
        i = i + 1
        index_pos_sub = (event_t_end_polar[i] > 0)
        index_neg_sub = (event_t_end_polar[i] < 0)
        event_map[view_num][i][index_pos_sub] = event_map[view_num][i][index_pos_sub] + 0.5
        event_map[view_num][i][index_neg_sub] = event_map[view_num][i][index_neg_sub] - 0.5



        # Calculate the weight of each frame
        frames_time.insert(0, exposure_start)   # add extra start time for calculation
        frames_time.append(exposure_end)        # add extra end time for calculation
        frames_weight = []
        for i in range(bin_num + 1):
            frames_weight.append((frames_time[i + 2] - frames_time[i]) / 2 / (exposure_end - exposure_start))
        print("The weight of each frame: ", frames_weight)
        frames_weights.append(frames_weight)

    return event_map, frames_weights



scenes = ["lab", "lobby", "shelf", "table"]
views_num = 16  # number of training views
bin_num = 6  # value of b



# Generate events_offset.pt and frames_weights.npy for training
print("------Stage 1: events_offset.pt and  frames_weights.npy Generation------")

for scene in scenes:
    print("------Processing {0} Scene------".format(scene))

    event_map, frames_weights = load_events_txt("./{0}/events/".format(scene), views_num, bin_num)
    event_map = torch.tensor(event_map).view(-1, bin_num, 260 * 346)
    torch.save(event_map, "./{0}/events_offset.pt".format(scene))
    np.savetxt("./{0}/frames_weights.npy".format(scene), frames_weights)



# Generate Images_Pose_Estimation and Image Folders for training
print("------Stage 2: Images_Pose_Estimation, Image Folders Generation------")

for scene in scenes:
    print("------Processing {0} Scene------".format(scene))

    events = torch.load("./{0}/events_offset.pt".format(scene)).view(views_num, bin_num, 260, 346)
    os.makedirs("./{0}/images_pose_estimation".format(scene), exist_ok=True)

    for i in range(views_num):
        print("View: {0}".format(i))

        blurry_image = torch.tensor(cv2.imread("./{0}/images/{1:03d}.jpg".format(scene, i)), dtype=torch.float)

        # EDI Model
        event_sum = torch.zeros(260, 346)
        EDI = torch.ones(260, 346)
        for j in range(bin_num):
            event_sum = event_sum + events[i][j]
            EDI = EDI + torch.exp(0.3 * event_sum)

        EDI = torch.stack([EDI, EDI, EDI], axis=-1)
        sharp_image = (bin_num + 1) * blurry_image / EDI
        sharp_image = torch.clamp(sharp_image, max=255)

        # Save the pre deblurred images for training pose estimation
        cv2.imwrite("./{0}/images_pose_estimation/{1:03d}.jpg".format(scene, i * (bin_num + 1)), sharp_image.numpy())

        offset = torch.zeros(260, 346)
        for j in range(bin_num):
            offset = offset + events[i][j]
            img = sharp_image * torch.exp(0.3 * torch.stack([offset, offset, offset], axis=-1))
            cv2.imwrite("./{0}/images_pose_estimation/{1:03d}.jpg".format(scene, i * (bin_num + 1) + 1 + j), img.numpy())

    # Move the testing image for testing pose estimation
    for j in range(28):
        shutil.copy("./{0}/gt/{1:03d}.jpg".format(scene, j),
                    "./{0}/images_pose_estimation/{1:03d}.jpg".format(scene, views_num * (bin_num + 1) + j))



# Generate pose_bounds.npy for training and testing
print("------Stage 4: Colmap Calling and pose_bounds.npy Generation------")

for scene in scenes:
    print("------Processing {0} Scene------".format(scene))

    # Colmap Calling
    # We recommand use the GUI Colmap manually with shared_intrinsics parameter which is not support in the command line
    current_path = os.getcwd().replace('\\', '/')   # for windows command line
    images_pose_path = "{0}/{1}/images_pose_estimation/".format(current_path, scene)
    command = "colmap automatic_reconstructor --workspace_path {0} --image_path {0} --data_type individual " \
              "--sparse on --dense off --num_threads 8 --gpu_index 0".format(images_pose_path)
    os.system(command)

    # Generate pose_bounds.npy file for training
    gen_poses("./{0}/images_pose_estimation/".format(scene), "exhaustive_matcher")
    shutil.move("./{0}/images_pose_estimation/poses_bounds.npy".format(scene), "./{0}/poses_bounds.npy".format(scene))

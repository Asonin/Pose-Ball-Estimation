# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import numpy as np
import torchvision
import cv2
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from pose_estimator.utils.cameras import project_pose_cpu
from pose_estimator.utils.transforms import affine_transform

# coco17
LIMBS17 = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13], [13, 15],
        [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]

# shelf / campus
LIMBS14 = [[0, 1], [1, 2], [3, 4], [4, 5], [2, 3], [6, 7], [7, 8], [9, 10],
          [10, 11], [2, 8], [3, 9], [8, 12], [9, 12], [12, 13]]

# panoptic
LIMBS15 = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10],
         [10, 11], [2, 6], [2, 12], [6, 7], [7, 8], [12, 13], [13, 14]]

# colors for visualization
colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']


def save_heatmaps(batch_images, batch_heatmaps, prefix):
    basename = os.path.basename(prefix)
    dirname = os.path.join(os.path.dirname(prefix), 'heatmaps')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    prefix = os.path.join(dirname, basename)

    batch_size, num_views, _, image_height, image_width = batch_images.shape
    num_joints = batch_heatmaps.shape[2]

    for c in range(num_views):
        file_name = prefix + '_view_{}.jpg'.format(c + 1)

        batch_image = batch_images[:, c].clone()
        min = float(batch_image.min())
        max = float(batch_image.max())
        batch_image.add_(-min).div_(max - min + 1e-5)
    
        batch_image = batch_image.flip(1)
        grid_image = np.zeros((batch_size * image_height, (num_joints + 1) * image_width, 3), dtype=np.uint8)

        for i in range(batch_size):
            image = batch_image[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            heatmaps = batch_heatmaps[i, c].mul(255).clamp(0, 255).byte().cpu().numpy()
            height_begin = image_height * i
            height_end = image_height * (i + 1)
            for j in range(num_joints):
                resized_heatmap = cv2.resize(heatmaps[j], (int(image_width), int(image_height)))
                colored_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
                masked_image = colored_heatmap * 0.7 + image * 0.3

                width_begin = image_width * (j + 1)
                width_end = image_width * (j + 2)
                grid_image[height_begin:height_end, width_begin:width_end, :] = masked_image

            grid_image[height_begin:height_end, :image_width, :] = image
        cv2.imwrite(file_name, grid_image)


def save_2d_planes(config, meta, final_poses, poses, proposal_centers, prefix):
    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, '2d_planes')
    individual_space_size = config.INDIVIDUAL_SPEC.SPACE_SIZE

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    prefix = os.path.join(dirname1, basename)
    file_name = "{}.png".format(prefix)

    batch_size = final_poses.shape[0]
    xplot = 4 
    yplot = batch_size

    width = 4.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.2, hspace=0.15)
    
    mask = (proposal_centers[:, :, 3] >= 0).detach().cpu().numpy()
    
    for i in range(batch_size):
        ax = plt.subplot(yplot, xplot, 4*i+1, projection='3d')
        curr_poses = [poses[k][i, mask[i]] for k in range(3)]

        if 'joints_3d' in meta:
            num_person = meta['num_person'][i]
            joints_3d = meta['joints_3d'][i]
            joints_3d_vis = meta['joints_3d_vis'][i]

            # 3d projection
            for n in range(num_person):
                joint = joints_3d[n]
                joint_vis = joints_3d_vis[n]
                for k in eval("LIMBS{}".format(len(joint))):
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    if joint_vis[k[0]] > 0.1 and joint_vis[k[1]] > 0.1:
                        ax.plot(x, y, z, c='r', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)
                    else:
                        ax.plot(x, y, z, c='r', ls='--', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)

        final_pose = final_poses[i, mask[i]]
        for n, joint in enumerate(final_pose):
            for k in eval("LIMBS{}".format(len(joint))):
                x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                ax.plot(x, y, z, c=colors[int(n % 10)], lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                        markeredgewidth=1)
        if i == 0:
            plt.title('3d pose',fontdict={'weight':'normal','size': 15})
        
        # xy_projection
        ax = plt.subplot(yplot, xplot, 4*i+2)
        
        if 'joints_3d' in meta:
            for n in range(num_person):
                joint = joints_3d[n]
                joint_vis = joints_3d_vis[n]
                for k in eval("LIMBS{}".format(len(joint))):
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    if joint_vis[k[0]] > 0.1 and joint_vis[k[1]] > 0.1:
                        ax.plot(x, y, c='r', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)
                    else:
                        ax.plot(x, y, c='r', ls='--', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)
        
        # bbox visualization
        for j in range(len(proposal_centers[i])):
            if proposal_centers[i, j, 3] < 0:
                continue
            top_left_x = proposal_centers[i, j, 0] - proposal_centers[i, j, 5] * individual_space_size[0] / 2
            top_left_y = proposal_centers[i, j, 1] - proposal_centers[i, j, 6] * individual_space_size[1] / 2
            width = proposal_centers[i, j, 5] * individual_space_size[0]
            height = proposal_centers[i, j, 6] * individual_space_size[1]
            rect = plt.Rectangle((top_left_x, top_left_y), width, height, fill=False, edgecolor = 'red',linewidth=1)
            ax.add_patch(rect)

        for n, joint in enumerate(curr_poses[0]):
            for k in eval("LIMBS{}".format(len(joint))):
                x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                ax.plot(x, y, c=colors[int(n % 10)], lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                        markeredgewidth=1)
        if i == 0:
            plt.title('xy projection',fontdict={'weight':'normal','size': 15})

        # xz_projection
        ax = plt.subplot(yplot, xplot, 4*i+3)

        if 'joints_3d' in meta:
            for n in range(num_person):
                joint = joints_3d[n]
                joint_vis = joints_3d_vis[n]
                for k in eval("LIMBS{}".format(len(joint))):
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    if joint_vis[k[0]] > 0.1 and joint_vis[k[1]] > 0.1:
                        ax.plot(x, z, c='r', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)
                    else:
                        ax.plot(x, z, c='r', ls='--', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)

        for j in range(len(proposal_centers[i])):
            if proposal_centers[i, j, 3] < 0:
                continue
            top_left_x = proposal_centers[i, j, 0] - proposal_centers[i, j, 5] * individual_space_size[0] / 2
            top_left_y = proposal_centers[i, j, 2] - individual_space_size[2] / 2
            width = proposal_centers[i, j, 5] * individual_space_size[0]
            height = individual_space_size[2]
            rect = plt.Rectangle((top_left_x, top_left_y), width, height, fill=False, edgecolor = 'red',linewidth=1)
            ax.add_patch(rect)
        
        for n, joint in enumerate(curr_poses[1]):
            for k in eval("LIMBS{}".format(len(joint))):
                x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                z = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                ax.plot(x, z, c=colors[int(n % 10)], lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                        markeredgewidth=1)
        if i == 0:
            plt.title('xz projection',fontdict={'weight':'normal','size': 15})
        
        # yz_projection
        ax = plt.subplot(yplot, xplot, 4*i+4)
        if 'joints_3d' in meta:
            for n in range(num_person):
                joint = joints_3d[n]
                joint_vis = joints_3d_vis[n]
                for k in eval("LIMBS{}".format(len(joint))):
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    if joint_vis[k[0]] > 0.1 and joint_vis[k[1]] > 0.1:
                        ax.plot(y, z, c='r', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)
                    else:
                        ax.plot(y, z, c='r', ls='--', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)

        for j in range(len(proposal_centers[i])):
            if proposal_centers[i, j, 3] < 0:
                continue
            top_left_x = proposal_centers[i, j, 1] - proposal_centers[i, j, 6] * individual_space_size[1] / 2
            top_left_y = proposal_centers[i, j, 2] - individual_space_size[2] / 2
            width = proposal_centers[i, j, 6] * individual_space_size[1]
            height = individual_space_size[2]
            rect = plt.Rectangle((top_left_x, top_left_y), width, height, fill=False, edgecolor = 'red',linewidth=1)
            ax.add_patch(rect)

        for n, joint in enumerate(curr_poses[2]):
            for k in eval("LIMBS{}".format(len(joint))):
                y = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                z = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                ax.plot(y, z, c=colors[int(n % 10)], lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                    markeredgewidth=1)
        if i == 0:
            plt.title('yz projection',fontdict={'weight':'normal','size': 15})
        
    plt.savefig(file_name)
    plt.close(0)

        
def save_image_with_projected_ball_and_poses(config, images, ball_pos, poses, meta, prefix, cameras, transform, colorid):
    basename = os.path.basename(prefix)
    dirname = os.path.join(os.path.dirname(prefix), 'image_with_projected_ball_and_poses')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    prefix = os.path.join(dirname, basename)

    batch_size, num_views, _, height, width = images.shape
    max_people = poses.shape[1]
    if max_people == 0:
        num_joints = 0
    else:
        num_joints = poses.shape[2]
    
    for c in range(num_views):
        file_name = prefix + '_view_{}.jpg'.format(c + 1)
        batch_image = images[:, c].flip(1)
        grid = torchvision.utils.make_grid(batch_image, 1, padding=0, normalize=True)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy().copy()
        if num_joints != 0:
            limbs = eval("LIMBS{}".format(num_joints))

        for i in range(batch_size):
            curr_seq = meta['seq'][i]
            camera = cameras[curr_seq][c]
            
            # project the ball
            color = np.flip(np.array(matplotlib.colors.to_rgb('r'))) * 255
            if ball_pos is not None:
                ball_2d = project_pose_cpu(ball_pos.reshape(1, 3), camera)
                ball_2d = affine_transform(ball_2d[0], transform)
                if is_valid_coord(ball_2d, width, height):
                    xc = ball_2d[0]
                    yc = i * height + ball_2d[1]
                    cv2.circle(ndarr, (int(xc), int(yc)), 10, color, -1)
            
            # project the pose
            for n in range(max_people):
                # if poses[i, n, 0, 4] < config.CAPTURE_SPEC.MIN_SCORE:
                #     continue

                color = np.flip(np.array(matplotlib.colors.to_rgb(colors[int(colorid[n] % 10)]))) * 255
                pose_2d = project_pose_cpu(poses[i, n, :, :3], cameras[curr_seq][c])
                
                for j in range(num_joints):
                    pose_2d[j] = affine_transform(pose_2d[j], transform)
                    if is_valid_coord(pose_2d[j], width, height):
                        xc = pose_2d[j][0]
                        yc = i * height + pose_2d[j][1]
                        cv2.circle(ndarr, (int(xc), int(yc)), 4, color, -1)
                    
                for limb in limbs:
                    parent = pose_2d[limb[0]]
                    if not is_valid_coord(parent, width, height):
                        continue
                    child = pose_2d[limb[1]]
                    if not is_valid_coord(child, width, height):
                        continue

                    px = parent[0]
                    py = i * height + parent[1]
                    cx = child[0]
                    cy = i * height + child[1]
                    cv2.line(ndarr, (int(px), int(py)), (int(cx), int(cy)), color, 2)
                    
        cv2.imwrite(file_name, ndarr)


def save_3d_images(config, ball_pos, poses, prefix, colorid):
    basename = os.path.basename(prefix) # file name 
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, '3d_vis')

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    prefix = os.path.join(dirname1, basename)
    file_name = prefix + "_3d.jpg"

    batch_size = 1
    xplot = 1 
    yplot = batch_size

    width = 25.0 * xplot
    height = 25.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.2, hspace=0.15)

    space_size = config.CAPTURE_SPEC.SPACE_SIZE
    space_center = config.CAPTURE_SPEC.SPACE_CENTER
    
    for i in range(batch_size):
        ax = plt.subplot(yplot, xplot, 4*i+1, projection='3d')
        ax.set_xlim3d(space_center[0]-space_size[0]/2, space_center[0]+space_size[0]/2)
        ax.set_ylim3d(space_center[1]-space_size[1]/2, space_center[1]+space_size[1]/2)
        ax.set_zlim3d(space_center[2]-space_size[2]/2, space_center[2]+space_size[2]/2)
        ax.set_box_aspect((space_size[0], space_size[1], space_size[2]))
        ax.set_xticks(np.linspace(space_center[0]-space_size[0]/2, space_center[0]+space_size[0]/2, 17))
        ax.set_yticks(np.linspace(space_center[1]-space_size[1]/2, space_center[1]+space_size[1]/2, 17))
        ax.set_zticks(np.linspace(space_center[2]-space_size[2]/2, space_center[2]+space_size[2]/2, 3))

        # plot the ball
        if ball_pos is not None:
            N = 100
            stride = 2
            radius = 120
            u = np.linspace(0, 2 * np.pi, N)
            v = np.linspace(0, np.pi, N)
            x = np.outer(np.cos(u), np.sin(v)) * radius + ball_pos[0]
            y = np.outer(np.sin(u), np.sin(v)) * radius + ball_pos[1]
            z = np.outer(np.ones(np.size(u)), np.cos(v)) * radius + ball_pos[2]
            ax.plot_surface(x, y, z, linewidth=0.0, cstride=stride, rstride=stride, color='r')

        # plot the poses
        final_pose = poses
        for n, joint in enumerate(final_pose):
            # if joint[0, 4] < config.CAPTURE_SPEC.MIN_SCORE:
            #     continue
            for k in eval("LIMBS{}".format(len(joint))):
                x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                ax.plot(x, y, z, c=colors[int(colorid[n] % 10)], lw=4, marker='o', markerfacecolor='w', markersize=5,
                        markeredgewidth=2)
        if i == 0:
            plt.title('3d pose',fontdict={'weight':'normal','size': 15})

    plt.savefig(file_name)
    plt.close(0)


def is_valid_coord(joint, width, height):
    valid_x = joint[0] >= 0 and joint[0] < width
    valid_y = joint[1] >= 0 and joint[1] < height
    return valid_x and valid_y
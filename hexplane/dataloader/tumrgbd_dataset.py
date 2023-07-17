from torch.utils.data import Dataset
from torchvision import transforms as T

import numpy as np
import torch
import os
import tqdm
import cv2
from datetime import datetime

from .ray_utils import get_ray_directions, get_rays, ndc_rays


def trans_t(t):
    return torch.Tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
    ).float()


def rot_phi(phi):
    return torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def rot_theta(th):
    return torch.Tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
            torch.Tensor(
                np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            )
            @ c2w
    )
    return c2w

class TUMRgbdDataset(Dataset):
    def __init__(
            self,
            datadir,
            split="train",
            downsample=1.0,
            is_stack=False,
            cal_fine_bbox=False,
            N_vis=-1,
            time_scale=1.0,
            scene_bbox_min=[-1.0, -1.0, -1.0],
            scene_bbox_max=[1.0, 1.0, 1.0],
            N_random_pose=1000,
    ):
        self.root_dir = datadir
        self.split = split
        self.downsample = downsample
        self.img_wh = (int(640 / downsample), int(480 / downsample))  # 640x480
        self.is_stack = is_stack
        self.N_vis = N_vis  # evaluate images for every N_vis images

        self.time_scale = time_scale
        self.world_bound_scale = 1.1

        self.near = 2.0
        self.frame_rate = 32
        self.far = 6.0
        self.near_far = [2.0, 6.0]

        self.transform = T.ToTensor()

        fx = 535.4 / downsample
        fy = 539.2 / downsample
        self.focal = [fx, fy]
        cx = 320.1
        cy = 247.6
        self.opt_centers = [cx, cy]

        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        self.read_meta()

        if cal_fine_bbox:
            xyz_min, xyz_max = self.compute_bbox()
            self.scene_bbox = torch.stack((xyz_min, xyz_max), dim=0)

        self.define_proj_mat()

        self.white_bg = True
        self.ndc_ray = False
        self.depth_data = False

        self.N_random_pose = N_random_pose
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        # Generate N_random_pose random poses, which we could render depths from these poses and apply depth smooth loss to the rendered depth.
        if split == "train":
            self.init_random_pose()

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "time": self.all_times[idx],
            }
        else:  # create data for each image separately
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            time = self.all_times[idx]
            sample = {"rays": rays, "rgbs": img, "time": time}
        return sample


    def init_random_pose(self):
        # Randomly sample N_random_pose radius, phi, theta and times.
        radius = np.random.randn(self.N_random_pose) * 0.1 + 4
        phi = np.random.rand(self.N_random_pose) * 360 - 180
        theta = np.random.rand(self.N_random_pose) * 360 - 180
        random_times = self.time_scale * (torch.rand(self.N_random_pose) * 2.0 - 1.0)
        self.random_times = random_times

        # Generate rays from random radius, phi, theta and times.
        self.random_rays = []
        for i in range(self.N_random_pose):
            random_poses = pose_spherical(theta[i], phi[i], radius[i])
            rays_o, rays_d = get_rays(self.directions, random_poses)
            self.random_rays += [torch.cat([rays_o, rays_d], 1)]

        self.random_rays = torch.stack(self.random_rays, 0).reshape(-1, *self.img_wh[::-1], 6)

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    def compute_bbox(self):
        print("compute_bbox_by_cam_frustrm: start")
        xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
        xyz_max = -xyz_min
        rays_o = self.all_rays[:, 0:3]
        viewdirs = self.all_rays[:, 3:6]
        pts_nf = torch.stack(
            [rays_o + viewdirs * self.near, rays_o + viewdirs * self.far]
        )
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1)))
        print("compute_bbox_by_cam_frustrm: xyz_min", xyz_min)
        print("compute_bbox_by_cam_frustrm: xyz_max", xyz_max)
        print("compute_bbox_by_cam_frustrm: finish")
        xyz_shift = (xyz_max - xyz_min) * (self.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
        return xyz_min, xyz_max

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))
            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def parse_dataset(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths = [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose @ c2w
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return images, poses, tstamp_image

    def read_meta(self):
        w, h = self.img_wh
        self.directions = get_ray_directions(h, w, self.focal)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal[0], 0, self.opt_centers[0]],
                                        [0, self.focal[1], self.opt_centers[1]],
                                        [0, 0, 1]]).float()

        self.color_paths, self.poses, self.t_stamps = self.parse_dataset(self.root_dir, frame_rate=self.frame_rate)
        """
        self.colorpaths: List[str] -> contains rgb image paths
        self.poses: List[torch.Tensor] with shape [4,4] -> pose values for the corresponding timestamp
        self.t_stamps: List[TimeStamps] -> List of timestamps for the corresponding frame
        """
        self.all_rgbs = []
        self.all_rays = []
        self.all_times = []
        for i in range(len(self.poses)):
            c2w = self.poses[i]
            colar_path = self.color_paths[i]
            cur_color_image = cv2.imread(colar_path)
            cur_color_image = cv2.cvtColor(cur_color_image, cv2.COLOR_BGR2RGB)
            cur_color_image = cur_color_image / 255.

            if self.downsample != 1.0:
                cur_color_image = cv2.resize(cur_color_image, self.img_wh, interpolation=cv2.INTER_AREA)

            # cur_color_image = self.transform(cur_color_image)
            cur_color_image = torch.from_numpy(cur_color_image)
            cur_color_image = cur_color_image.view(h * w, -1)
            self.all_rgbs += [cur_color_image]

            rays_o, rays_d = get_rays(self.directions, c2w)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]
            cur_time_stmp = float(i) / (len(self.t_stamps) - 1)
            cur_time = torch.tensor(cur_time_stmp).expand(rays_o.shape[0], 1)
            self.all_times += [cur_time]

        self.poses = torch.stack(self.poses)

        if not self.is_stack:
            self.all_rays = torch.cat(
                self.all_rays, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(
                self.all_rgbs, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_times = torch.cat(self.all_times, 0)

        else:
            self.all_rays = torch.stack(
                self.all_rays, 0
            )  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(
                -1, *self.img_wh[::-1], 3
            )  # (len(self.meta['frames]),h,w,3)
            self.all_times = torch.stack(self.all_times, 0)

        self.all_times = self.time_scale * (self.all_times * 2.0 - 1.0)

    def get_val_pose(self):
        render_poses = self.val_poses
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times

    def get_val_rays(self):
        val_poses, val_times = self.get_val_pose()  # get valitdation poses and times
        rays_all = []  # initialize list to store [rays_o, rays_d]

        for i in range(val_poses.shape[0]):
            c2w = torch.FloatTensor(val_poses[i])
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            if self.ndc_ray:
                W, H = self.img_wh
                rays_o, rays_d = ndc_rays(
                    H, W, self.focal, 1.0, rays_o, rays_d
                )
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            rays_all.append(rays)
        return rays_all, torch.FloatTensor(val_times)
from torch.utils.data import Dataset
from torchvision import transforms as T

import numpy as np
import torch
import os
import gc
import tqdm
import cv2
from datetime import datetime
from PIL import Image
from .ray_utils import get_ray_directions_blender, get_rays, ndc_rays_blender_tum



def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
    )
    return np.stack(render_poses)


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
    :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


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


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


class TUMRgbdSlamDataset(Dataset):
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
            bd_factor=0.75,
            eval_step=1,
            eval_index=0,
            sphere_scale=1.0,
            images=[],
            poses=[],
            depths=[],
            timestamps=[],
            intrinsics=[]
    ):
        # self.img_wh = (int(640 / downsample), int(480 / downsample))  # 640x480
        # self.root_dir = datadir
        self.split = split
        self.downsample = downsample

        self.is_stack = is_stack
        self.N_vis = N_vis  # evaluate images for every N_vis images

        self.time_scale = time_scale
        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])
        self.world_bound_scale = 1.1
        self.bd_factor = bd_factor
        self.eval_step = eval_step
        self.eval_index = eval_index
        self.blender2opencv = np.eye(4)

        self.poses = None
        self.depths = None
        self.images = None
        self.image_stride = None
        self.time_number = None
        self.all_rgbs = None
        self.all_times = None
        self.all_rays = None
        self.global_mean_rgb = None

        # TODO: CHECK NEAR AND FAR VALUES
        self.near = 0.0
        self.far = 9.88 # TODO: THIS IS PROBABLY 9.9, TRY TO FIND IT
        # self.far = 80
        self.near_far = np.array([self.near, self.far])  # NDC near far is [0, 1.0]
        self.frame_rate = 32
        self.white_bg = False
        self.ndc_ray = True
        self.transform = T.ToTensor()
        self.png_depth_scale = 6553.5
        self.depth_data = False

        self.transform = T.ToTensor()

        self.intrinsics = intrinsics
        self.focal = [intrinsics[0, 0], intrinsics[1, 1]]
        self.opt_centers = [intrinsics[0, 2], intrinsics[1, 2]]

        self.images = images
        self.poses = poses
        self.depths = depths

        self.timestamps = timestamps

        self.load_meta()
        print("____ Meta data is successfully loaded ! ____")

        # TODO: AFTER MAKE SURE ABOUT NEAR AND FAR VALUES, TRY TO CALL THIS FUNCTION
        if cal_fine_bbox:
            xyz_min, xyz_max = self.compute_bbox()
            self.scene_bbox = torch.stack((xyz_min, xyz_max), dim=0)

        # self.define_proj_mat()

        self.ndc_ray = False
        self.depth_data = True
        self.depth_data = False

        self.N_random_pose = N_random_pose
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        # Generate N_random_pose random poses, which we could render depths
        # from these poses and apply depth smooth loss to the rendered depth.
        if split == "train":
            self.init_random_pose()

    def _get_data_packet(self, k0, k1=None):
        if k1 is None:
            k1 = k0 + 1
        else:
            assert (k1 >= k0)

        images = []
        for k in np.arange(k0, k1):
            img_path = self.images[k]
            
            image = Image.open(img_path)
            
            if self.downsample != 1.0:
                image = image.resize(self.img_wh, Image.LANCZOS)
                
            image = self.transform(image)
            #image_opencv = cv2.imread(img_path)
            image = image.view(3, -1).permute(1, 0)  # (h*w, 3) RGBA
            # depth = cv2.imread(self.depths[k], cv2.IMREAD_UNCHANGED)
            # depth = depth[:, :, np.newaxis].astype(np.int32) # Pytorch cannot handle uint16
            images += [image]
            # depths  += [depth]
            del image
            # del depth

        return images

    
    
    def get_depth_data_packet(self, k0, k1=None):
        if k1 is None:
            k1 = k0 + 1
        else:
            assert (k1 >= k0)
        depths = []
        for k in np.arange(k0, k1):
            depth_path = self.depths[k]
            if '.png' in depth_path:
                depth = Image.open(depth_path)
            elif '.exr' in depth_path:
                depth= readEXR_onlydepth(depth_path) 
                
            if self.downsample != 1.0:
                depth = depth.resize(self.img_wh, Image.LANCZOS)
                
            depth = self.transform(depth)
            depth = depth / self.png_depth_scale   
            
            depth = depth.view(1, -1).permute(1, 0)
            
            depths += [depth]    
            del depth
        return depths   

    def __len__(self):
        #if self.split == "train" and self.is_stack is True:
        #    return self.time_number
        #else:
        #    return len(self.all_rgbs)
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        if self.split == "train":
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "depths": self.all_depths[idx],
                "time": self.all_times[idx]
            }
        else:
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            depths =  self.all_depths[idx]
            time = self.all_times[idx]
            sample = {"rays": rays, "rgbs": img, "depths": depths, "time": time}
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
            [rays_o + viewdirs * self.near, rays_o + viewdirs * self.far])

        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1)))
        print("compute_bbox_by_cam_frustrm: xyz_min", xyz_min)
        print("compute_bbox_by_cam_frustrm: xyz_max", xyz_max)
        print("compute_bbox_by_cam_frustrm: finish")
        xyz_shift = (xyz_max - xyz_min) * (self.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
        return xyz_min, xyz_max

    def load_meta(self):
        # read poses and video file paths
        assert len(self.images) == self.poses.shape[0]
        assert len(self.images) == len(self.timestamps)

        H, W = cv2.imread(self.images[0]).shape[0:2]
        self.img_wh = (W, H)

        # recenter poses
        self.poses, pose_avg = center_poses(self.poses, self.blender2opencv)

        #  # Sample N_views poses for validation - NeRF-like camera trajectory.
        N_views = 120
        self.val_poses = get_spiral(self.poses, self.near_far, N_views=N_views)

        # TODO: try to use only get_ray_directions in here
        self.directions = get_ray_directions_blender(H, W, self.focal)

        """
        self.colorpaths: List[str] -> contains rgb image paths
        self.poses: List[torch.Tensor] with shape [4,4] -> pose values for the corresponding timestamp
        self.t_stamps: List[TimeStamps] -> List of timestamps for the corresponding frame
        """
        all_rays = []
        all_times = []
        frame_count = len(self.timestamps)
        for idx in range(0, len(self.timestamps)):
            rays_o, rays_d = get_rays(self.directions,
                                      torch.FloatTensor(self.poses[idx]))  # both (h*w, 3)
            rays_o, rays_d = ndc_rays_blender_tum(H, W, self.focal, 1.0, rays_o, rays_d)
            all_rays += [torch.cat([rays_o, rays_d], 1)]
            cur_time = torch.tensor(float(idx) / (frame_count - 1)).expand(rays_o.shape[0], 1)
            all_times += [cur_time]
            print(f"timestamp {idx} is loaded")
            gc.collect()

        all_rgbs = self._get_data_packet(0, len(self.images))
        all_depths = self.get_depth_data_packet(0, len(self.depths))
        N_cam, N_rays, C = torch.stack(all_rgbs).shape

        # all_times = torch.from_numpy(np.array(t_stamps))

        if not self.is_stack:
            print("Catting ...")
            all_rays = torch.cat(all_rays, 0)
            all_rgbs = torch.cat(all_rgbs, 0)
            all_depths = torch.cat(all_depths, 0)
            all_times = torch.cat(all_times, 0)
            print("Catting performed !!")
        else:
            print("Stacking ...")
            all_rays = torch.stack(all_rays, 0)
            all_rgbs = torch.cat(all_rgbs, 0).reshape(-1, *self.img_wh[::-1], 3)
            all_depths = torch.stack(all_depths, 0).reshape(-1, *self.img_wh[::-1], 1)
            all_times = torch.stack(all_times, 0)
            print("Stack performed !!")
        # apply time scale
        all_times = self.time_scale * (all_times * 2.0 - 1.0)

        self.image_stride = N_rays
        self.time_number = N_cam
        self.all_rgbs = all_rgbs
        self.all_times = all_times
        self.all_depths = all_depths
        # self.all_rays = all_rays.reshape(N_cam, N_rays, 6)
        self.all_rays = all_rays
        self.global_mean_rgb = torch.mean(all_rgbs, dim=1)
        print('____LOADING META DATA REPORT______')
        print(f"Image H: {self.img_wh[1]}, W: {self.img_wh[0]}, HXW={self.img_wh[0]*self.img_wh[1]}")
        print(f"'All rgbs': type->{type(self.all_rgbs)}, shape->{self.all_rgbs.shape}, datatype->{self.all_rgbs.dtype}")
        print(f"'All rays': type->{type(self.all_rays)}, shape->{self.all_rays.shape}, datatype->{self.all_rays.dtype}")
        print(f"'All depths': type->{type(self.all_depths)}, shape->{self.all_depths.shape}, datatype->{self.all_depths.dtype}")
        print(f"'All times': type->{type(self.all_times)}, shape->{self.all_times.shape}, datatype->{self.all_times.dtype}")


    def get_val_pose(self):
        render_poses = self.val_poses
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times

    def get_val_rays(self):
        val_poses, val_times = self.get_val_pose()  # get validation poses and times
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
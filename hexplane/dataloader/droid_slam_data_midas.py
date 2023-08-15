import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
from scipy.spatial.transform import Rotation as R
from evo.core import lie_algebra as lie

from .ray_utils import *

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
    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.Tensor(
            np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        )
        @ c2w
        @ blender2opencv
    )
    return c2w


class YourOwnDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):

        self.N_vis = -1
        self.root_dir = "/media/nico/TOSHIBA EXT/rgbd_bonn_dataset/rgbd_bonn_moving_nonobstructing_box"
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()

        self.ndc_ray = False
        self.depth_data = True

        self.scene_bbox = torch.tensor([[-1.0, -1, -1], [1, 1, 1]])
        self.blender2opencv = torch.Tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # self.blender2opencv = np.eye(4)
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = False
        self.near_far = [0.0, 1.0]
        self.near = self.near_far[0]
        self.far = self.near_far[1]
        self.world_bound_scale = 1.1

        cal_fine_bbox = True
        if cal_fine_bbox and split=="train":
            xyz_min, xyz_max = self.compute_bbox()
            self.scene_bbox = torch.stack((xyz_min, xyz_max), dim=0)
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):

        frame_rate = 32

        trajectories_droid = np.load('trajectories.npz')
        traj_droid = trajectories_droid['arr_1']

        rot = R.from_quat(traj_droid[:, 3:])
        rot = rot.as_matrix()
        poses = np.eye(4, 4)[None, :].repeat(rot.shape[0], axis=0)
        poses[:, :3, :3] = rot
        poses[:, :3, 3] = traj_droid[:, :3]

        t = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        poses = [np.dot(t, p) for p in poses]

        t = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        poses = [np.dot(t, p) for p in poses]
        
        w, h = int(640/self.downsample), int(480/self.downsample)
        self.img_wh = [w,h]

        self.focal_x = 542.822841
        self.focal_y = 542.576870
        self.cx = 315.593520
        self.cy = 237.756098

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal_x,self.focal_y], center=[self.cx, self.cy])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal_x,0,self.cx],[0,self.focal_y,self.cy],[0,0,1]]).float()

        image_list = os.path.join(self.root_dir, 'rgb.txt')

        # parse data
        image_data = self.parse_list(image_list)
        
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depths = []
        self.all_times = []
        timestamps = []

        # for ix in tqdm(indicies, desc=f'Loading data {self.split} ({len(indicies)})'):#img_list:#
        img_eval_interval = 1
        idxs = list(range(0, len(image_data), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#

            c2w = torch.FloatTensor(poses[i])
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, image_data[i, 1])
            self.image_paths += [image_path]
            img = Image.open(image_path)
            
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(-1, w*h).permute(1, 0)  # (h*w, 4) RGBA
            if img.shape[-1]==4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]

            image_name = image_path.split('/')[-1].replace('.png', '')

            # load depth
            disp_path = os.path.join(
                    self.root_dir, "rgb/dpt", str(image_name).zfill(3) + ".npy"
                )
            disp_data = np.load(disp_path)
            disp_resized = cv2.resize(disp_data, (w, h), interpolation=cv2.INTER_LINEAR)
            disp = torch.from_numpy(disp_resized).view(-1)
            self.all_depths += [1 / disp]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            if self.ndc_ray:
                rays_o, rays_d = ndc_rays_blender_tum(h, w, [self.focal_x, self.focal_y], 1.0, rays_o, rays_d)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

            timestamps.append(float(image_name))

        timestamps = np.array(timestamps)
        timestamps -= timestamps[0]
        timestamps /= timestamps[-1]

        for i in range(len(timestamps)):
            timestamp = torch.tensor(timestamps[i], dtype=torch.float32).expand(rays_o.shape[0], 1)
            self.all_times.append(timestamp)

        self.poses = torch.stack(self.poses)[200:700]
        # self.poses, avg_pose = center_poses(self.poses[:, :3, :], self.blender2opencv)
        # self.poses = torch.from_numpy(self.poses)

        # # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # # See https://github.com/bmild/nerf/issues/34
        # self.near_fars = torch.Tensor([0.0, 10.0])
        # near_original = self.near_fars.min()
        # scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # # the nearest depth is at 1/0.75=1.33
        # self.near_fars /= scale_factor
        # self.poses[..., 3] /= scale_factor

        self.all_rays = torch.stack(self.all_rays, 0)[200:700]
        self.all_rgbs = torch.stack(self.all_rgbs, 0)[200:700]
        self.all_depths = torch.stack(self.all_depths, 0)[200:700]
        # self.all_depths = (self.all_depths - self.all_depths.min()) / (self.all_depths.max() - self.all_depths.min())
        # self.all_depths /= self.all_depths.max()
        # mask = self.all_depths != 0
        # self.all_depths = (self.all_depths - self.all_depths[mask].min()) / (self.all_depths[mask].max() - self.all_depths[mask].min())
        self.all_times = torch.stack(self.all_times)[200:700]
        self.all_times = ((self.all_times - self.all_times.min()) / (self.all_times.max() - self.all_times.min()) * 2.0 - 1.0)

        if not self.is_stack:
            train_mask = np.mod(np.arange(self.poses.shape[0])+1,8)!=0
            self.poses = self.poses[train_mask]
            self.all_rays = list(self.all_rays[train_mask].split(1, dim=0))
            self.all_rgbs = list(self.all_rgbs[train_mask].split(1, dim=0))
            self.all_depths = list(self.all_depths[train_mask].split(1, dim=0))
            self.all_times = list(self.all_times[train_mask].split(1, dim=0))

            self.all_rays = torch.cat([t.squeeze() for t in self.all_rays], 0)  # (len(self.meta['frames])*h*w, 6)
            self.all_rgbs = torch.cat([t.squeeze() for t in self.all_rgbs], 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_depths = torch.cat([t.squeeze() for t in self.all_depths], 0)  # (len(self.meta['frames])*h*w, 1)
            self.all_times = torch.cat([t.squeeze() for t in self.all_times], 0).unsqueeze(1)

        else:
            test_mask = np.mod(np.arange(self.poses.shape[0])+1,8)==0
            self.poses = self.poses[test_mask]
            self.all_rays = self.all_rays[test_mask]  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = self.all_rgbs[test_mask].reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            self.all_depths = self.all_depths[test_mask].reshape(-1,*self.img_wh[::-1], 1)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)
            self.all_times = self.all_times[test_mask]

    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        poses = torch.eye(4).unsqueeze(0).repeat(self.poses.shape[0], 1, 1)
        poses[:, :, :] = self.poses
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'time': self.all_times[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            time = self.all_times[idx]
            # mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'time': time}
            
        if self.depth_data:
            sample['depths'] = self.all_depths[idx]

        return sample
    
    def get_val_pose(self):
        """
        Get validation poses and times (NeRF-like rotating cameras poses).
        """
        render_poses = torch.stack(
            [
                pose_spherical(angle, 0.0, 0.0)
                for angle in np.linspace(60, 100, 500 + 1)[:-1]
            ],
            0,
        )
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, 1.0 * render_times

    def get_val_rays(self):
        """
        Get validation rays and times (NeRF-like rotating cameras poses).
        """
        val_poses, val_times = self.get_val_pose()  # get valitdation poses and times
        # val_poses = torch.Tensor([[0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]).unsqueeze(0).repeat(400, 1, 1)
        rays_all = []  # initialize list to store [rays_o, rays_d]
        # val_poses = self.poses[0].unsqueeze(0).repeat(100, 1, 1)

        for i in range(val_poses.shape[0]):
            c2w = val_poses[i]
            rays_o, rays_d = get_rays(self.directions, c2w.float())  # both (h*w, 3)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            rays_all.append(rays)
        return rays_all, torch.FloatTensor(val_times)

    
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
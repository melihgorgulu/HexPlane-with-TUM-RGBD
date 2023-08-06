import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T


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
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-5.0, -5, -5], [5, 5, 5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [0.1, 10.0]
        self.near = self.near_far[0]
        self.far = self.near_far[1]
        self.world_bound_scale = 1.1

        cal_fine_bbox = True
        if cal_fine_bbox and split=="train":
            xyz_min, xyz_max = self.compute_bbox()
            self.scene_bbox = torch.stack((xyz_min, xyz_max), dim=0)
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample=downsample

        self.ndc_ray = False
        self.depth_data = False

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):

        with open(os.path.join(self.root_dir, f"transforms.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = int(self.meta['w']/self.downsample), int(self.meta['h']/self.downsample)
        self.img_wh = [w,h]
        # self.focal_x = 0.5 * w / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        # self.focal_y = 0.5 * h / np.tan(0.5 * self.meta['camera_angle_y'])  # original focal length
        # self.cx, self.cy = self.meta['cx'],self.meta['cy']

        self.focal_x = 542.822841
        self.focal_y = 542.576870
        self.cx = 315.593520
        self.cy = 237.756098

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal_x,self.focal_y], center=[self.cx, self.cy])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal_x,0,self.cx],[0,self.focal_y,self.cy],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.all_times = []
        timestamps = []

        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(-1, w*h).permute(1, 0)  # (h*w, 4) RGBA
            if img.shape[-1]==4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]


            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

            image_name = image_path.split('/')[-1].replace('.png', '')
            timestamps.append(float(image_name))

        indices = np.argsort(self.image_paths)
        self.image_paths = np.array(self.image_paths)[indices].tolist()
        self.poses = np.array(self.poses)[indices].tolist()
        self.all_rays = np.array(self.all_rays)[indices].tolist()
        self.all_rgbs = np.array(self.all_rgbs)[indices].tolist()
        timestamps = np.array(timestamps)[indices]
        timestamps -= timestamps[0]
        timestamps /= timestamps[-1]

        for i in range(len(timestamps)):
            timestamp = torch.tensor(timestamps[i], dtype=torch.float32).expand(rays_o.shape[0], 1)
            self.all_times.append(timestamp)

        self.poses = torch.stack(self.poses[:400])
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays[:400], 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs[:400], 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_times = torch.cat(self.all_times[:400], 0)

#             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays[:400], 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs[:400], 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)
            self.all_times = torch.stack(self.all_times[:400], 0)

        self.all_times = (self.all_times / self.all_times.max() * 2.0 - 1.0)


    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

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
        return sample
    
    def get_val_pose(self):
        """
        Get validation poses and times (NeRF-like rotating cameras poses).
        """
        render_poses = torch.stack(
            [
                pose_spherical(angle, 0.0, 0.0)
                for angle in np.linspace(95, 95, 400 + 1)[:-1]
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

        for i in range(val_poses.shape[0]):
            c2w = val_poses[i]
            rays_o, rays_d = get_rays(self.directions, c2w.float())  # both (h*w, 3)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            rays_all.append(rays)
        return rays_all, torch.FloatTensor(val_times)

    # def get_val_rays(self, times=0, time=0.0):

    #     pose = self.poses[0]

    #     # times = torch.linspace(0.0, 1.0, self.poses.shape[0]) * 2.0 - 1.0

    #     xs = list(np.zeros(times))
    #     ys = list(np.zeros(times))
    #     zs = list(np.zeros(times))

    #     rot_deg = 0.5
    #     rxs = list(np.zeros(2)) + list(np.ones(2) * -rot_deg) + list(np.zeros(2)) + list(np.ones(2)*1.5*rot_deg) + list(np.zeros(2)) + list(np.ones(2)*-2*rot_deg) + list(np.ones(2)*1.5*rot_deg)
    #     rys = list(np.ones(2) * rot_deg) + list(np.zeros(2)) + list(np.ones(2) * -1.5*rot_deg) + list(np.zeros(2)) + list(np.ones(2)*2*rot_deg) + list(np.zeros(2)) + list(np.ones(2)*-1.5*rot_deg)
    #     rzs = list(np.zeros(times))

    #     if times==0: # useful for simulating fixed pose for all timesteps
    #         rays_all = []  # initialize list to store [rays_o, rays_d]
    #         poses = pose.reshape(1,pose.shape[0], pose.shape[1])
    #         time_a = time
    #         for i in range(100):
    #             c2w = torch.FloatTensor(poses)
    #             rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
    #             rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
    #             rays_all.append(rays)
    #         time_a = torch.linspace(0.0, 1.0, 100) * 2.0 - 1.0

    #     for i in range(times):
    #         # x, y, z, rx, ry, rz = ...
    #         x, y, z, rx, ry, rz = xs[i], 0, zs[i], rxs[i], rys[i], rzs[i]
    #         c2w = np.array([[1, 0, 0, x],
    #                         [0, 1, 0, y],
    #                         [0, 0, 1, z],
    #                         [0, 0, 0, 1]])
    #         R_X = np.array([
    #             [1.0, 0.0, 0.0, 0.0],
    #             [0.0, np.cos(rx * 3.1415 / 180), -np.sin(rx * 3.1415 / 180), 0.0],
    #             [0.0, np.sin(rx * 3.1415 / 180), np.cos(rx * 3.1415 / 180), 0.0],
    #             [0.0, 0.0, 0.0, 1.0]])

    #         R_Y = np.array([
    #             [np.cos(ry * 3.1415 / 180), 0.0, np.sin(ry * 3.1415 / 180), 0.0],
    #             [0.0, 1.0, 0.0, 0.0],
    #             [-np.sin(ry * 3.1415 / 180), 0.0, np.cos(ry * 3.1415 / 180), 0.0],
    #             [0.0, 0.0, 0.0, 1.0]])

    #         R_Z = np.array([
    #             [np.cos(rz * 3.1415 / 180), -np.sin(rz * 3.1415 / 180), 0.0, 0.0],
    #             [np.sin(rz * 3.1415 / 180), np.cos(rz * 3.1415 / 180), 0.0, 0.0],
    #             [0.0, 0.0, 1.0, 0.0],
    #             [0.0, 0.0, 0.0, 1.0]])

    #         if i == 0:
    #             pose = pose.detach().cpu().numpy()

    #         new_pose = R_X @ R_Y @ R_Z @ c2w @ pose[:4, :]

    #         new_time = time + (i + 1) * 0.00001

    #         new_pose = new_pose.reshape(1, new_pose.shape[0], new_pose.shape[1])#[:, :3]

    #         if i == 0:
    #             poses = np.concatenate((pose.reshape(1, pose.shape[0], pose.shape[1]), new_pose))
    #             time_a = np.concatenate((time, new_time))
    #             time = new_time
    #             pose = new_pose[0]
    #         else:
    #             poses = np.concatenate((poses, new_pose))
    #             time_a = np.concatenate((time_a, new_time))
    #             pose = new_pose[0]
    #             time = new_time

    #     rays_all = []  # initialize list to store [rays_o, rays_d]
    #     for i in range(poses.shape[0]):
    #         c2w = torch.FloatTensor(poses[i])
    #         rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
    #         rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
    #         rays_all.append(rays)

    #     return rays_all, torch.FloatTensor(time_a)
    
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
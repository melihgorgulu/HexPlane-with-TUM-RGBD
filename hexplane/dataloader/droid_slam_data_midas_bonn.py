import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
from scipy.spatial.transform import Rotation as R
from evo.core import lie_algebra as lie

from torchvision.transforms import Compose
from ..model.midas.dpt_depth import DPTDepthModel
from ..model.midas.transforms import Resize, NormalizeImage, PrepareForNet

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


class BonnDataset(Dataset):
    def __init__(self, images, poses, timestamps, intrinsics,
                 scene_bbox_min, scene_bbox_max, split='train', downsample=1.0,
                 is_stack=False, N_vis=-1, cal_fine_bbox=False):
        
        if split == 'train':
            assert(is_stack == False)

        self.N_vis = N_vis
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()

        self.ndc_ray = False # currently does not work
        self.depth_data = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load Midas depth model
        self.midas = DPTDepthModel(
            path='models/dpt_large-midas-2f21e586.pt',
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "lower_bound"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.midas_transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        self.images = images
        self.poses = poses
        self.timestamps = timestamps
        self.intrinsics = intrinsics

        self.white_bg = False
        self.near_far = [0.0, 10.0]
        self.near = self.near_far[0]
        self.far = self.near_far[1]
        self.world_bound_scale = 1.1

        self.scene_bbox = torch.Tensor([scene_bbox_min, scene_bbox_max])
        self.blender2opencv = torch.Tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        if cal_fine_bbox and split == "train":
            xyz_min, xyz_max = self.compute_bbox()
            self.scene_bbox = torch.stack((xyz_min, xyz_max), dim=0)
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
   
    def read_meta(self):

        traj_droid = self.poses

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

        self.focal_x, self.focal_y, self.cx, self.cy = self.intrinsics

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal_x,self.focal_y], center=[self.cx, self.cy])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal_x,0,self.cx],[0,self.focal_y,self.cy],[0,0,1]]).float()
        
        self.midas.eval()
        self.midas.to(self.device)

        self.poses = []
        timestamps = []
        self.all_times = []

        if self.split == "train":
            idxs = list(range(0, len(self.images)))
            idxs = [num for num in idxs if (num+1) % 8 != 0]
            N_images_train = len(idxs)
            self.all_rays = torch.zeros(h*w*N_images_train, 6)
            self.all_rgbs = torch.zeros(h*w*N_images_train, 3)
            self.all_depths = torch.zeros(h*w*N_images_train)
        else:
            idxs = list(range(0, len(self.images)))
            idxs = [num for num in idxs if (num+1) % 8 == 0]
            self.all_rays = []
            self.all_rgbs = []
            self.all_depths = []

        for t, i in enumerate(tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})')):#img_list:#
            c2w = torch.from_numpy(poses[i]).float()
            self.poses += [c2w]

            image_path = self.images[i]
            img = Image.open(image_path)
            image = img.copy()
            
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            img = img.view(-1, w*h).permute(1, 0) # (h*w, 3) RGBA
            if img.shape[-1]==4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

            if self.is_stack:
                self.all_rgbs += [img]
            else:
                self.all_rgbs[t*h*w: (t+1)*h*w] = img

            # compute midas depth
            if self.depth_data:
                # always use full resolution RGB for depth map
                image = self.transform(image)
                img = self.midas_transform({"image": image.permute(1, 2, 0).cpu().numpy()})["image"]

                with torch.no_grad():
                    img = torch.from_numpy(img).to(self.device).unsqueeze(0)
                    disp = self.midas.forward(img)
                    disp = (
                        torch.nn.functional.interpolate(disp.unsqueeze(1), size=[h, w],
                            mode="bicubic", align_corners=False).squeeze()
                            ).clamp(1e-6, torch.inf)

                if self.is_stack:
                    self.all_depths += [1 / disp.view(-1).cpu()]
                else:
                    self.all_depths[t*h*w: (t+1)*h*w] = 1 / disp.view(-1)

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            if self.ndc_ray:
                rays_o, rays_d = ndc_rays_blender_tum(h, w, [self.focal_x, self.focal_y], 1.0, rays_o, rays_d)

            if self.is_stack:
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            else:
                self.all_rays[t*h*w: (t+1)*h*w] = torch.cat([rays_o, rays_d], 1)

            timestamps.append(self.timestamps[i])

        # timestamps = np.array(timestamps)
        # timestamps -= timestamps[0]
        # timestamps /= timestamps[-1]

        for i in range(len(timestamps)):
            timestamp = torch.tensor(timestamps[i], dtype=torch.float64).expand(rays_o.shape[0], 1)
            self.all_times.append(timestamp)

        self.poses = torch.stack(self.poses)

        if not self.is_stack:
            self.all_times = torch.cat(self.all_times, 0)
        else:
            print("Stacking ...")
            self.all_rays = torch.stack(self.all_rays, 0)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1], 3)
            if self.depth_data:
                self.all_depths = torch.stack(self.all_depths, 0).reshape(-1, *self.img_wh[::-1], 1)
            self.all_times = torch.stack(self.all_times, 0)
            print("Stack performed !!")

        # Normalization over all timestamps
        self.timestamps = np.array(self.timestamps)
        self.all_times = ((self.all_times - self.timestamps.min()) / (self.timestamps.max() - self.timestamps.min()) * 2.0 - 1.0).float()

    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        poses = torch.eye(4).unsqueeze(0).repeat(self.poses.shape[0], 1, 1)
        poses[:, :, :] = self.poses
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(poses)[:,:3]
        
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
        xyz_min = torch.tensor([np.inf, np.inf, np.inf])
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
    
    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose
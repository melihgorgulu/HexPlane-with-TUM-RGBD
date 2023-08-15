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

# def normalize(v):
#     """Normalize a vector."""
#     return v / np.linalg.norm(v)

# def average_poses(poses):
#     """
#     Calculate the average pose, which is then used to center all poses
#     using @center_poses. Its computation is as follows:
#     1. Compute the center: the average of pose centers.
#     2. Compute the z axis: the normalized average z axis.
#     3. Compute axis y': the average y axis.
#     4. Compute x' = y' cross product z, then normalize it as the x axis.
#     5. Compute the y axis: z cross product x.

#     Note that at step 3, we cannot directly use y' as y axis since it's
#     not necessarily orthogonal to z axis. We need to pass from x to y.
#     Inputs:
#         poses: (N_images, 3, 4)
#     Outputs:
#         pose_avg: (3, 4) the average pose
#     """
#     # 1. Compute the center
#     center = poses[..., 3].mean(0)  # (3)

#     # 2. Compute the z axis
#     z = normalize(poses[..., 2].mean(0))  # (3)

#     # 3. Compute axis y' (no need to normalize as it's not the final output)
#     y_ = poses[..., 1].mean(0)  # (3)

#     # 4. Compute the x axis
#     x = normalize(np.cross(z, y_))  # (3)

#     # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
#     y = np.cross(x, z)  # (3)

#     pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

#     return pose_avg

# def center_poses(poses, blender2opencv):
#     """
#     Center the poses so that we can use NDC.
#     See https://github.com/bmild/nerf/issues/34
#     Inputs:
#         poses: (N_images, 3, 4)
#     Outputs:
#         poses_centered: (N_images, 3, 4) the centered poses
#         pose_avg: (3, 4) the average pose
#     """
#     poses = poses @ blender2opencv
#     pose_avg = average_poses(poses)  # (3, 4)
#     pose_avg_homo = np.eye(4)
#     pose_avg_homo[
#         :3
#     ] = pose_avg  # convert to homogeneous coordinate for faster computation
#     pose_avg_homo = pose_avg_homo
#     # by simply adding 0, 0, 0, 1 as the last row
#     last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
#     poses_homo = np.concatenate(
#         [poses, last_row], 1
#     )  # (N_images, 4, 4) homogeneous coordinate

#     poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
#     #     poses_centered = poses_centered  @ blender2opencv
#     poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

#     return poses_centered, pose_avg_homo


class iPhoneDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):

        self.N_vis = -1
        self.root_dir = "/media/nico/TOSHIBA EXT/iPhone/iphone/paper-windmill"
        self.split = split
        self.is_stack = is_stack
        self.downsample = 2
        self.define_transforms()

        self.blender2opencv = torch.Tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # self.blender2opencv = np.eye(4)
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = False
        self.near_far = [0.10891095985144743, 0.5122392056138323]
        self.near = self.near_far[0]
        self.far = self.near_far[1]

        self.scene_bbox = torch.Tensor([[
            -0.28376930952072144,
            -0.25683197379112244,
            -0.321733295917511
        ],
        [
            0.28376930952072144,
            0.25683197379112244,
            0.3217332661151886
        ]])
        
        self.ndc_ray = False
        self.depth_data = True

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):

        stride = 1

        with open(os.path.join(self.root_dir, 'dataset.json'), 'r') as json_file:
            dataset_json = json.load(json_file)

        image_names = dataset_json['train_ids']

        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_depths = []
        self.all_times = []
        timestamps = []

        for t, imfile in tqdm(enumerate(image_names[::stride]), desc=f'Loading data {self.split} ({len(image_names[::stride])})'):

            file_path = os.path.join(self.root_dir, 'camera', imfile + '.json')

            with open(file_path, 'r') as json_file:
                camera_json = json.load(json_file)

            focal_length = camera_json["focal_length"]
            principal_point = camera_json["principal_point"]
            w, h = camera_json["image_size"]
            w //= self.downsample
            h //= self.downsample

            # Calculate the intrinsic matrix elements
            self.focal_x = self.focal_y = focal_length
            self.cx = principal_point[0]
            self.cy = principal_point[1]
            
            # Load image
            image_path = os.path.join(self.root_dir, 'rgb/2x', imfile + '.png')     
            img = Image.open(image_path)
            img = self.transform(img)  # (4, h, w)
            img = img.view(-1, w*h).permute(1, 0)  # (h*w, 4) RGBA
            if img.shape[-1]==4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]

            # Save timestamp
            timestamps.append(float(t))

            # Load depth
            depth_path = os.path.join(self.root_dir, 'depth/2x', imfile + '.npy')
            depth = torch.from_numpy(np.load(depth_path)).view(w*h, -1)
            self.all_depths.append(depth)

            # Load pose
            file_path = os.path.join(self.root_dir, 'camera', imfile + '.json')

            with open(file_path, 'r') as json_file:
                camera_json = json.load(json_file)

            orientation = camera_json["orientation"]
            position = camera_json["position"]

            c2w = torch.eye(4, 4)
            c2w[:3, :3] = torch.Tensor(orientation)
            c2w[:3, 3] = (torch.Tensor(position) - torch.Tensor([-0.20061972737312317, 0.1705830693244934, -1.1717479228973389])) * 0.2715916376300303
            c2w = torch.inverse(c2w)
            self.poses.append(c2w)

            # Create rays
            # ray directions for all pixels, same for all images (same H, W, focal)
            self.directions = get_ray_directions(h, w, [self.focal_x,self.focal_y], center=[self.cx, self.cy])  # (h, w, 3)
            self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
            self.intrinsics = torch.tensor([[self.focal_x,0,self.cx],[0,self.focal_y,self.cy],[0,0,1]]).float()

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        self.img_wh = [w,h]

        timestamps = np.array(timestamps)
        timestamps -= timestamps[0]
        timestamps /= timestamps[-1]

        for i in range(len(timestamps)):
            timestamp = torch.tensor(timestamps[i], dtype=torch.float32).expand(rays_o.shape[0], 1)
            self.all_times.append(timestamp)

        self.poses = torch.stack(self.poses)[:10]
        self.all_rays = torch.stack(self.all_rays, 0)[0:10]
        self.all_rgbs = torch.stack(self.all_rgbs, 0)[0:10]
        self.all_depths = torch.stack(self.all_depths, 0)[0:10]
        self.all_times = torch.stack(self.all_times)[0:10]
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
            test_mask = np.mod(np.arange(self.poses.shape[0])+1,8)!=0
            self.poses = self.poses[test_mask]
            self.all_rays = self.all_rays[test_mask]  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = self.all_rgbs[test_mask].reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            self.all_depths = self.all_depths[test_mask].reshape(-1,*self.img_wh[::-1], 1)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)
            self.all_times = self.all_times[test_mask]

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
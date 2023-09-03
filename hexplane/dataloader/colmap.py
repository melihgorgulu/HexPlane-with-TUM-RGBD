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


class ColmapDataset(Dataset):
    def __init__(self, image_dir, poses, intrinsics, scene_bbox_min, scene_bbox_max, split='train', downsample=1.0,
                  is_stack=False, N_vis=-1, cal_fine_bbox=False):

        if split == 'train':
            assert(is_stack == False)

        self.N_vis = N_vis
        self.root_dir = poses
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()

        self.ndc_ray = False
        self.depth_data = False

        self.image_dir = image_dir
        self.intrinsics = intrinsics

        self.white_bg = True
        self.near_far = [0.0, 20.0] # TODO: check this
        self.near = self.near_far[0]
        self.far = self.near_far[1]
        self.world_bound_scale = 1.1

        self.scene_bbox = torch.Tensor([scene_bbox_min, scene_bbox_max])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        if cal_fine_bbox and split == "train":
            xyz_min, xyz_max = self.compute_bbox()
            self.scene_bbox = torch.stack((xyz_min, xyz_max), dim=0)
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)


    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):

        frame_rate = 32

        with open(self.root_dir, 'r') as f:
            self.meta = json.load(f)

        sorted_poses = sorted(self.meta['frames'], key=lambda x: x['file_path'])

        w, h = int(self.meta['w']/self.downsample), int(self.meta['h']/self.downsample)
        self.img_wh = [w,h]

        image_list = os.path.join(self.image_dir, 'rgb.txt')
        depth_list = os.path.join(self.image_dir, 'depth.txt')

        # parse data
        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)

        # associate dept rgb and ground truth data
        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_image)  # (image, depth, pose) pairs
        
        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]  # take association index for image and then get timestamp
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.focal_x, self.focal_y, self.cx, self.cy = self.intrinsics

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal_x,self.focal_y], center=[self.cx, self.cy])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal_x,0,self.cx],[0,self.focal_y,self.cy],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        timestamps = []
        self.all_times = []

        if self.split == "train":
            idxs = list(range(0, len(associations)))
            idxs = [num for num in idxs if (num+1) % 8 != 0]
            N_images_train = len(idxs)
            self.all_rays = torch.zeros(h*w*N_images_train, 6)
            self.all_rgbs = torch.zeros(h*w*N_images_train, 3)
            self.all_depths = torch.zeros(h*w*N_images_train)
        else:
            idxs = list(range(0, len(associations)))
            idxs = [num for num in idxs if (num+1) % 8 == 0]
            self.all_rays = []
            self.all_rgbs = []
            self.all_depths = []

        for t, ix in enumerate(tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})')):#img_list:#

            (i, j, k) = associations[ix]

            frame = sorted_poses[k]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.image_dir, image_data[i, 1])
            self.image_paths += [image_path]
            img = Image.open(image_path)
            
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(-1, w*h).permute(1, 0)  # (h*w, 4) RGBA
            if img.shape[-1]==4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            if self.is_stack:
                self.all_rgbs += [img]
            else:
                self.all_rgbs[t*h*w: (t+1)*h*w] = img

            # load depth
            depth = cv2.imread(os.path.join(self.image_dir, depth_data[j, 1]), cv2.IMREAD_UNCHANGED)
            depth = depth[:, :, np.newaxis].astype(np.int32) # Pytorch cannot handle uint16
            depth = torch.from_numpy(depth) / 5000
            depth = depth.view(1, -1).permute(1, 0)

            if self.is_stack:
                self.all_depths += [depth.view(-1).cpu()]
            else:
                self.all_depths[t*h*w: (t+1)*h*w] = depth.view(-1)

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            if self.is_stack:
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            else:
                self.all_rays[t*h*w: (t+1)*h*w] = torch.cat([rays_o, rays_d], 1)

            image_name = image_path.split('/')[-1].replace('.png', '')
            timestamps.append(float(image_name))

        # indices = np.argsort(self.image_paths)
        # self.image_paths = np.array(self.image_paths)[indices].tolist()
        # self.poses = np.array(self.poses)[indices].tolist()
        # self.all_rays = np.array(self.all_rays)[indices].tolist()
        # self.all_rgbs = np.array(self.all_rgbs)[indices].tolist()
        # self.all_depths = np.array(self.all_depths)[indices].tolist()
        # timestamps = np.array(timestamps)[indices]
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
        self.all_times = ((self.all_times.cuda() - float(os.path.basename(sorted_poses[0]['file_path']).rsplit('.png', 1)[0])) /
                          (float(os.path.basename(sorted_poses[-1]['file_path']).rsplit('.png', 1)[0]) - float(os.path.basename(sorted_poses[0]['file_path']).rsplit('.png', 1)[0])) * 2.0 - 1.0).float()

    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]
        
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
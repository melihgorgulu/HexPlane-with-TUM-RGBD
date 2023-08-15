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

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


class BonngbdDataset(Dataset):
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
        self.depth_data = True

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):
        w, h = int(640/self.downsample), int(480/self.downsample)
        self.img_wh = [w,h]

        self.focal_x = 542.822841
        self.focal_y = 542.576870
        self.cx = 315.593520
        self.cy = 237.756098

        frame_rate = 32

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal_x, self.focal_y], center=[self.cx, self.cy])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal_x,0,self.cx],[0,self.focal_y,self.cy],[0,0,1]]).float()

        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(self.root_dir, 'groundtruth.txt')):
            pose_list = os.path.join(self.root_dir, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(self.root_dir, 'pose.txt')):
            pose_list = os.path.join(self.root_dir, 'pose.txt')

        image_list = os.path.join(self.root_dir, 'rgb.txt')
        depth_list = os.path.join(self.root_dir, 'depth.txt')

        # parse data
        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        # associate dept rgb and ground truth data
        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)  # (image, depth, pose) pairs

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]  # take association index for image and then get timestamp
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depths = []
        self.all_times = []
        timestamps = []

        for ix in tqdm(indicies, desc=f'Loading data {self.split} ({len(indicies)})'):#img_list:#

            (i, j, k) = associations[ix]  # image idx, depth idx, pose idx

            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
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

            # load depth
            depth = cv2.imread(os.path.join(self.root_dir, depth_data[j, 1]), cv2.IMREAD_UNCHANGED)
            depth = depth[:, :, np.newaxis].astype(np.int32) # Pytorch cannot handle uint16
            depth = torch.from_numpy(depth) / 5000
            depth = depth.view(1, -1).permute(1, 0)
            self.all_depths += [depth]

            image_name = image_path.split('/')[-1].replace('.png', '')
            timestamps.append(float(image_name))

        # indices = np.argsort(self.image_paths)
        # self.image_paths = np.array(self.image_paths)[indices].tolist()
        # self.poses = np.array(self.poses)[indices].tolist()
        # self.all_rays = np.array(self.all_rays)[indices].tolist()
        # self.all_rgbs = np.array(self.all_rgbs)[indices].tolist()
        # timestamps = np.array(timestamps)[indices]
        timestamps = np.array(timestamps)
        timestamps -= timestamps[0]
        timestamps /= timestamps[-1]
        
        # Transform TUM RGB-D format to NeRF format
        self.poses2nerf()

        # with open(os.path.join(self.root_dir, f"transforms.json"), 'r') as f:
        #     self.meta = json.load(f)

        for i in range(len(timestamps)):
            self.poses[i] = self.poses[i] @ self.blender2opencv
            rays_o, rays_d = get_rays(self.directions, self.poses[i].float())  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            timestamp = torch.tensor(timestamps[i], dtype=torch.float32).expand(rays_o.shape[0], 1)
            self.all_times.append(timestamp)

        self.poses = torch.stack(self.poses[:100])
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays[:100], 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs[:100], 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_times = torch.cat(self.all_times[:100], 0)
            self.all_depths = torch.cat(self.all_depths[:100], 0)

#             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays[:100], 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs[:100], 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)
            self.all_times = torch.stack(self.all_times[:100], 0)
            self.all_depths = torch.stack(self.all_depths[:100], 0)

        self.all_times = (self.all_times / self.all_times.max() * 2.0 - 1.0)
        self.valid_rays = self.all_rays.clone().detach()


    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses.float())[:,:3]

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
                pose_spherical(0.0, 0.0, 0.0)
                for angle in np.linspace(-180, 180, 400 + 1)[:-1]
            ],
            0,
        )
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, 1.0 * render_times

    def get_val_rays(self):
        """
        Get validation rays and times (NeRF-like rotating cameras poses).
        """
        val_poses, val_times = self.get_val_pose()  # get validation poses and times
        # val_poses = torch.Tensor([[0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]).unsqueeze(0).repeat(400, 1, 1)
        val_poses = self.poses
        rays_all = []  # initialize list to store [rays_o, rays_d]

        for i in range(val_poses.shape[0]):
            c2w = val_poses[0]
            rays_o, rays_d = get_rays(self.directions, c2w.float())  # both (h*w, 3)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            rays_all.append(rays)
        return rays_all, torch.FloatTensor(val_times)
        # return self.valid_rays[0:100], torch.FloatTensor(val_times)
    
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
    
    def poses2nerf(self):
        t = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # up = np.zeros(3)
        # for pose in self.poses:
        #     up += pose[0:3,1]

        # up = up / np.linalg.norm(up)
        # print("up vector was", up)
        # R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
        # R = np.pad(R,[0,1])
        # R[-1, -1] = 1

        # for f in range(len(self.poses)):
        #     self.poses[f] = np.matmul(R, self.poses[f]) # rotate up to be the z axis

        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in self.poses:
            mf = f[0:3,:]
            for g in self.poses:
                mg = g[0:3,:]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if w > 0.00001:
                    totp += p*w
                    totw += w
        if totw > 0.0:
            totp /= totw
        print(totp) # the cameras are looking at totp
        for f in range(len(self.poses)):
            self.poses[f][0:3,3] -= totp

        avglen = 0.
        for f in self.poses:
            avglen += np.linalg.norm(f[0:3,3])
        avglen /= len(self.poses)
        print("avg camera distance from origin", avglen)
        for f in range(len(self.poses)):
            self.poses[f][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

        self.poses = [torch.from_numpy(np.dot(t, p)) for p in self.poses]

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

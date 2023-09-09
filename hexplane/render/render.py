import os

import imageio
import numpy as np
import torch
from pytorch_msssim import ms_ssim as MS_SSIM
from tqdm.auto import tqdm
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torchviz import make_dot
import random

from HexPlane.hexplane.render.util.metric import rgb_lpips, rgb_ssim, masked_rgb_ssim
from HexPlane.hexplane.render.util.util import visualize_depth_numpy
from HexPlane.hexplane.render.util.learn_pose import LearnPose

def constrain_parameter(param, max_value):
    with torch.no_grad():
        param.data = torch.min(param.data, torch.tensor(max_value))
        param.data = torch.max(param.data, torch.tensor(-max_value))

def mse2psnr(mse):
    """
    :param mse: scalar
    :return:    scalar np.float32
    """
    mse = np.maximum(mse, 1e-10)  # avoid -inf or nan when mse is very small.
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)

def opt_eval_pose_one_epoch(model, idxs, test_dataset, eval_pose_param_net, cameras, optimizer_eval_pose, N_samples, ndc_ray, white_bg, my_devices):    
    # model.eval()
    eval_pose_param_net.train()

    L2_loss_epoch = []
    refined_camera = cameras.copy()
    for i in idxs:
        # constrain_parameter(eval_pose_param_net.r, np.radians(1))

        data = test_dataset[i]
        gt_rgb, sample_times, cam_id = data["rgbs"].view(-1, 3), data["time"], data["cam_id"]
        H, W, _ = data["rgbs"].shape
        chunk_size = 1024

        camera = cameras[cam_id]
        c2w = eval_pose_param_net(cam_id)  # (4, 4)
        camera.orientation = c2w[:3, :3]
        camera.position = c2w[:3, 3]
        refined_camera[cam_id] = camera
        pixels = camera.get_pixel_centers_torch().to(my_devices)
        rays_d = camera.pixels_to_rays_torch(pixels, camera.orientation).view([-1,3])
        rays_o = camera.position.unsqueeze(0).expand_as(rays_d)
        rays_o.retain_grad()
        rays_cat = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
        rays_cat.retain_grad()

        # sample pixel on an image and their rays for training.
        indicies = torch.randint(0, sample_times.shape[0], [1024])
        # indicies = torch.arange(0, 4096)
        # indicies = torch.arange(0, W)
        # indicies += torch.arange(H*W - W, H*W)
        # skip = (2*W - chunk_size) // 4
        # indicies = torch.cat((torch.arange(skip, W-skip), torch.arange(H*W + skip + 1 - W, H*W + 1 - skip)))

        ray_selected_cam = rays_cat[indicies]  # (N_select, 6)
        ray_selected_cam.retain_grad()
        img_selected = gt_rgb[indicies]  # (N_select, 3)
        sample_times = sample_times[indicies]

        # render an image using selected rays, pose, sample intervals, and the network
        rays = ray_selected_cam.view(-1, ray_selected_cam.shape[-1])
        rays.retain_grad()
        times = sample_times.view(-1, sample_times.shape[-1])
        rgb_map, _, _, _, _ = OctreeRender_trilinear_fast(
            rays,
            times,
            model,
            chunk=chunk_size,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=my_devices,
        )

        L2_loss = F.mse_loss(rgb_map, img_selected.to(my_devices))  # loss for one image
        # dot = make_dot(L2_loss, params=dict(model.named_parameters()))
        # dot.view()
        L2_loss.retain_grad()
        # L2_loss = F.mse_loss(val_poses, torch.zeros_like(val_poses))  # loss for one image
        L2_loss.backward()

        optimizer_eval_pose.step()
        optimizer_eval_pose.zero_grad()

        L2_loss_epoch.append(L2_loss.item())

    L2_loss_mean = np.mean(L2_loss_epoch)
    mean_losses = {
        'L2': L2_loss_mean,
    }

    return mean_losses, refined_camera

def OctreeRender_trilinear_fast(
    rays,
    time,
    model,
    chunk=4096,
    N_samples=-1,
    ndc_ray=False,
    white_bg=True,
    is_train=False,
    device="cuda",
):
    """
    Batched rendering function.
    """
    rgbs, alphas, depth_maps, z_vals = [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        time_chunk = time[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)

        rgb_map, depth_map, alpha_map, z_val_map = model(
            rays_chunk,
            time_chunk,
            is_train=is_train,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            N_samples=N_samples,
        )
        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        alphas.append(alpha_map)
        z_vals.append(z_val_map)
    return (
        torch.cat(rgbs),
        torch.cat(alphas),
        torch.cat(depth_maps),
        torch.cat(z_vals),
        None,
    )

# TODO: Check here
# @torch.no_grad()
def evaluation(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=5,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
):
    """
    Evaluate the model on the test rays and compute metrics.
    """
    PSNRs, rgb_maps, depth_maps, gt_depth_maps = [], [], [], []
    msssims, ssims, l_alex, l_vgg = [], [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(len(test_dataset) // N_vis, 1)
    idxs = list(range(0, len(test_dataset), img_eval_interval))

    val_poses = torch.stack([torch.eye(4)] * len(idxs), dim=0)
    all_cameras = []
    for t, i in enumerate(idxs):
        data = test_dataset[i]
        if "camera" in data.keys():
            camera = data["camera"]
            all_cameras += [camera]
            if isinstance(camera.orientation, np.ndarray):
                val_poses[t, :3, :3] = torch.from_numpy(camera.orientation)
            elif isinstance(camera.orientation, torch.Tensor):
                val_poses[t, :3, :3] = camera.orientation

            if isinstance(camera.position, np.ndarray):
                val_poses[t, :3, 3] = torch.from_numpy(camera.position)
            elif isinstance(camera.orientation, torch.Tensor):
                val_poses[t, :3, 3] = camera.position

    if len(all_cameras) != 0:
        '''Set Optimizer'''
        init_c2ws = torch.stack([val_poses[0], val_poses[-1]])
        eval_pose_param_net = LearnPose(2, True, True, init_c2ws).to(device)
        optimizer_eval_pose = torch.optim.Adam(eval_pose_param_net.parameters(), lr=1e-4)
        scheduler_eval_pose = torch.optim.lr_scheduler.MultiStepLR(optimizer_eval_pose,
                                                                   milestones=[100, 200, 300, 400],
                                                                   gamma=0.5)
        
        num_samples_refine = min(20, len(idxs)//3)
        num_images_first_val = test_dataset.num_first_cam

        if test_dataset.num_first_cam == len(test_dataset):
            filtered_list_1 = [x for x in idxs if x < num_images_first_val]
            idxs_pose_refine = random.sample(filtered_list_1, num_samples_refine*2)
        else:
            filtered_list_1 = [x for x in idxs if x < num_images_first_val]
            filtered_list_2 = [x for x in idxs if x >= num_images_first_val]
            idxs_pose_refine = random.sample(filtered_list_1, num_samples_refine)
            idxs_pose_refine += random.sample(filtered_list_2, num_samples_refine)

        refined_camera = [all_cameras[0], all_cameras[-1]]

        '''Optimise eval poses'''
        for epoch_i in tqdm(range(100), desc='optimising eval'):
            
            mean_losses, refined_camera = opt_eval_pose_one_epoch(model, idxs_pose_refine, test_dataset, eval_pose_param_net, refined_camera, optimizer_eval_pose,
                                                N_samples, ndc_ray, white_bg, device)

            opt_L2_loss = mean_losses['L2']
            opt_pose_psnr = mse2psnr(opt_L2_loss)
            scheduler_eval_pose.step()

            tqdm.write('{0:6d} ep: Opt: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, opt_L2_loss, opt_pose_psnr))

            # refined_camera = [all_cameras[0], all_cameras[-1]]
        
    for t, idx in enumerate(tqdm(idxs)):
        data = test_dataset[idx]
        if len(all_cameras) != 0:
            samples, gt_rgb, sample_times, cam_id = data["rays"], data["rgbs"], data["time"], data["cam_id"]
        else:
            samples, gt_rgb, sample_times = data["rays"], data["rgbs"], data["time"]
        depth = None

        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])

        if len(all_cameras) != 0:
            camera = all_cameras[t]
            camera_refined = refined_camera[cam_id]
            camera.position = camera_refined.position.detach().clone()
            camera.orientation = camera_refined.orientation.detach().clone()
            pixels = camera.get_pixel_centers_torch().to(device)
            rays_d = camera.pixels_to_rays_torch(pixels, camera.orientation).view([-1,3])
            rays_o = camera.position.unsqueeze(0).expand_as(rays_d)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        with torch.no_grad():
            times = sample_times.view(-1, sample_times.shape[-1])
            rgb_map, _, depth_map, _, _ = OctreeRender_trilinear_fast(
                rays,
                times,
                model,
                chunk=2048,
                N_samples=N_samples,
                ndc_ray=ndc_ray,
                white_bg=white_bg,
                device=device,
            )
            rgb_map = rgb_map.clamp(0.0, 1.0)
            rgb_map, depth_map = (
                rgb_map.reshape(H, W, 3).cpu(),
                depth_map.reshape(H, W).cpu(),
            )

            depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
            if "depths" in data.keys():
                depth = data["depths"]
                gt_depth, _ = visualize_depth_numpy(depth.numpy(), near_far)

            if "mask" in data.keys():
                mask = data["mask"]
            else:
                mask = None

            if len(test_dataset):
                gt_rgb = gt_rgb.view(H, W, 3)
                if mask is None:
                    loss = torch.mean((rgb_map - gt_rgb) ** 2)
                else:
                    eps = 1e-6
                    loss = (((rgb_map-gt_rgb)**2) * mask.permute(1, 2, 0)).sum() / (3*mask.sum().clip(eps))

                PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))
                print(-10.0 * np.log(loss.item()) / np.log(10.0))

                if compute_extra_metrics:
                    if mask is None:
                        ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                    else:
                        ssim = masked_rgb_ssim(rgb_map, gt_rgb, mask, 1)
                    ms_ssim = MS_SSIM(
                        rgb_map.permute(2, 0, 1).unsqueeze(0),
                        gt_rgb.permute(2, 0, 1).unsqueeze(0),
                        data_range=1,
                        size_average=True,
                    )
                    l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "alex", device)
                    l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "vgg", device)
                    ssims.append(ssim)
                    msssims.append(ms_ssim)
                    l_alex.append(l_a)
                    l_vgg.append(l_v)

            rgb_map = (rgb_map.numpy() * 255).astype("uint8")
            gt_rgb_map = (gt_rgb.numpy() * 255).astype("uint8")

            if depth is not None:
                gt_depth_maps.append(gt_depth)
            rgb_maps.append(rgb_map)
            depth_maps.append(depth_map)
            if savePath is not None:
                imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.png", rgb_map)
                imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_gt.png", gt_rgb_map)
                rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
                imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)
                if depth is not None:
                    rgb_map = np.concatenate((gt_rgb_map, gt_depth), axis=1)
                    imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}_gt.png", rgb_map)

    imageio.mimwrite(
        f"{savePath}/{prefix}video.mp4",
        np.stack(rgb_maps),
        fps=30,
        format="FFMPEG",
        quality=10,
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}depthvideo.mp4",
        np.stack(depth_maps),
        format="FFMPEG",
        fps=30,
        quality=10,
    )
    if depth is not None:
        imageio.mimwrite(
            f"{savePath}/{prefix}_gt_depthvideo.mp4",
            np.stack(gt_depth_maps),
            format="FFMPEG",
            fps=30,
            quality=10,
        )

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            msssim = np.mean(np.asarray(msssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(
                    f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}\n"
                )
                print(
                    f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}\n"
                )
                for i in range(len(PSNRs)):
                    f.write(
                        f"Index {i}, PSNR: {PSNRs[i]}, SSIM: {ssims[i]}, MS-SSIM: {msssim}, LPIPS_a: {l_alex[i]}, LPIPS_v: {l_vgg[i]}\n"
                    )
        else:
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(f"PSNR: {psnr} \n")
                print(f"PSNR: {psnr} \n")
                for i in range(len(PSNRs)):
                    f.write(f"Index {i}, PSNR: {PSNRs[i]}\n")

    return PSNRs


@torch.no_grad()
def evaluation_path(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=5,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
):
    """
    Evaluate the model on the valiation rays.
    """
    rgb_maps, depth_maps = [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    val_rays, val_times = test_dataset.get_val_rays()

    for idx in tqdm(range(val_times.shape[0])):
        torch.cuda.empty_cache()
        W, H = test_dataset.img_wh
        rays = val_rays[idx]
        time = val_times[idx]
        time = time.expand(rays.shape[0], 1)
        rgb_map, _, depth_map, _, _ = OctreeRender_trilinear_fast(
            rays,
            time,
            model,
            chunk=2048,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")

        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.png", rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)

    imageio.mimwrite(
        f"{savePath}/{prefix}video.mp4", np.stack(rgb_maps), fps=30, quality=8
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}depthvideo.mp4", np.stack(depth_maps), fps=30, quality=8
    )

    return 0

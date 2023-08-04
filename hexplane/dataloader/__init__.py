from .dnerf_dataset import DNerfDataset
from .neural_3D_dataset_NDC import Neural3D_NDC_Dataset
from .tumrgbd_dataset import TUMRgbdDataset
from .tumrgbd_dataset_slam import TUMRgbdSlamDataset

def get_train_dataset(cfg, is_stack=False, images=[], poses=[], timestamps=[], intrinsics=[]):
    if cfg.data.dataset_name == "dnerf":
        train_dataset = DNerfDataset(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
        )
    elif cfg.data.dataset_name == "neural3D_NDC":
        train_dataset = Neural3D_NDC_Dataset(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.nv3d_ndc_bd_factor,
            eval_step=cfg.data.nv3d_ndc_eval_step,
            eval_index=cfg.data.nv3d_ndc_eval_index,
            sphere_scale=cfg.data.nv3d_ndc_sphere_scale,
        )
        
    elif cfg.data.dataset_name == "tum_rgbd":
        train_dataset = TUMRgbdDataset(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.nv3d_ndc_bd_factor,
            eval_step=cfg.data.nv3d_ndc_eval_step,
            eval_index=cfg.data.nv3d_ndc_eval_index,
            sphere_scale=cfg.data.nv3d_ndc_sphere_scale,
        )

    elif cfg.data.dataset_name == "tum_rgbd_slam":
        train_dataset = TUMRgbdSlamDataset(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.nv3d_ndc_bd_factor,
            eval_step=cfg.data.nv3d_ndc_eval_step,
            eval_index=cfg.data.nv3d_ndc_eval_index,
            sphere_scale=cfg.data.nv3d_ndc_sphere_scale,
            images=images,
            poses=poses,
            timestamps=timestamps,
            intrinsics=intrinsics
        )
    else:
        raise NotImplementedError("No such dataset")
    return train_dataset


def get_test_dataset(cfg, is_stack=True, images=[], poses=[], timestamps=[], intrinsics=[]):
    if cfg.data.dataset_name == "dnerf":
        test_dataset = DNerfDataset(
            cfg.data.datadir,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
        )
    elif cfg.data.dataset_name == "neural3D_NDC":
        test_dataset = Neural3D_NDC_Dataset(
            cfg.data.datadir,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.nv3d_ndc_bd_factor,
            eval_step=cfg.data.nv3d_ndc_eval_step,
            eval_index=cfg.data.nv3d_ndc_eval_index,
            sphere_scale=cfg.data.nv3d_ndc_sphere_scale,
        )
        
    elif cfg.data.dataset_name == "tum_rgbd":
        test_dataset = TUMRgbdDataset(
            cfg.data.datadir,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.nv3d_ndc_bd_factor,
            eval_step=cfg.data.nv3d_ndc_eval_step,
            eval_index=cfg.data.nv3d_ndc_eval_index,
            sphere_scale=cfg.data.nv3d_ndc_sphere_scale,
        )

    elif cfg.data.dataset_name == "tum_rgbd_slam":
        test_dataset = TUMRgbdSlamDataset(
            cfg.data.datadir,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.nv3d_ndc_bd_factor,
            eval_step=cfg.data.nv3d_ndc_eval_step,
            eval_index=cfg.data.nv3d_ndc_eval_index,
            sphere_scale=cfg.data.nv3d_ndc_sphere_scale,
            images=images,
            poses=poses,
            timestamps=timestamps,
            intrinsics=intrinsics
        )
    else:
        raise NotImplementedError("No such dataset")
    return test_dataset

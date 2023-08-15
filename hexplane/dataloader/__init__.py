from .tumrgbd_dataset import TUMRgbdDataset
from .bonnrgbd_dataset import BonngbdDataset
# from .your_own_data import YourOwnDataset
from .droid_slam_data_midas import YourOwnDataset
# from .droid_slam_data import YourOwnDataset
from .iphone_data import iPhoneDataset

def get_train_dataset(cfg, is_stack=False, images=[], depths=[], poses=[], timestamps=[], intrinsics=[]):
    if cfg.data.dataset_name == "bonn_rgbd":
        train_dataset = BonnrgbdDataset(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=cfg.data.N_vis,
        )
    elif cfg.data.dataset_name == "own_data":
        train_dataset = YourOwnDataset(
            cfg.data.datadir,
            cfg.data.scene_bbox_min,
            cfg.data.scene_bbox_max,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=cfg.data.N_vis,
            cal_fine_bbox=cfg.data.cal_fine_bbox
        )
    elif cfg.data.dataset_name == "iphone":
        train_dataset = iPhoneDataset(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=cfg.data.N_vis
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
            depths=depths,
            poses=poses,
            timestamps=timestamps,
            intrinsics=intrinsics
        )
    else:
        raise NotImplementedError("No such dataset")
    return train_dataset


def get_test_dataset(cfg, is_stack=True, images=[], depths=[], poses=[], timestamps=[], intrinsics=[]):
    if cfg.data.dataset_name == "bonn_rgbd":
        test_dataset = BonngbdDataset(
            cfg.data.datadir,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=cfg.data.N_vis,
        )
    elif cfg.data.dataset_name == "own_data":
        test_dataset = YourOwnDataset(
            cfg.data.datadir,
            cfg.data.scene_bbox_min,
            cfg.data.scene_bbox_max,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=cfg.data.N_vis,
            cal_fine_bbox=False
        )
    elif cfg.data.dataset_name == "iphone":
        test_dataset = iPhoneDataset(
            cfg.data.datadir,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=cfg.data.N_vis,
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
            depths=depths,
            poses=poses,
            timestamps=timestamps,
            intrinsics=intrinsics
        )
    else:
        raise NotImplementedError("No such dataset")
    return test_dataset

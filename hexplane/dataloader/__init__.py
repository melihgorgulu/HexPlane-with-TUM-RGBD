from .colmap import ColmapDataset
from .droid_slam_data_midas_bonn import BonnDataset
from .droid_slam_data_midas_iphone import iPhoneSlamDataset

def get_train_dataset(cfg, is_stack=False, images=[], depths=[], poses=[],
                      timestamps=[], intrinsics=[], time_scaling=[]):
    if cfg.data.dataset_name == "colmap":
        train_dataset = ColmapDataset(
            images,
            poses,
            intrinsics,
            cfg.data.scene_bbox_min,
            cfg.data.scene_bbox_max,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=cfg.data.N_vis,
            cal_fine_bbox=cfg.data.cal_fine_bbox
        )
    elif cfg.data.dataset_name == "bonn_slam":
        train_dataset = BonnDataset(
            images,
            poses,
            timestamps,
            intrinsics,
            cfg.data.scene_bbox_min,
            cfg.data.scene_bbox_max,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=cfg.data.N_vis,
            cal_fine_bbox=cfg.data.cal_fine_bbox
        )
    elif cfg.data.dataset_name == "iphone_slam":
        train_dataset = iPhoneSlamDataset(
            images,
            poses,
            timestamps,
            intrinsics,
            cfg.data.scene_bbox_min,
            cfg.data.scene_bbox_max,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=cfg.data.N_vis,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            time_scaling=time_scaling
        )
    else:
        raise NotImplementedError("No such dataset")
    return train_dataset


def get_test_dataset(cfg, is_stack=True, images=[], depths=[], poses=[],
                     timestamps=[], intrinsics=[], time_scaling=[]):
    if cfg.data.dataset_name == "colmap":
        test_dataset = ColmapDataset(
            images,
            poses,
            intrinsics,
            cfg.data.scene_bbox_min,
            cfg.data.scene_bbox_max,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=cfg.data.N_vis,
            cal_fine_bbox=cfg.data.cal_fine_bbox
        )
    elif cfg.data.dataset_name == "bonn_slam":
        test_dataset = BonnDataset(
            images,
            poses,
            timestamps,
            intrinsics,
            cfg.data.scene_bbox_min,
            cfg.data.scene_bbox_max,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=cfg.data.N_vis,
            cal_fine_bbox=cfg.data.cal_fine_bbox
        )
    elif cfg.data.dataset_name == "iphone_slam":
        test_dataset = iPhoneSlamDataset(
            images,
            poses,
            timestamps,
            intrinsics,
            cfg.data.scene_bbox_min,
            cfg.data.scene_bbox_max,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            N_vis=cfg.data.N_vis,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            time_scaling=time_scaling
        )
    else:
        raise NotImplementedError("No such dataset")
    return test_dataset

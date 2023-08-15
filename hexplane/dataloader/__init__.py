from .tumrgbd_dataset import TUMRgbdDataset
# from .your_own_data import YourOwnDataset
from .droid_slam_data_midas import YourOwnDataset
# from .droid_slam_data import YourOwnDataset
from .iphone_data import iPhoneDataset

def get_train_dataset(cfg, is_stack=False):       
    if cfg.data.dataset_name == "tum_rgbd":
        train_dataset = TUMRgbdDataset(
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
    else:
        raise NotImplementedError("No such dataset")
    return train_dataset


def get_test_dataset(cfg, is_stack=True):       
    if cfg.data.dataset_name == "tum_rgbd":
        test_dataset = TUMRgbdDataset(
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
    else:
        raise NotImplementedError("No such dataset")
    return test_dataset

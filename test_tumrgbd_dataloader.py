from hexplane.dataloader.tumrgbd_dataset import TUMRgbdDataset
#from hexplane.dataloader.neural_3D_dataset_NDC import Neural3D_NDC_Dataset

data_dir = "/home/lindedigital/projects/HexPlane/data/rgbd_dataset_freiburg3_sitting_static"
dataset = TUMRgbdDataset(datadir=data_dir)

#data_dir = "/home/lindedigital/projects/HexPlane/data/n3dv/flame_salmon_1"
#dataset = Neural3D_NDC_Dataset(datadir=data_dir)
breakpoint()
i = 0
for cur_data in dataset:
    print(i)
    print(cur_data['time'])
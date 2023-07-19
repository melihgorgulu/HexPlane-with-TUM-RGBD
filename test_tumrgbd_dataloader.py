from hexplane.dataloader.tumrgbd_dataset import TUMRgbdDataset
#from hexplane.dataloader.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
#from hexplane.dataloader.dnerf_dataset import DNerfDataset


data_dir = "/Users/melihgorgulu/Desktop/Projects/Praktikum/HexPlane-with-TUM-RGBD/data/rgbd_dataset_freiburg3_sitting_static"
dataset = TUMRgbdDataset(datadir=data_dir, is_stack=True)

#data_dir = "/home/lindedigital/projects/HexPlane/data/n3dv/flame_salmon_1"
#dataset = Neural3D_NDC_Dataset(datadir=data_dir)

#data_dir = "/home/lindedigital/projects/HexPlane/data/rgbd_dataset_freiburg3_sitting_static"
#dataset = TUMRgbdDataset(datadir=data_dir)

#data_dir = "/home/lindedigital/projects/HexPlane/data/n3dv/flame_salmon_1"
#dataset = Neural3D_NDC_Dataset(datadir=data_dir)
#data_dir = "/home/lindedigital/projects/HexPlane/data/jumpingjacks"
#dataset = DNerfDataset(datadir=data_dir)

i = 0
print(len(dataset))
for cur_data in dataset:
    print(cur_data)
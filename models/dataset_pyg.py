import pandas as pd
import numpy as np
import torch
from pathlib import Path
import torch_geometric
from torch_geometric.data import Data, Batch, DataLoader, InMemoryDataset
from typing import Optional, Callable, List
from transform import Rotation,Transform
from torch_geometric.nn import radius,radius_graph

def read_data(path):
    data = np.load(path)
    return data["vertices"], data['vertice_normals']

class Grasp_Dataset(InMemoryDataset):
    def __init__(self, root: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.pcd_path = Path(root) / "pcd"
        super(Grasp_Dataset,self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> str:
        return 'check the existence of raw/foo/pcd'

    @property
    def processed_file_names(self) -> List[str]:
        return ['training.pt', 'test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        return
    def process(self):
        scene_ids = [f for f in self.pcd_path.iterdir() if f.suffix == ".npz"]
        data_list = []
        pn = 0
        nn=0
        for _i in range(len(scene_ids)):
            v, n = read_data(scene_ids[_i])
            v = torch.from_numpy(v).to(torch.float)
            n = torch.from_numpy(n).to(torch.float)
            data = Data(pos=v,normals=n)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            data_list.append(data)
        print('positive:',pn,'negative',nn)
        total_num = len(data_list)
        torch.save(self.collate(data_list[:int(total_num*0.8)]), self.processed_paths[0])
        torch.save(self.collate(data_list[int(total_num*0.8):]), self.processed_paths[1])

class GraspNormalization:
    def __init__(self):
        self.center = True
        self.scale=  False
    def __call__(self,data:Data):
        pos = data.pos
        center = pos.mean(dim=0,keepdim=True)
        pos = pos - center
        data.pos = pos
        return data
    
class GraspAugmentation:
    def __init__(self):
        self.angle_choices = 6
    def __call__(self,data:Data):
        R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, (np.pi/(self.angle_choices/2))*np.random.choice(self.angle_choices)])
        t_augment = np.r_[0.0, 0.0, 0.0]
        T_augment = Transform(R_augment, t_augment)
        pos = data.pos.numpy()
        normals = data.normals.numpy()
        pos = T_augment.transform_point(pos)
        pos = torch.from_numpy(pos).to(torch.float)
        normals = T_augment.transform_vector(normals)
        normals = torch.from_numpy(normals).to(torch.float)
        data.normals = normals
        data.pos = pos
        return data
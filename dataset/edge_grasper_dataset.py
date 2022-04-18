import pandas as pd
import numpy as np
import torch
from pathlib import Path
import open3d as o3d
import torch_geometric
from torch_geometric.data import Data, Batch, DataLoader, InMemoryDataset
from typing import Optional, Callable, List
from transform import Rotation,Transform
#from vis_grasp import draw_scene
#import mayavi.mlab as mlab
from torch_geometric.nn import radius,radius_graph
import torch.nn.functional as F
from utils import get_geometry_mask
from torch_geometric.transforms import Compose

def read_data(path):
    data = np.load(path)
    return data["vertices"], data['vertice_normals']

class Grasp_Dataset(InMemoryDataset):
    def __init__(self, root: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.pcd_path = Path(root) / "pcd"
        self.cvs_path = Path(root) / "grasps_multi_labels.csv"
        super(Grasp_Dataset,self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> str:
        return 'grasps_multi_labels.csv'

    @property
    def processed_file_names(self) -> List[str]:
        return ['training.pt', 'test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        return
    def process(self):
        scene_ids = [f for f in self.pcd_path.iterdir() if f.suffix == ".npz"]
        data_list = []
        for _i in range(len(scene_ids)):
            v, n = read_data(scene_ids[_i])
            #  To torch.tensor
            v = torch.from_numpy(v).to(torch.float32)
            n = torch.from_numpy(n).to(torch.float32)
            data = Data(pos=v,normals=n)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
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

class EdgeLabel:
    def __init__(self):
        self.sample_num = 32
    def __call__(self, data):
        sample = np.random.choice(len(data.pos), self.sample_num,replace=False)
        sample_pos = data.pos[sample, :]
        sample_normal = data.normals[sample, :]
        radius_p_batch_index = radius(data.pos, sample_pos, r=0.038, max_num_neighbors=1024)
        radius_p_index = radius_p_batch_index[1, :]
        radius_p_batch = radius_p_batch_index[0, :]
        sample = torch.from_numpy(sample)
        sample_node = torch.from_numpy(sample).unsqueeze(dim=-1)
        sample_node = torch.cat([sample_node[i, :].repeat((radius_p_batch == i).sum(),1) for i in range(len(sample))],dim=0)
        sample_pos = torch.cat([sample_pos[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))],dim=0)
        sample_normal = torch.cat(
            [sample_normal[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))], dim=0)
        des_pos = data.pos[radius_p_index, :]
        des_normals = data.normals[radius_p_index, :]
        normals_dot = torch.einsum('ik,ik->i', des_normals, sample_normal).unsqueeze(dim=-1)
        relative_pos = des_pos - sample_pos
        relative_pos_normalized = F.normalize(relative_pos, p=2, dim=1)
        third_axis = torch.cross(relative_pos_normalized, sample_normal, dim=1)
        third_axis = F.normalize(third_axis,p=2,dim=1)
        dot_product_2 = torch.einsum('ik,ik->i', des_normals, third_axis).unsqueeze(dim=-1)
        # label creating
        geometry_mask, depth_projection, orth_mask, angle_mask = get_geometry_mask(normals_dot, dot_product_2,
                                                                                   relative_pos, des_normals,
                                                                                   sample_normal, sample_pos, data.pos,
                                                                                   use_o3d=False, strict=True)
        edges_index = torch.cat((sample_node,radius_p_index.unsqueeze(dim=-1)),dim=-1).T
        data.edge_index = edges_index
        data.edge_mask = geometry_mask
        data.depth_projection = depth_projection
        data.dot1 = normals_dot
        data.dot2 = dot_product_2
        data.sample = sample
        data.radius_p_batch = radius_p_batch
        data.radius_p_index = radius_p_index
        return data

test = True
if test:
    dataset = Grasp_Dataset(root='./raw1/foo',pre_transform = Compose([GraspNormalization(),EdgeLabel()]),train=True)
    loader = DataLoader(dataset,batch_size=1,shuffle=False)
    for batch in loader:
        print(batch)
        print(batch.ptr)
        print(batch.edge_index)
        # if sum(batch.edge_mask):
        #     draw_grasps(batch.edge_mask, batch.depth_projection,
        #                 batch.normals[batch.edge_index[0,:],:], batch.normals[batch.edge_index[1,:],:],
        #                 batch.pos[batch.edge_index[0,:],:], batch.edge_index[1,:],
        #                 np.unique(batch.edge_index[0,:].numpy()),batch.pos, diverse=True, scores=None)
        break
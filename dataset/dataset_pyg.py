import pandas as pd
import numpy as np
import torch
from pathlib import Path
import open3d as o3d
import torch_geometric
from torch_geometric.data import Data, Batch, DataLoader, InMemoryDataset
from typing import Optional, Callable, List
from transform import Rotation,Transform

def read_data(path):
    data = np.load(path)
    return data["vertices"], data['vertice_normals']
def vis_pcd(points,normals, vis=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    if vis:
        o3d.visualization.draw_geometries([pcd])
    return pcd
def vis_samples(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    inlier_cloud.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],)
def vis_samples_2(cloud, pos, neg):
    inlier1_cloud = cloud.select_by_index(pos)

    if len(neg)>0:
        inlier2_cloud = cloud.select_by_index(neg)
        inlier2_cloud.paint_uniform_color([1, 0, 0])
        ind = np.concatenate((pos, neg))
    else:
        ind = pos

    outlier_cloud = cloud.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    inlier1_cloud.paint_uniform_color([0, 1, 0])
    if len(neg)>0:
        o3d.visualization.draw_geometries([inlier1_cloud, inlier2_cloud, outlier_cloud],)
    else:
        o3d.visualization.draw_geometries([inlier1_cloud, outlier_cloud],)

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
        df = pd.read_csv(self.cvs_path)
        scene_ids = [f for f in self.pcd_path.iterdir() if f.suffix == ".npz"]
        data_list = []
        pn = 0
        nn=0
        for _i in range(len(scene_ids)):
            scene_id = str(scene_ids[_i])
            sparsed = scene_id.split('/')
            #sparsed = scene_id.split('\\')
            # print(sparsed)
            scene_id = sparsed[-1][:-4]
            scene_df = df[df["scene_id"] == scene_id]
            scene_df_positive = scene_df[scene_df["label"] == 1]
            scene_df_negative = scene_df[scene_df["label"] == 0]
            #print(scene_id)
            #print('positive number', len(scene_df_positive), '; negative number', len(scene_df_negative))
            if len(scene_df_negative) == 0:
                #print('no negative data: ', _i)
                continue
                pass
                #continue
            if len(scene_df_positive) < 20:
                #print('not enough positive')
                continue
                pass
            # print(scene_ids[i])
            v, n = read_data(scene_ids[_i])
            #pcd = vis_pcd(v, n, vis=False)
            labels = scene_df.loc[:, "label_0":"label_8"].to_numpy()
            total_pt_number = np.arange(0, len(v), 1)
            nums_success = labels.sum(axis=-1)
            num_positive_mask = nums_success >= 4
            num_negative_mask = nums_success < 4
            #print(len(num_positive_mask), num_positive_mask.sum())
            #print(len(num_negative_mask), num_negative_mask.sum())
            scene_nums_positive = scene_df[num_positive_mask]
            scene_nums_negative = scene_df[num_negative_mask]
            #vis_samples_2(pcd, scene_nums_positive['idx_gobal'].to_numpy(),scene_nums_negative['idx_gobal'].to_numpy(), )

            # vis_samples_2(pcd, scene_df_positive['idx_gobal'].to_numpy(),scene_df_negative['idx_gobal'].to_numpy(),)
            positive_mask = scene_nums_positive['idx_gobal'].to_numpy()
            negative_mask = scene_nums_negative['idx_gobal'].to_numpy()
            positive_pitch = scene_nums_positive['pitch_idx'].to_numpy()
            positive_width = scene_nums_positive['width'].to_numpy()
            #positive_normal = n[positive_mask, :]
            orientation_gt = scene_df.loc[:, "qx":"qw"].to_numpy()
            position_gt = scene_df.loc[:, "x":"z"].to_numpy()
            #print(orientation_gt.shape)
            #  To torch.tensor
            v = torch.from_numpy(v).to(torch.float)
            n = torch.from_numpy(n).to(torch.float)
            positive_mask = torch.from_numpy(positive_mask).to(torch.long)
            negative_mask = torch.from_numpy(negative_mask).to(torch.long)
            print('positive number', len(positive_mask), '; negative number', len(negative_mask))
            positive_pitch = torch.from_numpy(positive_pitch).to(torch.long)
            #print(num_positive_mask)
            positive_numbers = torch.as_tensor(len(positive_mask)).to(torch.long)
            negative_numbers = torch.as_tensor(len(negative_mask)).to(torch.long)
            points_numbers = torch.as_tensor(len(v)).to(torch.long)
            label = torch.as_tensor(num_positive_mask).to(torch.long)
            labels = torch.from_numpy(labels).to(torch.long)
            rotations_gt = np.empty((len(orientation_gt), 2, 4), dtype=np.single)
            # process the rotation label
            R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
            for pos_num in range(len(orientation_gt)):
                ori = Rotation.from_quat(orientation_gt[pos_num])
                rotations_gt[pos_num, 0, :] = ori.as_quat()
                rotations_gt[pos_num, 1, :] = (ori * R).as_quat()
            position_gt = torch.from_numpy(position_gt).to(torch.float)
            orientations_gt = torch.from_numpy(rotations_gt).to(torch.float)
            data = Data(pos=v,normals=n,positive_mask=positive_mask,negative_mask=negative_mask,
                        positive_pitch=positive_pitch,position_gt=position_gt,orientation_gt=orientations_gt,
                        labels=labels,label=label,positive_numbers =positive_numbers, negative_numbers=negative_numbers,
                        points_numbers=points_numbers,)
            pn = pn+len(positive_mask)
            nn = nn+len(negative_mask)
            #print(data)
            #break
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
        position_gt = data.position_gt
        center = pos.mean(dim=0,keepdim=True)
        #print('============================',pos.size(),center.size())
        pos = pos - center
        position_gt = position_gt - center
        data.pos = pos
        data.position_gt = position_gt
        return data

dataset = Grasp_Dataset(root='raw/foo',train=True)
print(len(dataset))
loader = DataLoader(dataset,batch_size=2,shuffle=False)
for batch in loader:
    print(batch)
    print(batch.ptr)
    break

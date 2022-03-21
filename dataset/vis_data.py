import pandas as pd
import numpy as np
import torch
from pathlib import Path
import open3d as o3d

def read_data(path):
    data = np.load(path)
    return data["vertices"], data['vertice_normals']
def vis_pcd(points,normals):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
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


pcd_path = Path('./raw/foo') / "pcd"
df = pd.read_csv('./raw/foo/grasps_multi_labels.csv')

#print(df['label'])
scene_ids = [f for f in pcd_path.iterdir() if f.suffix == ".npz"]
data_list = []
#print(len(scene_ids))
print(len(scene_ids))

for _i in range(len(scene_ids)):
    #_i = 86
    #scene_id = str(scene_ids[_i])[11:-4]
    scene_id = str(scene_ids[_i])
    sparsed = scene_id.split('/')
    #print(sparsed)
    scene_id = sparsed[-1][:-4]
    scene_df = df[df["scene_id"] == scene_id]
    scene_df_positive = scene_df[scene_df["label"] == 1]
    scene_df_negative = scene_df[scene_df["label"] == 0]
    print(scene_id)
    print('positive number',len(scene_df_positive),'; negative number',len(scene_df_negative) )
    if len(scene_df_negative) ==0:
        #print('no negative data: ', _i)
        continue
    if len(scene_df_positive) < 20:
        #print('not enough positive')
        continue
    #print(scene_ids[i])
    v, n = read_data(scene_ids[_i])
    pcd = vis_pcd(v,n)
    #vis_samples_2(pcd, scene_df_positive['idx_gobal'].to_numpy(),scene_df_negative['idx_gobal'].to_numpy(),)
    positive_mask = scene_df_positive['idx_gobal'].to_numpy()
    negative_mask = scene_df_negative['idx_gobal'].to_numpy()
    positive_pitch = scene_df_positive['pitch_idx'].to_numpy()
    positive_width = scene_df_positive['width'].to_numpy()
    positive_normal = n[positive_mask, :]
    orientation_gt = scene_df_positive.loc[:,"qx":"qw"].to_numpy()
    position_gt = scene_df_positive.loc[:,"x":"z"].to_numpy()
    labels = scene_df.loc[:,"label_0":"label_8"].to_numpy()
    total_pt_number = np.arange(0,len(v),1)
    v = torch.from_numpy(v).to(torch.float)
    n = torch.from_numpy(n).to(torch.float)
    print(labels)
    nums_success = labels.sum(axis=-1)
    print(nums_success)
    print(len(nums_success))
    num_positive_mask = nums_success>=2
    num_negative_mask = nums_success<2
    print(len(num_positive_mask),num_positive_mask.sum())
    print(len(num_negative_mask),num_negative_mask.sum())
    scene_nums_positive = scene_df[num_positive_mask]
    scene_nums_negative = scene_df[num_negative_mask]
    vis_samples_2(pcd, scene_nums_positive['idx_gobal'].to_numpy(), scene_nums_negative['idx_gobal'].to_numpy(),)
    break

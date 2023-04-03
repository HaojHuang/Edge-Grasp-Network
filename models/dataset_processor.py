import os.path
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
#sys.path.append('..')
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import torch_geometric
from torch_geometric.data import Data, Batch, DataLoader, InMemoryDataset
from typing import Optional, Callable, List
from transform import Rotation,Transform
from torch_geometric.nn import radius,radius_graph,knn
import torch.nn.functional as F


def read_data(path):
    data = np.load(path,allow_pickle=True)
    #print(list(data.keys()))
    #print(data.files)
    pos = data["pos"]
    normals = data["normals"]
    sample = data["sample"]
    radius_p_index = data["radius_p_index"]
    radius_p_batch = data["radius_p_batch"]
    edges = data["edges"]
    approaches = data["approachs"]
    edge_sample_index = data["edge_sample_index"]
    grasp_label = data["grasp_label"]
    depth_projection = data['depth_projection']
    return pos, normals,  sample, radius_p_index, radius_p_batch, grasp_label, edge_sample_index, approaches, edges,depth_projection


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
        #for _i in range(100):
            pos, normals, sample, radius_p_index, radius_p_batch, grasp_label, \
            edge_sample_index, approaches, edges, depth_proj = read_data(scene_ids[_i])

            #  To torch.tensor
            pos = torch.from_numpy(pos).to(torch.float32)
            normals = torch.from_numpy(normals).to(torch.float32)
            sample = torch.from_numpy(sample).to(torch.long)
            radius_p_index = torch.from_numpy(radius_p_index).to(torch.long)
            radius_p_batch = torch.from_numpy(radius_p_batch).to(torch.long)
            edges = torch.from_numpy(edges).to(torch.long)
            #print(approaches)
            approaches = torch.from_numpy(approaches).to(torch.float32)
            grasp_label = torch.from_numpy(grasp_label).to(torch.float32)
            edge_sample_index = torch.from_numpy(edge_sample_index).to(torch.long)
            # sort the egde_sample_index to match the depth and the projection
            edge_sample_index,_ = torch.sort(edge_sample_index)
            #print(edge_sample_index)
            depth_proj = torch.from_numpy(depth_proj).to(torch.float32)

            print('# labels of datapoint {}:'.format(_i), grasp_label.size(0), '# of postive labels', torch.sum(grasp_label).item())
            if sum(grasp_label)<10 or sum(grasp_label) == len(grasp_label):
                print('continue')
                continue

            data = Data(pos=pos,normals=normals,sample=sample,radius_p_index=radius_p_index,
                        radius_p_batch=radius_p_batch,edges=edges,approaches=approaches,
                        grasp_label=grasp_label,edge_sample_index = edge_sample_index,depth_proj=depth_proj)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            data_list.append(data)
            #break
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        total_num = len(data_list)
        torch.save(self.collate(data_list[:int(total_num*0.85)]), self.processed_paths[0])
        torch.save(self.collate(data_list[int(total_num*0.85):]), self.processed_paths[1])


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
        self.angle_choices = 36*2
        self.no_augment = 0.05
    def __call__(self,data:Data):
        # normal_delta = np.pi/np.random.randint(60,70,)
        # the normal dot dosesn't change
        rand = np.random.uniform(0,1)
        if rand>self.no_augment:
            rz = np.pi / 2.0 * np.random.choice(self.angle_choices)
            ry = np.pi / 2.0 * np.random.choice(self.angle_choices)
            rx = np.pi / 2.0 * np.random.choice(self.angle_choices)
            R_augment = Rotation.from_rotvec(np.r_[rx, ry, rz])
        else:
            R_augment = Rotation.from_rotvec(np.r_[0., 0., 0.])
        t_augment = np.r_[0.0, 0.0, 0.0]
        T_augment = Transform(R_augment, t_augment)
        pos = data.pos.numpy()
        normals = data.normals.numpy()
        pos = T_augment.transform_point(pos)
        pos = torch.from_numpy(pos).to(torch.float)
        normals = T_augment.transform_vector(normals)
        normals = torch.from_numpy(normals).to(torch.float)
        # other equivariant features that will change under augmentation
        approaches = data.approaches.numpy()
        approaches = T_augment.transform_vector(approaches)
        approaches = torch.from_numpy(approaches).to(torch.float)
        y_axis = data.y_axis.numpy()
        relative_pos = data.relative_pos.numpy()
        ###
        y_axis = T_augment.transform_vector(y_axis)
        relative_pos = T_augment.transform_vector(relative_pos)
        ###
        y_axis = torch.from_numpy(y_axis).to(torch.float)
        relative_pos = torch.from_numpy(relative_pos).to(torch.float)
        # save the data
        data.normals = normals
        data.pos = pos
        data.approaches = approaches
        data.y_axis = y_axis
        data.relative_pos = relative_pos
        return data


class PreTransformBallBox:
    def __init__(self, max_width=True):
        # self.backup = True
        self.max_width = max_width
        # pass

    def __call__(self, data):
        edges = data.edges
        radius_p_index = data.radius_p_index
        sample_pos = data.pos[edges[:, 0], :]
        des_pos = data.pos[radius_p_index, :]
        des_normals = data.normals[radius_p_index, :]
        relative_pos = des_pos - sample_pos
        approaches = data.approaches
        edge_sample_index = data.edge_sample_index
        relative_pos = relative_pos[edge_sample_index, :]
        depth_proj = data.depth_proj
        # the approach axis (-z), des normal (~y axis), x_axis (perpendicular to the gripper plane)
        x_axis = torch.cross(des_normals[edge_sample_index, :], -approaches, dim=1)
        x_axis = F.normalize(x_axis, p=2, dim=1)
        y_axis = torch.cross(-approaches, x_axis, dim=1)
        y_axis = F.normalize(y_axis, p=2, dim=1)
        # print(torch.abs(y_axis - des_normals[edge_sample_index,:]).sum())
        data.relative_pos = relative_pos
        data.y_axis = y_axis
        data.selected_edges = torch.as_tensor(len(approaches))
        data.edge_num = torch.as_tensor(len(data.edges))

        # add ball
        ball_batch_index = radius(data.pos, data.pos[data.sample, :], r=0.055, max_num_neighbors=len(data.pos))
        ball_index = ball_batch_index[1, :]
        ball_batch = ball_batch_index[0, :]
        sample = data.sample.clone().unsqueeze(dim=1)
        ball_sample_index = torch.cat([sample[i, :].repeat((ball_batch == i).sum(), 1) for i in range(len(sample))],
                                      dim=0)
        ball_edges = torch.cat((ball_sample_index, ball_index.unsqueeze(dim=-1)), dim=1)
        ball_edge_num_cusum = torch.as_tensor([(ball_batch == i).sum() for i in range(len(sample))])
        ball_edge_num_cusum = torch.cumsum(torch.cat((torch.tensor([0]), ball_edge_num_cusum), dim=0), dim=0)
        sampled_edge = edges[edge_sample_index, :]
        reindexes = []
        for i in range(len(edge_sample_index)):
            sample_batch = data.radius_p_batch[edge_sample_index[i]]
            sample_index = sampled_edge[i, 0]
            des_index = sampled_edge[i, 1]
            search_filed = ball_edges[ball_batch == sample_batch, 1]
            out = (search_filed == des_index).to(torch.long)
            assert out.sum() == 1
            reindex = torch.argmax(out) + ball_edge_num_cusum[sample_batch]
            assert sample_index == ball_edges[reindex, 0]
            reindexes.append(reindex)
        reindexes = torch.as_tensor(reindexes)
        data.ball_edges = ball_edges
        data.ball_batch = ball_batch
        data.reindexes = reindexes
        data.ball_edge_num_cusum = ball_edge_num_cusum

        # add box
        p_in_box = []
        b_for_box = []
        reindexes_batch = ball_batch[reindexes]
        ball_reindexes_num_cusum = torch.as_tensor([(reindexes_batch == i).sum() for i in range(len(sample))])
        ball_reindexes_num_cusum = torch.cumsum(torch.cat((torch.tensor([0]), ball_reindexes_num_cusum), dim=0), dim=0)

        for i in range(len(data.sample)):
            reindex_num = reindexes_batch == i
            if reindex_num.sum() < 1:
                continue
            else:
                ball_reindex = reindexes[reindex_num]
                ball_labeled_edges = ball_edges[ball_reindex, :]
                approach_index = ball_labeled_edges[:, 0].reshape(-1)
                contact_index = ball_labeled_edges[:, 1].reshape(-1)
                app = approaches[reindex_num, :].reshape(-1, 3)
                depth = depth_proj[reindex_num]
                contact_pos = data.pos[contact_index, :].reshape(-1, 3)
                contact_normals = data.normals[contact_index, :].reshape(-1, 3)
                app_pos = data.pos[approach_index, :].reshape(-1, 3)
                ball_mask = ball_batch == i
                ball_pos = data.pos[ball_index[ball_mask], :]
                ball_pos = ball_pos.unsqueeze(dim=0).repeat(len(contact_pos), 1, 1)
                rela_pos = contact_pos - app_pos
                half_gripper_width = torch.abs(torch.sum(rela_pos * contact_normals, dim=1))
                width = 2 * (half_gripper_width + 0.013).clip(max=0.075).unsqueeze(dim=1)
                if self.max_width:
                    width[:, :] = 0.075

                bottom_left = app_pos - app * (depth.unsqueeze(dim=1) + 0.01) + contact_normals * width / 2
                up_left = bottom_left + app * 0.065  # gripper height 0.065
                # up_right = up_left - contact_normals * width
                bottom_right = bottom_left - contact_normals * width
                bl_relative = ball_pos - bottom_left.unsqueeze(dim=1)
                br_relative = ball_pos - bottom_right.unsqueeze(dim=1)
                lr_contrain = torch.logical_and(torch.sum(bl_relative * (-contact_normals.unsqueeze(dim=1)), dim=2) > 0,
                                                torch.sum(br_relative * contact_normals.unsqueeze(dim=1), dim=2) > 0)
                ul_relative = ball_pos - up_left.unsqueeze(dim=1)
                ub_constrain = torch.logical_and(torch.sum(ul_relative * (-app.unsqueeze(dim=1)), dim=2) > 0,
                                                 torch.sum(bl_relative * (-app.unsqueeze(dim=1)), dim=2) < 0)
                x_axis = torch.cross(contact_normals, -app, dim=1)
                x_axis = F.normalize(x_axis, p=2, dim=1)
                inside_contrain = torch.einsum('bij,bij->bi', bl_relative,
                                               x_axis.unsqueeze(dim=1).repeat(1, bl_relative.size(1), 1))
                inside_contrain = torch.abs(inside_contrain) < 0.004
                box_constrain = torch.logical_and(ub_constrain, lr_contrain)
                # size = number of labeled grasp x number of points of the ball
                box_constrain = torch.logical_and(box_constrain, inside_contrain)
                assert torch.all(torch.any(box_constrain, dim=1)) == True

                global_box_index = torch.arange(ball_edge_num_cusum[i], ball_edge_num_cusum[i + 1]).reshape(1, -1)
                global_box_index = global_box_index.repeat(len(contact_pos), 1)
                global_box_batch = torch.arange(ball_reindexes_num_cusum[i], ball_reindexes_num_cusum[i + 1]).reshape(
                    -1, 1)
                global_box_batch = global_box_batch.repeat(1, box_constrain.size(1))
                # get the inside point for each grasp pose
                assert global_box_batch.shape == global_box_index.shape == box_constrain.shape
                global_box_index = global_box_index.reshape(-1)[box_constrain.reshape(-1)].to(torch.long)
                # get the corresponding batch
                global_box_batch = global_box_batch.reshape(-1)[box_constrain.reshape(-1)].to(torch.long)
                p_in_box.append(global_box_index)
                b_for_box.append(global_box_batch)

        p_in_box = torch.cat(p_in_box, dim=0)
        b_for_box = torch.cat(b_for_box, dim=0)
        # print('ball transform', edge_sample_index)
        data.p_in_box = p_in_box
        data.b_for_box = b_for_box
        return data

class SubsampleBall:
    def __init__(self,):
        pass
    def __call__(self, data):
        #print(data)
        ball_batch = data.ball_batch
        ball_edges = data.ball_edges
        reindexes = data.reindexes
        sample = data.sample
        approaches = data.approaches
        depth_proj = data.depth_proj
        p_in_box = data.p_in_box
        b_for_box = data.b_for_box
        label = data.grasp_label
        relative_pos = data.relative_pos
        y_axis = data.y_axis
        # reduce the batch size tp sample number
        if len(sample) < 3:
            new_sample_index = torch.arange(0, len(sample)).to(torch.long)
            rear = False
            break_points = len(sample)
        else:
            if np.random.uniform() < 0.5:
                new_sample_index = len(sample) // 2
                new_sample_index = torch.arange(0, new_sample_index).to(torch.long)
                rear = False
                break_points = len(sample) // 2
            else:
                rear = True
                new_sample_index = torch.arange(len(sample) // 2, len(sample)).to(torch.long)
                break_points = len(sample) // 2

        new_sample = sample[new_sample_index]
        # print('new sample index', new_sample_index, len(sample))
        if not rear:
            ball_break_point = torch.as_tensor([(ball_batch == i).sum() for i in range(break_points)]).sum()
            ball_batch = ball_batch[:ball_break_point]
            ball_edges = ball_edges[:ball_break_point, :]
            reindexes_mask = reindexes < ball_break_point
            reindexes = reindexes[reindexes_mask]
            approaches = approaches[reindexes_mask, :]
            depth_proj = depth_proj[reindexes_mask]
            label = label[reindexes_mask]
            y_axis = y_axis[reindexes_mask, :]
            relative_pos = relative_pos[reindexes_mask, :]

            box_mask = b_for_box < sum(reindexes_mask)
            b_for_box = b_for_box[box_mask]
            p_in_box = p_in_box[box_mask]
        else:
            ball_break_point_font = torch.as_tensor([(ball_batch == i).sum() for i in range(break_points)]).sum()
            ball_break_point_rear = torch.as_tensor(
                [(ball_batch == i + break_points).sum() for i in range(len(sample) - break_points)]).sum()
            assert ball_break_point_font + ball_break_point_rear == len(ball_batch)
            ball_batch = ball_batch[ball_break_point_font:]
            ball_batch = ball_batch - break_points
            #print('ball batch', ball_batch)

            ball_edges = ball_edges[ball_break_point_font:, :]
            reindexes_mask_font = reindexes < ball_break_point_font
            reindexes_mask = reindexes >= ball_break_point_font

            reindexes = reindexes[reindexes_mask]
            reindexes = reindexes - ball_break_point_font
            approaches = approaches[reindexes_mask, :]
            depth_proj = depth_proj[reindexes_mask]
            label = label[reindexes_mask]
            y_axis = y_axis[reindexes_mask, :]
            relative_pos = relative_pos[reindexes_mask, :]
            # box
            box_mask = b_for_box >= sum(reindexes_mask_font)
            b_for_box = b_for_box[box_mask]
            p_in_box = p_in_box[box_mask]
            b_for_box = b_for_box - sum(reindexes_mask_font)
            p_in_box = p_in_box - ball_break_point_font
            #print('b_for_box', b_for_box)

        data.ball_batch = ball_batch
        data.ball_edges = ball_edges
        data.reindexes = reindexes
        data.sample = new_sample
        data.approaches = approaches
        data.depth_proj = depth_proj
        data.p_in_box = p_in_box
        data.b_for_box = b_for_box
        data.grasp_label = label
        data.relative_pos = relative_pos
        data.y_axis = y_axis
        return data

class PreTransform:
    def __init__(self):
        #self.backup = True
        self.gripper_half_width = 0.035

    def __call__(self,data):
        edges = data.edges
        radius_p_index = data.radius_p_index
        sample_pos = data.pos[edges[:, 0], :]
        des_pos = data.pos[radius_p_index, :]
        des_normals = data.normals[radius_p_index, :]
        relative_pos = des_pos - sample_pos
        approaches = data.approaches
        edge_sample_index = data.edge_sample_index
        relative_pos = relative_pos[edge_sample_index, :]
        depth_proj = data.depth_proj
        # the approach axis (-z), des normal (~y axis), x_axis (perpendicular to the gripper plane)
        x_axis = torch.cross(des_normals[edge_sample_index, :], -approaches, dim=1)
        x_axis = F.normalize(x_axis, p=2, dim=1)
        y_axis = torch.cross(-approaches, x_axis, dim=1)
        y_axis = F.normalize(y_axis, p=2, dim=1)
        #print(torch.abs(y_axis - des_normals[edge_sample_index,:]).sum())
        # other side point
        # replace 0.035 with halfbaseline later
        side_points_1 = -y_axis * 0.035 + sample_pos[edge_sample_index, :]
        side_batch_index = knn(data.pos, side_points_1, k=32)
        # additional other side point
        side_points_1_end = -y_axis * 0.035 + (sample_pos[edge_sample_index, :] - torch.abs(depth_proj.unsqueeze(dim=-1).repeat(1, 3)) * approaches)
        side_end_batch_index = knn(data.pos, side_points_1_end, k=32)
        # store the data
        # will change under transformation
        data.y_axis = y_axis
        data.side_points_1 = side_points_1
        data.side_points_1_end = side_points_1_end
        # won't change under transformation
        data.depth_proj = depth_proj
        data.relative_pos = relative_pos
        data.side_batch_index = side_batch_index
        data.side_end_batch_index = side_end_batch_index
        data.selected_edges = torch.as_tensor(len(approaches))
        data.edge_num = torch.as_tensor(len(data.edges))
        return data

class PreTransformBall:
    def __init__(self):
        # self.backup = True
        pass

    def __call__(self, data):
        edges = data.edges
        radius_p_index = data.radius_p_index
        sample_pos = data.pos[edges[:, 0], :]
        des_pos = data.pos[radius_p_index, :]
        des_normals = data.normals[radius_p_index, :]
        relative_pos = des_pos - sample_pos
        approaches = data.approaches
        edge_sample_index = data.edge_sample_index
        relative_pos = relative_pos[edge_sample_index, :]
        depth_proj = data.depth_proj
        # the approach axis (-z), des normal (~y axis), x_axis (perpendicular to the gripper plane)
        x_axis = torch.cross(des_normals[edge_sample_index, :], -approaches, dim=1)
        x_axis = F.normalize(x_axis, p=2, dim=1)
        y_axis = torch.cross(-approaches, x_axis, dim=1)
        y_axis = F.normalize(y_axis, p=2, dim=1)
        # print(torch.abs(y_axis - des_normals[edge_sample_index,:]).sum())
        # other side point
        # replace 0.035 with halfbaseline later
        side_points_1 = -y_axis * 0.035 + sample_pos[edge_sample_index, :]
        side_batch_index = knn(data.pos, side_points_1, k=32)
        # additional other side point
        side_points_1_end = -y_axis * 0.035 + (sample_pos[edge_sample_index, :] - torch.abs(
            depth_proj.unsqueeze(dim=-1).repeat(1, 3)) * approaches)
        side_end_batch_index = knn(data.pos, side_points_1_end, k=32)
        # store the data
        # will change under transformation
        data.y_axis = y_axis
        data.side_points_1 = side_points_1
        data.side_points_1_end = side_points_1_end
        # won't change under transformation
        data.depth_proj = depth_proj
        data.relative_pos = relative_pos
        data.side_batch_index = side_batch_index
        data.side_end_batch_index = side_end_batch_index
        data.selected_edges = torch.as_tensor(len(approaches))
        data.edge_num = torch.as_tensor(len(data.edges))

        # add ball
        ball_batch_index = radius(data.pos, data.pos[data.sample, :], r=0.05, max_num_neighbors=len(data.pos))
        ball_index = ball_batch_index[1, :]
        ball_batch = ball_batch_index[0, :]
        sample = data.sample.clone().unsqueeze(dim=1)
        ball_sample_index = torch.cat([sample[i, :].repeat((ball_batch == i).sum(), 1) for i in range(len(sample))],
                                      dim=0)
        ball_edges = torch.cat((ball_sample_index, ball_index.unsqueeze(dim=-1)), dim=1)
        ball_edge_num_cusum = torch.as_tensor([(ball_batch == i).sum() for i in range(len(sample))])
        ball_edge_num_cusum = torch.cumsum(torch.cat((torch.tensor([0]), ball_edge_num_cusum), dim=0), dim=0)
        sampled_edge = edges[edge_sample_index, :]
        reindexes = []
        for i in range(len(edge_sample_index)):
            sample_batch = data.radius_p_batch[edge_sample_index[i]]
            sample_index = sampled_edge[i, 0]
            des_index = sampled_edge[i, 1]
            search_filed = ball_edges[ball_batch == sample_batch, 1]
            out = (search_filed == des_index).to(torch.long)
            assert out.sum() == 1
            reindex = torch.argmax(out) + ball_edge_num_cusum[sample_batch]
            assert sample_index == ball_edges[reindex, 0]
            reindexes.append(reindex)
        reindexes = torch.as_tensor(reindexes)
        data.ball_edges = ball_edges
        data.ball_batch = ball_batch
        data.reindexes = reindexes
        data.ball_edge_num_cusum = ball_edge_num_cusum
        return data

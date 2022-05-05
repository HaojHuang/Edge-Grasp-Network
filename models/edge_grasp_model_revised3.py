from dataset_pyg import Grasp_Dataset,GraspNormalization
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import torch_geometric
from torch_geometric.data import Data, Batch, DataLoader, InMemoryDataset
from typing import Optional, Callable, List
from transform import Rotation,Transform
from torch_geometric.nn import radius,radius_graph
import torch.nn.functional as F
import time
from utils import get_geometry_mask2
from torch_geometric.nn import  PPFConv,knn_graph,global_max_pool
from torch_geometric.nn import  PointConv as PointNetConv
from torch.nn import Sequential, Linear, ReLU
import torch
from sklearn.metrics import accuracy_score,balanced_accuracy_score


class Classifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=(512,256,128)):
        super().__init__()
        self.head =  Sequential(Linear(in_channels, hidden_channels[0]),
                                ReLU(),
                                Linear(hidden_channels[0], hidden_channels[1]),
                                ReLU(),
                                Linear(hidden_channels[1], hidden_channels[2]),
                                ReLU(),
                                Linear(hidden_channels[2], 1),)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.head(x)
        return self.sigmoid(x)


class PointNet(torch.nn.Module):
    def __init__(self, out_channels=(32,64,128), train_with_norm=False):
        super().__init__()
        torch.manual_seed(12345)
        if train_with_norm:
            in_channels = 6
        else:
            in_channels = 3
        #out_channels = out_channels
        self.train_with_normal = train_with_norm
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        mlp1 = Sequential(Linear(in_channels + 3, out_channels[0]),
                          ReLU(),
                          Linear(out_channels[0], out_channels[0]))
        self.conv1 = PointNetConv(local_nn=mlp1)

        mlp2 = Sequential(Linear(out_channels[0] + 3, out_channels[1]),
                          ReLU(),
                          Linear(out_channels[1], out_channels[1]))
        self.conv2 = PointNetConv(local_nn=mlp2)

        mlp3 = Sequential(Linear(out_channels[1] + 3, out_channels[2]),
                          ReLU(),
                          Linear(out_channels[2], out_channels[2]))
        self.conv3 = PointNetConv(local_nn=mlp3)

    def forward(self, pos, batch=None, normal=None):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        if self.train_with_normal:
            assert normal is not None
            h = torch.cat((pos, normal), dim=-1)
        else:
            h = pos
        edge_index = knn_graph(pos, k=6, batch=batch, loop=True)
        # 3. Start bipartite message passing.
        h = self.conv1(x=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(x=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv3(x=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        # # 5. Classifier.
        return h

class EdgeGrasp:
    def __init__(self,device,sample_num=32,position_emd=True,lr=1e-4):
        self.device = device
        self.sample_num = sample_num
        self.local_emd_model = PointNet(out_channels=(32,64,128),train_with_norm=True).to(device)
        self.global_emd_model = Sequential(Linear(128, 256),
                                           ReLU(),
                                           Linear(256, 256),
                                           ReLU(),
                                           Linear(256, 256),
                                           ).to(device)
        self.position_emd = position_emd
        if self.position_emd:
            addition_dim = 20+3+7
        else:
            addition_dim = 14+3+7
        self.classifier = Classifier(in_channels=256+256+addition_dim,hidden_channels=(512,256,128)).to(device)
        self.classifier_collision = Classifier(in_channels=256 + 256 + addition_dim, hidden_channels=(512, 256, 128)).to(device)
        #self.classifier_collision = Classifier(in_channels=256 + 256 + addition_dim, hidden_channels=(512, 256, 128)).to(device)
        self.parameter = list(self.local_emd_model.parameters()) + list(self.global_emd_model.parameters())\
                         + list(self.classifier.parameters()) + list(self.classifier_collision.parameters())
        self.optim = torch.optim.Adam(self.parameter, lr=lr, weight_decay=1e-6)

    def forward(self,batch,train=True,):
        #Todo get the local emd for every point in the batch
        # get local_emd
        if train:
            self.local_emd_model.train()
            features = self.local_emd_model(pos=batch.pos,normal= batch.normals)
        else:
            self.local_emd_model.eval()
            with torch.no_grad():
                features = self.local_emd_model(pos=batch.pos, normal=batch.normals)
        #print(features.size())
        #features = torch.rand(len(batch.pos), 128)
        sample = np.random.choice(len(batch.pos), self.sample_num, replace=False)
        #sample = np.asarray([59]) # 149 (the magic of dot2), 247, 664(collision), 13(small shape lost),717 (plausible candidates)
        sample_pos = batch.pos[sample, :]
        sample_normal = batch.normals[sample, :]
        sample_emd = features[sample, :]
        radius_p_batch_index = radius(batch.pos, sample_pos, r=0.04, max_num_neighbors=1024)
        radius_p_index = radius_p_batch_index[1, :]
        radius_p_batch = radius_p_batch_index[0, :]
        sample_pos = torch.cat([sample_pos[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))],
                               dim=0)
        sample_normal = torch.cat(
            [sample_normal[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))], dim=0)
        sample_emd = torch.cat([sample_emd[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))],
                               dim=0)
        des_pos = batch.pos[radius_p_index, :]
        des_normals = batch.normals[radius_p_index, :]
        des_emd = features[radius_p_index, :]
        normals_dot = torch.einsum('ik,ik->i', des_normals, sample_normal).unsqueeze(dim=-1)
        relative_pos = des_pos - sample_pos
        relative_pos_normalized = F.normalize(relative_pos, p=2, dim=1)
        third_axis = torch.cross(relative_pos_normalized, sample_normal, dim=1)
        dot_product_2 = torch.einsum('ik,ik->i', des_normals, third_axis).unsqueeze(dim=-1) #impressive
        depth_projection = -torch.sum(relative_pos * sample_normal, dim=-1)
        center_dis_from_source = (0.105 - 0.059 - depth_projection).unsqueeze(dim=-1)
        side_points_1 = -des_normals * 0.04 + sample_pos
        gripper_center = center_dis_from_source.repeat(1, 3) * sample_normal + sample_pos
        side_points_1_end = -des_normals * 0.04 + (gripper_center - 0.04627 * sample_normal)
        #side_points_2 = des_normals * 0.04 + sample_pos
        side_points = torch.cat((side_points_1,side_points_1_end),dim=-1)

        # label creating
        geometry_mask, _, orth_mask, angle_mask, no_collision_mask, \
        pitch_constrain_mask = get_geometry_mask2(normals_dot, dot_product_2, relative_pos, des_normals,
                                                   sample_normal, sample_pos, batch.pos, strict=True)
        if train:
            self.global_emd_model.train()
            global_emd = self.global_emd_model(des_emd)
        else:
            self.global_emd_model.eval()
            with torch.no_grad():
                global_emd = self.global_emd_model(des_emd)

        global_emd = global_max_pool(global_emd,radius_p_batch)
        global_emd = torch.cat([global_emd[i,:].repeat((radius_p_batch==i).sum(),1) for i in range(len(global_emd))],dim=0)
        #print('global emd',global_emd.size())
        #print('global emd',global_emd.size())
        if self.position_emd:
            sample_emd = torch.cat((sample_emd, torch.zeros(len(des_pos), 1,device=self.device)), dim=-1)
            sample_pos = torch.cat((sample_pos, torch.zeros(len(des_pos), 1,device=self.device)), dim=-1)
            sample_normal = torch.cat((sample_normal, torch.zeros(len(des_pos), 1,device=self.device)), dim=-1)
            des_emd = torch.cat((des_emd, torch.ones(len(des_pos), 1,device=self.device)), dim=-1)
            des_pos = torch.cat((des_pos, torch.ones(len(des_pos), 1,device=self.device)), dim=-1)
            des_normals = torch.cat((des_normals, torch.ones(len(des_pos,), 1,device=self.device)), dim=-1)
            # print(des_emd.shape)
        sample_cat_all = torch.cat((sample_pos, sample_normal, sample_emd), dim=-1)
        des_cat_all = torch.cat((des_pos, des_normals, des_emd, dot_product_2, normals_dot), dim=-1)
        cat_all = torch.cat((sample_cat_all, des_cat_all), dim=-1)
        # should also add the relative pos here in order to infer constrain 2
        cat_all = torch.cat((cat_all,global_emd,relative_pos,depth_projection.unsqueeze(dim=-1),side_points),dim=-1)
        if train:
            self.classifier.train()
            self.classifier_collision.train()
            scores = self.classifier(cat_all)
            scores_colli = self.classifier_collision(cat_all)
        else:
            self.classifier.eval()
            self.classifier_collision.eval()
            with torch.no_grad():
                scores = self.classifier(cat_all)
                scores_colli = self.classifier_collision(cat_all)
        labels = geometry_mask.to(torch.float).unsqueeze(dim=-1).to(scores.device)
        labels_collision = no_collision_mask.to(torch.float).unsqueeze(dim=-1).to(scores.device)
        labels_angle = angle_mask.to(torch.float).unsqueeze(dim=-1).to(scores.device)
        #labels_a = angle_mask.to(torch.float).unsqueeze(dim=-1).to(scores.device)
        #print(scores.size())
        #print(labels.size())
        # orthogonality + noncollision
        return scores, labels, scores_colli, labels_collision, labels_angle, depth_projection, \
               sample, radius_p_index, sample_pos[:,:3],sample_normal[:,:3]

    def train(self,batch,):
        scores, labels, scores_2, labels_2, labels_angle, _,_,_,_,_ = self.forward(batch,train=True)
        if sum(labels)==0 or sum(labels_2)==0:
            pass
        else:
            labels_or = labels_angle
            prediction = scores>0.5
            prediction = prediction.to(torch.float).cpu().numpy()
            accuracy = accuracy_score(labels_or.cpu().numpy(),prediction)
            #print(1)
            balanced_accuracy = balanced_accuracy_score(labels_or.cpu().numpy(),prediction)
            #print(2)
            weights = torch.ones_like(labels_or).to(labels.device)
            weights[labels_or==1.] = (len(labels_or)-sum(labels_or))/(sum(labels_or))
            #print(scores.size(),labels.size(),weights.size())
            loss = F.binary_cross_entropy(scores,labels_or,weight=weights)
            # prediction collision on angle edge(satisfy constrain 1,2,3 but not collision)
            #inverse_lable_2 = ~labels_2
            labels_2 = labels[labels_angle.bool()]
            #print(scores_2.size(), labels_angle.size())
            scores_2 = scores_2[labels_angle.bool()]
            prediction_2 = scores_2 > 0.5
            prediction_2 = prediction_2.to(torch.float).cpu().numpy()
            accuracy_2 = accuracy_score(labels_2.cpu().numpy(), prediction_2)
            # print(1)
            balanced_accuracy_2 = balanced_accuracy_score(labels_2.cpu().numpy(), prediction_2)
            # print(2)

            # many orthogonal edge are non-collision, collision label (0) should have large weights
            weights = torch.ones_like(labels_2).to(scores_2.device)
            if len(labels_2)!=sum(labels_2):
                weights[labels_2 == 0.] = (sum(labels_2))/(len(labels_2)-sum(labels_2))
            else:
                weights[labels_2 == 0.] = (sum(labels_2)) / (len(labels_2) - sum(labels_2) + 1)
            # print(scores.size(),labels.size(),weights.size())
            loss_2 = F.binary_cross_entropy(scores_2, labels_2, weight=weights)
            loss_total = loss + loss_2
            self.optim.zero_grad()
            loss_total.backward()
            self.optim.step()
            #print('Training:','Loss', loss.item(), 'ratio', (labels.sum()/len(labels)).item(), 'Acc', accuracy, 'Bacc', balanced_accuracy)
            return np.float32(loss_total.item()), np.float32(loss.item()), accuracy,balanced_accuracy, np.float32(loss_2.item()), accuracy_2, balanced_accuracy_2
            #Todo from sclearn import the balanced accuracy score also search for the unbalanced weights in sklearn

    def test(self,batch):
        scores,labels,scores_2, labels_2,labels_angle, _,_,_,_,_ = self.forward(batch,train=False)
        if sum(labels) == 0 or sum(labels_2) == 0:
            return None
        else:
            labels_or = labels_angle
            prediction = scores > 0.5
            prediction = prediction.to(torch.float).cpu().numpy()
            accuracy = accuracy_score(labels_or.cpu().numpy(), prediction)
            # print(1)
            balanced_accuracy = balanced_accuracy_score(labels_or.cpu().numpy(), prediction)
            # print(2)
            weights = torch.ones_like(labels_or).to(labels.device)
            # many neighbor are not orthogonal, thus orthogonal neighbor should have large weights
            weights[labels_or == 1.] = (len(labels_or) - sum(labels_or)) / (sum(labels_or))
            # print(scores.size(),labels.size(),weights.size())
            loss = F.binary_cross_entropy(scores, labels_or, weight=weights)
            # prediction collision on angle edge(satisfy constrain 1,2,3 but not collision)
            # inverse_lable_2 = ~labels_2
            labels_2 = labels[labels_angle.bool()]
            # print(scores_2.size(), labels_angle.size())
            scores_2 = scores_2[labels_angle.bool()]
            prediction_2 = scores_2 > 0.5
            prediction_2 = prediction_2.to(torch.float).cpu().numpy()
            accuracy_2 = accuracy_score(labels_2.cpu().numpy(), prediction_2)
            # print(1)
            balanced_accuracy_2 = balanced_accuracy_score(labels_2.cpu().numpy(), prediction_2)
            # print(2)

            # many orthogonal edge are non-collision, collision label (0) should have large weights
            weights = torch.ones_like(labels_2).to(scores_2.device)
            if len(labels_2) != sum(labels_2):
                weights[labels_2 == 0.] = (sum(labels_2)) / (len(labels_2) - sum(labels_2))
            else:
                weights[labels_2 == 0.] = (sum(labels_2)) / (len(labels_2) - sum(labels_2) + 1)
            # print(scores.size(),labels.size(),weights.size())
            loss_2 = F.binary_cross_entropy(scores_2, labels_2, weight=weights)
            loss_total = loss + loss_2
            # print('Training:','Loss', loss.item(), 'ratio', (labels.sum()/len(labels)).item(), 'Acc', accuracy, 'Bacc', balanced_accuracy)
            return np.float32(loss_total.item()), np.float32(loss.item()), accuracy, balanced_accuracy, np.float32(
                loss_2.item()), accuracy_2, balanced_accuracy_2

    def save(self,filename1,filename2,filename3,filename4):
        self.local_emd_model.eval()
        self.global_emd_model.eval()
        self.classifier.eval()
        self.classifier_collision.eval()
        torch.save(self.local_emd_model.state_dict(),filename1)
        torch.save(self.global_emd_model.state_dict(), filename2)
        torch.save(self.classifier.state_dict(), filename3)
        torch.save(self.classifier_collision.state_dict(),filename4)

    def load(self,path1,path2,path3,path4):
        self.local_emd_model.eval()
        self.global_emd_model.eval()
        self.classifier.eval()
        self.classifier_collision.eval()
        self.local_emd_model.load_state_dict(torch.load(path1,self.device))
        self.global_emd_model.load_state_dict(torch.load(path2, self.device))
        self.classifier.load_state_dict(torch.load(path3, self.device))
        self.classifier_collision.load_state_dict(torch.load(path4, self.device))


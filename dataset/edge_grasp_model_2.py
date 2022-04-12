import numpy as np
from torch_geometric.nn import radius,radius_graph
import torch.nn.functional as F
import time
from utils import get_geometry_mask
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
        edge_index = knn_graph(pos, k=16, batch=batch, loop=True)
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
            addition_dim = 18
        else:
            addition_dim = 12
        self.classifier = Classifier(in_channels=256+256+addition_dim,hidden_channels=(512,256,128)).to(device)
        self.parameter = list(self.local_emd_model.parameters()) + list(self.global_emd_model.parameters())\
                         + list(self.classifier.parameters())
        self.optim = torch.optim.Adam(self.parameter, lr=lr, weight_decay=1e-8)

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
        sample = np.random.randint(0, len(batch.pos), self.sample_num)
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
        third_axis = F.normalize(third_axis, p=2, dim=1)
        dot_product_2 = torch.einsum('ik,ik->i', des_normals, third_axis).unsqueeze(dim=-1)
        # label creating
        geometry_mask, depth_projection, orth_mask, angle_mask = get_geometry_mask(normals_dot, dot_product_2,
                                                                                   relative_pos, des_normals,
                                                                                   sample_normal, sample_pos, batch.pos,
                                                                                   strict=True)
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
            des_emd = torch.cat((des_emd, torch.zeros(len(des_pos), 1,device=self.device)), dim=-1)
            des_pos = torch.cat((des_pos, torch.zeros(len(des_pos), 1,device=self.device)), dim=-1)
            des_normals = torch.cat((des_normals, torch.zeros(len(des_pos,), 1,device=self.device)), dim=-1)
            # print(des_emd.shape)
        sample_cat_all = torch.cat((sample_pos, sample_normal, sample_emd), dim=-1)
        des_cat_all = torch.cat((des_pos, des_normals, des_emd,), dim=-1)
        cat_all = torch.cat((sample_cat_all, des_cat_all), dim=-1)
        cat_all = torch.cat((cat_all,global_emd),dim=-1)
        if train:
            self.classifier.train()
            scores = self.classifier(cat_all)
        else:
            self.classifier.eval()
            with torch.no_grad():
                scores = self.classifier(cat_all)
        labels = geometry_mask.to(torch.float).unsqueeze(dim=-1).to(scores.device)
        #print(scores.size())
        #print(labels.size())
        return scores, labels, depth_projection, sample, radius_p_index,sample_pos[:,:3],sample_normal[:,:3]

    def train(self,batch,):
        scores,labels,_,_,_,_,_ = self.forward(batch,train=True)
        if sum(labels)==0:
            pass
        else:
            prediction = scores>0.5
            prediction = prediction.to(torch.float).cpu().numpy()
            accuracy = accuracy_score(labels.cpu().numpy(),prediction)
            #print(1)
            balanced_accuracy = balanced_accuracy_score(labels.cpu().numpy(),prediction)
            #print(2)
            weights = torch.ones_like(labels).to(labels.device)
            weights[labels==1] = len(labels)/sum(labels)
            #print(scores.size(),labels.size(),weights.size())
            loss = F.binary_cross_entropy(scores,labels,weight=weights)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            #print('Training:','Loss', loss.item(), 'ratio', (labels.sum()/len(labels)).item(), 'Acc', accuracy, 'Bacc', balanced_accuracy)
            return np.float32(loss.item()),accuracy,balanced_accuracy
            #Todo from sclearn import the balanced accuracy score also search for the unbalanced weights in sklearn

    def test(self,batch):
        scores,labels,_,_,_,_,_ = self.forward(batch,train=False)
        prediction = scores>0.5
        prediction = prediction.to(torch.float).cpu().numpy()
        accuracy = accuracy_score(labels.cpu().numpy(),prediction)
        balanced_accuracy = balanced_accuracy_score(labels.cpu().numpy(),prediction)
        weights = torch.ones_like(labels).to(labels.device)
        weights[labels==1] = len(labels)/sum(labels)
        #print(scores.size(),labels.size(),weights.size())
        loss = F.binary_cross_entropy(scores,labels,weight=weights)
        #print('Testing:','Loss', loss.item(), 'ratio', (labels.sum()/len(labels)).item(), 'Acc', accuracy, 'Bacc', balanced_accuracy)
        return np.float32(loss.item()),accuracy,balanced_accuracy

    def save(self,filename1,filename2,filename3):
        self.local_emd_model.eval()
        self.global_emd_model.eval()
        self.classifier.eval()
        torch.save(self.local_emd_model.state_dict(),filename1)
        torch.save(self.global_emd_model.state_dict(), filename2)
        torch.save(self.classifier.state_dict(), filename3)

    def load(self,path1,path2,path3):
        self.local_emd_model.eval()
        self.global_emd_model.eval()
        self.classifier.eval()
        self.local_emd_model.load_state_dict(torch.load(path1,self.device))
        self.global_emd_model.load_state_dict(torch.load(path2, self.device))
        self.classifier.load_state_dict(torch.load(path3, self.device))



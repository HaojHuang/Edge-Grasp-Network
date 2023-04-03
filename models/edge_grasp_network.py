import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import  PPFConv,knn_graph,global_max_pool
from torch_geometric.nn import PointConv as PointNetConv
from torch.nn import Sequential, Linear, ReLU
import torch
from sklearn.metrics import accuracy_score,balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")

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
        #self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.head(x)
        return x


class PointNetSimple(torch.nn.Module):
    def __init__(self, out_channels=(64,64,128), train_with_norm=True):
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
        edge_index = knn_graph(pos, k = 16, batch=batch, loop=True)
        # 3. Start bipartite message passing.
        h1 = self.conv1(x=h, pos=pos, edge_index=edge_index)
        h1 = h1.relu()
        h2 = self.conv2(x=h1, pos=pos, edge_index=edge_index)
        #print('h', h.size())
        h2 = h2.relu()
        h3 = self.conv3(x=h2, pos=pos, edge_index=edge_index)
        h3 = h3.relu()
        # # 5. Classifier.
        return h1, h2, h3

class GlobalEmdModel(torch.nn.Module):
    def __init__(self,input_c = 128, inter_c=(256,512,512), output_c=512):
        super().__init__()
        self.mlp1 = Sequential(Linear(input_c, inter_c[0]), ReLU(), Linear(inter_c[0], inter_c[1]), ReLU(), Linear(inter_c[1], inter_c[2]),)
        self.mlp2 = Sequential(Linear(input_c+inter_c[2], output_c), ReLU(), Linear(output_c, output_c))
    def forward(self,pos_emd,radius_p_batch):
        global_emd = self.mlp1(pos_emd)
        global_emd = global_max_pool(global_emd, radius_p_batch)
        global_emd = torch.cat([global_emd[i,:].repeat((radius_p_batch==i).sum(),1) for i in range(len(global_emd))],dim=0)
        global_emd = torch.cat((pos_emd,global_emd),dim=-1)
        global_emd = self.mlp2(global_emd)
        global_emd = global_max_pool(global_emd, radius_p_batch)
        return global_emd


class EdgeGrasp:
    def __init__(self, device, sample_num=32, lr=1e-4):
        self.device = device
        self.sample_num = sample_num
        self.local_emd_model = PointNetSimple(out_channels=(32, 64, 128), train_with_norm=True).to(device)
        self.global_emd_model = GlobalEmdModel(input_c=32+64+128, inter_c=(256,512,512),output_c=1024).to(device)
        self.classifier_fail = Classifier(in_channels=1162, hidden_channels=(512, 256, 128)).to(device)
        self.parameter = list(self.local_emd_model.parameters()) + list(self.global_emd_model.parameters()) \
                         + list(self.classifier_fail.parameters())
        self.classifier_para = list(self.global_emd_model.parameters()) + list(self.classifier_fail.parameters())
        self.optim = torch.optim.Adam([{'params': self.local_emd_model.parameters(), 'lr': lr},
                                       {'params': self.classifier_para}, ], lr=lr, weight_decay=1e-8)
        print('edge_grasper ball: ', sum(p.numel() for p in self.parameter if p.requires_grad))

    def forward(self, batch, train=True,):
        # Todo get the local emd for every point in the batch
        # balls setup
        ball_batch = batch.ball_batch
        ball_edges = batch.ball_edges
        reindexes = batch.reindexes
        balls = batch.pos[ball_edges[:, 1], :] - batch.pos[ball_edges[:, 0], :]
        ball_normals = batch.normals[ball_edges[:, 1], :]
        sample = batch.sample

        if train:
            self.local_emd_model.train()
            f1, f2, features = self.local_emd_model(pos=balls, normal=ball_normals, batch=ball_batch)
        else:
            self.local_emd_model.eval()
            with torch.no_grad():
                f1, f2, features = self.local_emd_model(pos=balls, normal=ball_normals, batch=ball_batch)

        approaches = batch.approaches
        depth_proj = batch.depth_proj


        des_emd = torch.cat((f1,f2,features),dim=1)
        #print(des_emd.size())
        if train:
            self.global_emd_model.train()
            global_emd = self.global_emd_model(des_emd,ball_batch)
        else:
            self.global_emd_model.eval()
            with torch.no_grad():
                global_emd = self.global_emd_model(des_emd,ball_batch)

        valid_batch = ball_batch[reindexes]
        global_emd_valid = torch.cat([global_emd[i, :].repeat((valid_batch == i).sum(), 1) for i in range(len(sample))],dim=0)
        des_cat = torch.cat((balls[reindexes,:], ball_normals[reindexes,:], features[reindexes,:]), dim=-1)
        edge_attributes = torch.cat((depth_proj.unsqueeze(dim=-1),approaches),dim=-1)
        cat_all_orth_mask = torch.cat((des_cat, global_emd_valid, edge_attributes), dim=-1)

        #print(cat_all_orth_mask.size())

        if train:
            self.classifier_fail.train()
            scores_succ = self.classifier_fail(cat_all_orth_mask)
        else:
            self.classifier_fail.eval()
            with torch.no_grad():
                scores_succ = self.classifier_fail(cat_all_orth_mask)
        return scores_succ, depth_proj

    def train(self, batch, balance = True):
        scores_succ, _, = self.forward(batch, train=True)
        scores_succ = scores_succ.squeeze(dim=-1)
        labels_succ = batch.grasp_label
        # balance the positive and the negative
        if balance:
            # torch range include the end
            all_index = torch.range(0,len(labels_succ)-1,dtype=torch.long,device=self.device)
            positive_mask = labels_succ==1.
            negative_mask = labels_succ==0.
            positive_index = all_index[positive_mask]
            negative_index = all_index[negative_mask]
            if sum(positive_mask) == len(labels_succ) or sum(negative_mask) == len(labels_succ):
                return
            if sum(positive_mask) >= sum(negative_mask):
                sample_index = torch.randperm(len(positive_index))[:len(negative_index)]
                positive_index = positive_index[sample_index]
                balanced_index = torch.cat((positive_index, negative_index),dim=0)
            else:
                sample_index = torch.randperm(len(negative_index))[:len(positive_index)]
                negative_index = negative_index[sample_index]
                balanced_index = torch.cat((positive_index, negative_index), dim=0)
                #print(positive_index.size(),negative_index.size())
            labels_succ = labels_succ[balanced_index]
            scores_succ = scores_succ[balanced_index]
            #print('positive', sum(labels_succ), len(labels_succ))

        # metrics
        prediction_succ = scores_succ > 0.5
        prediction_succ = prediction_succ.to(torch.float).cpu().numpy()
        accuracy_succ = accuracy_score(labels_succ.cpu().numpy(), prediction_succ)
        balanced_accuracy_succ = balanced_accuracy_score(labels_succ.cpu().numpy(), prediction_succ)
        weights_succ = torch.ones_like(labels_succ, dtype=torch.float32).to(scores_succ.device)
        weights_succ[labels_succ == 1.] = (len(labels_succ)-sum(labels_succ))/sum(labels_succ)
        #print(weights_succ.sum(),len(weights_succ))
        loss = F.binary_cross_entropy_with_logits(scores_succ, labels_succ, weight=weights_succ)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return np.float32(loss.item()), accuracy_succ, balanced_accuracy_succ

    def test(self, batch, balance=True):

        scores_succ, _, = self.forward(batch, train=False)
        scores_succ = scores_succ.squeeze(dim=-1)
        labels_succ = batch.grasp_label

        if balance:
            # torch range include the end
            all_index = torch.range(0,len(labels_succ)-1,dtype=torch.long,device=self.device)
            positive_mask = labels_succ==1.
            negative_mask = labels_succ==0.
            positive_index = all_index[positive_mask]
            negative_index = all_index[negative_mask]
            if sum(positive_mask) == len(labels_succ) or sum(negative_mask) == len(labels_succ):
                return
            if sum(positive_mask) >= sum(negative_mask):
                sample_index = torch.randperm(len(positive_index))[:len(negative_index)]
                positive_index = positive_index[sample_index]
                balanced_index = torch.cat((positive_index, negative_index),dim=0)
            else:
                sample_index = torch.randperm(len(negative_index))[:len(positive_index)]
                negative_index = negative_index[sample_index]
                balanced_index = torch.cat((positive_index, negative_index), dim=0)
                #print(positive_index.size(),negative_index.size())

            labels_succ = labels_succ[balanced_index]
            scores_succ = scores_succ[balanced_index]
            #print('positive', sum(labels_succ), len(labels_succ))
        # metrics
        prediction_succ = scores_succ > 0.5
        prediction_succ = prediction_succ.to(torch.float).cpu().numpy()
        accuracy_succ = accuracy_score(labels_succ.cpu().numpy(), prediction_succ)
        balanced_accuracy_succ = balanced_accuracy_score(labels_succ.cpu().numpy(), prediction_succ)
        weights_succ = torch.ones_like(labels_succ, dtype=torch.float32).to(scores_succ.device)
        weights_succ[labels_succ == 1.] = (len(labels_succ) - sum(labels_succ)) / sum(labels_succ)
        loss = F.binary_cross_entropy_with_logits(scores_succ, labels_succ, weight=weights_succ)
        return np.float32(loss.item()), accuracy_succ, balanced_accuracy_succ

    def act(self, batch, train=False):
        scores_succ, depth_proj = self.forward(batch,train=train)
        approaches = batch.approaches
        sample_pos = batch.pos[batch.ball_edges[:, 0][batch.reindexes], :]
        des_normals = batch.normals[batch.ball_edges[:, 1][batch.reindexes], :]
        scores_succ = scores_succ.squeeze(dim=-1)

        return scores_succ, depth_proj, \
               approaches, sample_pos, \
               des_normals


    def save(self, filename1, filename2, filename3):
        self.local_emd_model.eval()
        self.global_emd_model.eval()
        self.classifier_fail.eval()

        torch.save(self.local_emd_model.state_dict(), filename1)
        torch.save(self.global_emd_model.state_dict(), filename2)
        torch.save(self.classifier_fail.state_dict(), filename3)


    def load(self, path1, path2, path3):
        self.local_emd_model.eval()
        self.global_emd_model.eval()
        self.classifier_fail.eval()

        self.local_emd_model.load_state_dict(torch.load(path1, self.device))
        self.global_emd_model.load_state_dict(torch.load(path2, self.device))
        self.classifier_fail.load_state_dict(torch.load(path3, self.device))
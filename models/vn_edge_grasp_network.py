from vn_pointnetpp import PointNetSimpleVn,GlobalEmdModelVn,Classifier,VNStdFeature
import torch
from torch_geometric.nn import  global_max_pool
#test rotation
import numpy as np
from transform import Rotation,Transform
from sklearn.metrics import accuracy_score,balanced_accuracy_score
import torch.nn.functional as F

class EdgeGrasp:
    def __init__(self, device, sample_num=32, lr= 0.5* 1e-3, ubn=False, normal=True, aggr='max'):
        self.device = device
        self.sample_num = sample_num
        self.local_emd_model = PointNetSimpleVn(out_channels=(32, 64, 128), train_with_norm=normal, ubn=False,
                                                aggr=aggr, train_with_all=False, k=16).to(device)

        self.global_emd_model = GlobalEmdModelVn(input_c=32+64+128, output_c=381, ubn=ubn, aggr=aggr).to(device)
        self.std = VNStdFeature(in_channels=512, normalize_frame=False, negative_slope=0.0, ubn=ubn).to(device)
        self.classifier_fail = Classifier(in_channels=512*3, hidden_channels=(256, 128, 64), ubn=False).to(
            device)
        self.parameter = list(self.local_emd_model.parameters()) + list(self.global_emd_model.parameters()) \
                         + list(self.classifier_fail.parameters()) + list(self.std.parameters())
        print('vn-ball # para', sum(p.numel() for p in self.parameter if p.requires_grad))
        self.classifier_para = list(self.global_emd_model.parameters()) + list(self.classifier_fail.parameters()) \
                               + list(self.std.parameters())

        self.optim = torch.optim.Adam([{'params': self.local_emd_model.parameters(), 'lr': lr},
                                       {'params': self.classifier_para}, ], lr=lr, weight_decay=1e-8)
        #self.optim = torch.optim.Adam(self.parameter, lr=lr, weight_decay=1e-8)

    def forward(self, batch, train=True, ):
        # balls setup
        ball_batch = batch.ball_batch
        ball_edges = batch.ball_edges
        reindexes = batch.reindexes
        balls = batch.pos[ball_edges[:, 1], :] - batch.pos[ball_edges[:, 0], :]
        ball_normals = batch.normals[ball_edges[:, 1], :]

        # todo add more node features
        # relative_pos_normalized = F.normalize(balls, p=2, dim=1)
        # ball_x_axis = torch.cross(ball_normals, relative_pos_normalized)
        # ball_x_axis = F.normalize(ball_x_axis, p=2, dim=1)
        # ball_approaches = torch.cross(ball_x_axis, ball_normals)
        # ball_approaches = -F.normalize(ball_approaches, p=2, dim=1)
        # get local_emd

        if train:
            self.local_emd_model.train()
            f1, f2, features = self.local_emd_model(pos=balls.unsqueeze(dim=1), batch=ball_batch,
                                            normal=ball_normals.unsqueeze(dim=1))
        else:
            self.local_emd_model.eval()
            with torch.no_grad():
                f1,f2,features = self.local_emd_model(pos=balls.unsqueeze(dim=1), batch=ball_batch,
                                                normal=ball_normals.unsqueeze(dim=1))

        #print(f1.size(),f2.size(),features.size())
        sample = batch.sample
        approaches = batch.approaches
        depth_proj = batch.depth_proj

        # todo consider all feature (skip connection)
        features_cat = torch.cat((f1,f2,features),dim=1)
        #print(features.size())
        if train:
            self.global_emd_model.train()
            global_emd = self.global_emd_model(features_cat, ball_batch)
        else:
            self.global_emd_model.eval()
            with torch.no_grad():
                global_emd = self.global_emd_model(features_cat, ball_batch)

        valid_batch = ball_batch[reindexes]
        global_emd_valid = torch.cat([global_emd[i, :,:].repeat((valid_batch == i).sum(), 1,1) for i in range(len(sample))],dim=0)
        des_cat = torch.cat((balls[reindexes,:].unsqueeze(dim=1),ball_normals[reindexes,:].unsqueeze(dim=1),features[reindexes, :,:]), dim=1)
        edge_cat_all = torch.cat((approaches.unsqueeze(dim=1), des_cat, global_emd_valid), dim=1)

        # print(edge_cat_all.size())
        # todo consider edge-wise z0
        if train:
            self.std.train()
            edge_std, z0 = self.std(edge_cat_all)
        else:
            self.std.eval()
            with torch.no_grad():
                edge_std, z0 = self.std(edge_cat_all)
        edge_std = torch.flatten(edge_std, start_dim=1)
        #print(edge_std.size())

        if train:
            self.classifier_fail.train()
            scores_succ = self.classifier_fail(edge_std)
        else:
            self.classifier_fail.eval()
            with torch.no_grad():
                scores_succ = self.classifier_fail(edge_std)

        return scores_succ, depth_proj, global_emd

    def act(self,batch):
        score_succ,_,_ = self.forward(batch,train=False)
        sample_pos = batch.pos[batch.ball_edges[:, 0][batch.reindexes], :]
        des_normals = batch.normals[batch.ball_edges[:, 1][batch.reindexes], :]
        score_succ = score_succ.squeeze(dim=-1)
        return score_succ,batch.depth_proj, batch.approaches, sample_pos, des_normals

    def check_equiv(self,batch):
        mean = batch.pos.mean(dim=0, keepdim=True)
        batch.pos = batch.pos - mean
        score, _, global_emd = self.forward(batch,train=False)
        #print(global_emd.size())
        rz = np.pi / 2.0 * np.random.choice(36)
        ry = np.pi / 2.0 * np.random.choice(36)
        rx = np.pi / 2.0 * np.random.choice(36)
        rot = Rotation.from_rotvec(np.r_[rx, ry, rz]).as_matrix()
        rot = torch.from_numpy(rot).to(torch.float).to(score.device)
        batch.pos = torch.einsum('nk,kj->nj',batch.pos, rot.T)
        mean = batch.pos.mean(dim=0, keepdim=True)
        batch.pos = batch.pos - mean
        score_from_trans, _, global_emd_from_transformed = self.forward(batch, train=False)
        transformed_global_emd = torch.einsum('nij,jk->nik',global_emd,rot.T)
        print(transformed_global_emd.size(),global_emd_from_transformed.size())
        print((global_emd_from_transformed-transformed_global_emd).sum())
        #print(torch.abs(score-score_from_trans).sum())

    def train(self, batch, balance=True):
        scores_succ, _,_ = self.forward(batch, train=True)
        scores_succ = scores_succ.squeeze(dim=-1)
        labels_succ = batch.grasp_label

        # balance the positive and the negative
        if balance:
            # torch range include the end
            all_index = torch.range(0, len(labels_succ) - 1, dtype=torch.long, device=self.device)
            positive_mask = labels_succ == 1.
            negative_mask = labels_succ == 0.
            positive_index = all_index[positive_mask]
            negative_index = all_index[negative_mask]
            if sum(positive_mask)==len(labels_succ) or sum(negative_mask)==len(labels_succ):
                return

            if sum(positive_mask) >= sum(negative_mask):
                sample_index = torch.randperm(len(positive_index))[:len(negative_index)]
                positive_index = positive_index[sample_index]
                balanced_index = torch.cat((positive_index, negative_index), dim=0)
            else:
                sample_index = torch.randperm(len(negative_index))[:len(positive_index)]
                negative_index = negative_index[sample_index]
                balanced_index = torch.cat((positive_index, negative_index), dim=0)
                # print(positive_index.size(),negative_index.size())
            labels_succ = labels_succ[balanced_index]
            scores_succ = scores_succ[balanced_index]
            # print('positive', sum(labels_succ), len(labels_succ))

        # metrics
        prediction_succ = scores_succ > 0.5
        prediction_succ = prediction_succ.to(torch.float).cpu().numpy()
        accuracy_succ = accuracy_score(labels_succ.cpu().numpy(), prediction_succ)
        balanced_accuracy_succ = balanced_accuracy_score(labels_succ.cpu().numpy(), prediction_succ)
        weights_succ = torch.ones_like(labels_succ, dtype=torch.float32).to(scores_succ.device)
        weights_succ[labels_succ == 1.] = (len(labels_succ) - sum(labels_succ)) / sum(labels_succ)
        # print(weights_succ.sum(),len(weights_succ))
        loss = F.binary_cross_entropy_with_logits(scores_succ, labels_succ, weight=weights_succ)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return np.float32(loss.item()), accuracy_succ, balanced_accuracy_succ

    def test(self, batch, balance=True):
        scores_succ, _,_ = self.forward(batch, train=False)
        scores_succ = scores_succ.squeeze(dim=-1)
        labels_succ = batch.grasp_label

        if balance:
            # torch range include the end
            all_index = torch.range(0, len(labels_succ) - 1, dtype=torch.long, device=self.device)
            positive_mask = labels_succ == 1.
            negative_mask = labels_succ == 0.
            positive_index = all_index[positive_mask]
            negative_index = all_index[negative_mask]
            if sum(positive_mask)==len(labels_succ) or sum(negative_mask)==len(labels_succ):
                return
            if sum(positive_mask) >= sum(negative_mask):
                sample_index = torch.randperm(len(positive_index))[:len(negative_index)]
                positive_index = positive_index[sample_index]
                balanced_index = torch.cat((positive_index, negative_index), dim=0)
            else:
                sample_index = torch.randperm(len(negative_index))[:len(positive_index)]
                negative_index = negative_index[sample_index]
                balanced_index = torch.cat((positive_index, negative_index), dim=0)
                # print(positive_index.size(),negative_index.size())

            labels_succ = labels_succ[balanced_index]
            scores_succ = scores_succ[balanced_index]
            # print('positive', sum(labels_succ), len(labels_succ))
        # metrics
        prediction_succ = scores_succ > 0.5
        prediction_succ = prediction_succ.to(torch.float).cpu().numpy()
        accuracy_succ = accuracy_score(labels_succ.cpu().numpy(), prediction_succ)
        balanced_accuracy_succ = balanced_accuracy_score(labels_succ.cpu().numpy(), prediction_succ)
        weights_succ = torch.ones_like(labels_succ, dtype=torch.float32).to(scores_succ.device)
        weights_succ[labels_succ == 1.] = (len(labels_succ) - sum(labels_succ)) / sum(labels_succ)
        loss = F.binary_cross_entropy_with_logits(scores_succ, labels_succ, weight=weights_succ)
        return np.float32(loss.item()), accuracy_succ, balanced_accuracy_succ

    def save(self, filename1, filename2, filename3, filename4):
        self.local_emd_model.eval()
        self.global_emd_model.eval()
        #self.classifier_orth.eval()
        self.classifier_fail.eval()
        self.std.eval()
        torch.save(self.local_emd_model.state_dict(), filename1)
        torch.save(self.global_emd_model.state_dict(), filename2)
        torch.save(self.classifier_fail.state_dict(), filename3)
        torch.save(self.std.state_dict(), filename4)

    def load(self, path1, path2, path3, path4):
        self.local_emd_model.eval()
        self.global_emd_model.eval()
        self.classifier_fail.eval()
        self.std.eval()

        self.local_emd_model.load_state_dict(torch.load(path1, self.device))
        self.global_emd_model.load_state_dict(torch.load(path2, self.device))
        self.classifier_fail.load_state_dict(torch.load(path3, self.device))
        self.std.load_state_dict(torch.load(path4, self.device))

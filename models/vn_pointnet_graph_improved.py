import torch
from torch.nn import Sequential, Linear, ReLU,BatchNorm1d
#from torch_geometric.nn import PointConv as PointNetConv
from torch_geometric.nn import knn_interpolate,knn_graph
from torch_scatter import scatter,scatter_max
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")


EPS = 1e-7

class Classifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=(512,256,128),ubn=False):
        super().__init__()
        if ubn:
            self.head = Sequential(Linear(in_channels, hidden_channels[0]),
                                   BatchNorm1d(hidden_channels[0]),
                                   ReLU(),
                                   Linear(hidden_channels[0], hidden_channels[1]),
                                   BatchNorm1d(hidden_channels[1]),
                                   ReLU(),
                                   Linear(hidden_channels[1], hidden_channels[2]),
                                   BatchNorm1d(hidden_channels[2]),
                                   ReLU(),
                                   Linear(hidden_channels[2], 1), )
        else:
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


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [ N_samples, N feat, 3]
        '''
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.0):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        x: point features of shape [ N_samples, N feat, 3]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
                    mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out

class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim=3):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        return x

class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, share_nonlinearity=False, negative_slope=0.0, ubn=False):
        super(VNLinearLeakyReLU, self).__init__()

        self.negative_slope = negative_slope
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        # use batch norm or not
        self.ubn = ubn
        if self.ubn:
            self.bn = VNBatchNorm(num_features=out_channels)
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)


    def forward(self, x):
        '''
        #x: point features of shape [B, N_feat, 3, N_samples, ...]
        x: point features of shape [ N_samples, N feat, 3]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        if self.ubn:
            p = self.bn(p)
        # BatchNorm todo later
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (
                    mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out




class VNStdFeature(nn.Module):
    def __init__(self, in_channels, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2,ubn=False):
        super(VNStdFeature, self).__init__()
        self.normalize_frame = normalize_frame
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels // 2, share_nonlinearity=share_nonlinearity,
                                     negative_slope=negative_slope,ubn=ubn)
        self.vn2 = VNLinearLeakyReLU(in_channels // 2, in_channels // 4, share_nonlinearity=share_nonlinearity,
                                     negative_slope=negative_slope,ubn=ubn)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels // 4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''

        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)

        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:, 0, :]
            # u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm + EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdims=True) * u1
            # u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm + EPS)

            # compute the cross product of the two output vectors
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        if len(z0.size()) == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        if len(z0.size()) == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif len(z0.size()) == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)

        return x_std, z0


class PointNetConvVn(torch.nn.Module):
    # todo enable batch training later
    def __init__(self, in_channels=2, out_channels=64, aggr='max'):
        # Message passing with "max" aggregation.
        super().__init__()
        self.vn_linear_1 = VNLinear(in_channels,out_channels)
        self.relu_1 = VNLeakyReLU(out_channels,negative_slope=0.0)
        self.vn_linear_2 = VNLinear(out_channels,out_channels)
        self.aggr = aggr
        if self.aggr == 'max':
            self.map_to_dir = VNLinear(out_channels, out_channels)

    def forward(self, x_vn, pos_vn, edge_index):
        # Start propagating messages.
        # pos: N x 3 -> N x 1 x 3
        # x: N x 3F -> N x F x 3
        relative_pos_vn = pos_vn[edge_index[0, :], :, :] - pos_vn[edge_index[1, :], :, :]
        # relative pos_vn satisfy the equivariance
        x_vn = torch.cat((x_vn[edge_index[0, :], :, :], relative_pos_vn),dim=1)
        x_vn = self.vn_linear_1(x_vn)
        x_vn = self.relu_1(x_vn)
        x_vn = self.vn_linear_2(x_vn)
        if self.aggr == 'mean':
            x_vn = scatter(x_vn,edge_index[1,:],dim=0,reduce='mean')
        if self.aggr == 'max':
            d = self.map_to_dir(x_vn)
            dotprod = (x_vn * d).sum(2, keepdims=True)
            _, idx = scatter_max(dotprod,edge_index[1,:],dim=0,out=torch.zeros(pos_vn.size(0),x_vn.size(1),1,device=pos_vn.device)+torch.min(dotprod)-1,)
            # idx = idx.squeeze(dim=-1)
            # out = [x_vn[:,i,:][idx[:,i],:] for i in range(idx.size(1))]
            # out = torch.stack(out,dim=1)
            x_vn = torch.gather(input=x_vn, dim=0, index=idx.repeat(1,1,3))
        return x_vn


class PointNetSimpleVn(torch.nn.Module):
    def __init__(self, out_channels=(64,64,128), train_with_norm=False,ubn=False,aggr='max',train_with_all=False,
                 train_with_xaxis=False,k=24):
        super().__init__()
        torch.manual_seed(12345)

        if train_with_norm:
            in_channels = 6//3
        elif train_with_xaxis:
            in_channels = 9//3
        elif train_with_all:
            in_channels = 12//3
        else:
            in_channels = 3//3
        #out_channels = out_channels
        self.train_with_normal = train_with_norm
        self.train_with_all = train_with_all
        self.train_with_xaxis = train_with_xaxis
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.conv1 = PointNetConvVn(in_channels+1, out_channels[0],aggr=aggr)
        self.relu1 = VNLeakyReLU(out_channels[0], negative_slope=0.0)
        self.conv2 = PointNetConvVn(out_channels[0]+1,  out_channels[1],aggr=aggr)
        self.relu2 = VNLeakyReLU(out_channels[1], negative_slope=0.0)
        self.conv3 = PointNetConvVn(out_channels[1] + 1, out_channels[2], aggr=aggr)
        self.relu3 = VNLeakyReLU(out_channels[2], negative_slope=0.0)
        self.ubn = ubn
        self.k = k
        if self.ubn:
            self.bn1 = VNBatchNorm(out_channels[0])
            self.bn2 = VNBatchNorm(out_channels[1])
            self.bn3 = VNBatchNorm(out_channels[2])


    def forward(self, pos, batch=None, normal=None, ball_approach=None, ball_xaxis=None):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.

        assert len(pos.size())==3
        if self.train_with_normal:
            assert normal is not None
            h = torch.cat((pos, normal), dim=1)
        elif self.train_with_xaxis:
            assert ball_xaxis is not None
            h = torch.cat((pos,normal,ball_xaxis),dim=1)
        elif self.train_with_all:
            assert ball_approach is not None
            h = torch.cat((pos,normal,ball_approach,ball_xaxis),dim=1)
        else:
            h = pos

        #print('pointnet++ input',h.size())
        edge_index = knn_graph(pos.squeeze(dim=1), k=self.k, batch=batch, loop=True)
        h1 = self.conv1(x_vn=h, pos_vn = pos, edge_index=edge_index)
        # batch norm should be added on the node features instead of edges
        if self.ubn:
            h1 = self.bn1(h1)
        h1 = self.relu1(h1)
        h2 = self.conv2(x_vn =h1, pos_vn=pos, edge_index=edge_index)
        if self.ubn:
            h2 = self.bn2(h2)
        h2 = self.relu2(h2)
        h3 = self.conv3(x_vn=h2, pos_vn=pos, edge_index=edge_index)
        if self.ubn:
            h3 = self.bn3(h3)
        h3 = self.relu3(h3)

        return h1,h2,h3

class GlobalEmdModelVn(torch.nn.Module):
    def __init__(self,input_c = 128, inter_c=128, output_c=256, aggr='max', share_nonlinearity=False,negative_slope=0.0,ubn=False):
        super().__init__()
        self.vn1 = VNLinearLeakyReLU(input_c, inter_c, share_nonlinearity=share_nonlinearity,negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(inter_c, inter_c, share_nonlinearity=share_nonlinearity,negative_slope=negative_slope)
        self.vn3 = VNLinear(inter_c,inter_c)
        self.aggr = aggr
        if self.aggr == 'max':
            self.map_to_dir_1 = VNLinear(inter_c, inter_c)
        self.vn4 = VNLinearLeakyReLU(input_c+inter_c, output_c, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn5 = VNLinear(output_c, output_c)
        if self.aggr == 'max':
            self.map_to_dir_2 = VNLinear(output_c, output_c)
        self.ubn = ubn
        if self.ubn:
            self.bn3 = VNBatchNorm(inter_c)
            self.bn5 = VNBatchNorm(output_c)
            self.relu3 = VNLeakyReLU(inter_c)
            self.relu5 = VNLeakyReLU(output_c)
        if not self.ubn:
            self.relu3 = VNLeakyReLU(inter_c)
            self.relu5 = VNLeakyReLU(output_c)

    def forward(self,pos_emd, radius_p_batch):
        # pos_emd: (M,F,3), radius_p_batch: (M,)
        global_emd = self.vn1(pos_emd)
        global_emd = self.vn2(global_emd)
        global_emd = self.vn3(global_emd)
        # first max pool
        if self.aggr == 'mean':
            global_emd = scatter(global_emd,radius_p_batch,dim=0,reduce='mean')
        if self.aggr == 'max':
            d1 = self.map_to_dir_1(global_emd)
            dotprod1 = (global_emd * d1).sum(2, keepdims=True)
            _, idx = scatter_max(dotprod1,radius_p_batch,dim=0,out=torch.zeros(torch.max(radius_p_batch).to(torch.long)+1,
                                                                               global_emd.size(1),1,device=d1.device)+torch.min(dotprod1)-1)
            # idx = idx.squeeze(dim=-1)
            # out = [x_vn[:,i,:][idx[:,i],:] for i in range(idx.size(1))]
            # out = torch.stack(out,dim=1)
            global_emd = torch.gather(input=global_emd, dim=0, index=idx.repeat(1,1,3))

        if not self.ubn:
            global_emd = self.relu3(global_emd)
        # todo batch norm + relu after
        if self.ubn:
            global_emd = self.relu3(self.bn3(global_emd))

        global_emd = torch.cat([global_emd[i,:,:].repeat((radius_p_batch==i).sum(),1,1) for i in range(len(global_emd))],dim=0)
        global_emd = torch.cat((pos_emd,global_emd),dim=1)
        global_emd = self.vn4(global_emd)
        global_emd = self.vn5(global_emd)
        #second max pool
        if self.aggr == 'mean':
            global_emd = scatter(global_emd, radius_p_batch, dim=0, reduce='mean')
        if self.aggr == 'max':
            d2 = self.map_to_dir_2(global_emd)
            dotprod2 = (global_emd * d2).sum(2, keepdims=True)
            _, idx = scatter_max(dotprod2, radius_p_batch, dim=0,
                                 out=torch.zeros(torch.max(radius_p_batch).to(torch.long)+1,global_emd.size(1), 1,device=d2.device) + torch.min(dotprod2) - 1.)
            # idx = idx.squeeze(dim=-1)
            # out = [x_vn[:,i,:][idx[:,i],:] for i in range(idx.size(1))]
            # out = torch.stack(out,dim=1)
            global_emd = torch.gather(input=global_emd, dim=0, index=idx.repeat(1, 1, 3))

        if not self.ubn:
            global_emd = self.relu5(global_emd)
        # todo batch norm + relu after
        if self.ubn:
            global_emd = self.relu5(self.bn5(global_emd))
        return global_emd



# normal = torch.rand(1024,1,3)
# pointnetsimple = PointNetSimpleVn()
# y = pointnetsimple(pos=x,normal=normal)
# print(y.size())
# stdmodule = VNStdFeature(in_channels=128,negative_slope=0.0,)
# x_std, z0 = stdmodule(y)
# print(z0.size(),x_std.size())
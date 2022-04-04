from torch.nn import Sequential, Linear, ReLU
import torch
import torch.nn.functional as F
#from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import  PPFConv,knn_graph
from torch_geometric.nn import  PointConv as PointNetConv
from torch_cluster import fps
from dataset_pyg import Grasp_Dataset
from torch_geometric.data import DataLoader



class Classifier(torch.nn.Module):
    def __init__(self, in_channels,hidden_channels):
        super().__init__()
        self.head =  Sequential(Linear(in_channels, hidden_channels),
                                ReLU(),
                                Linear(hidden_channels, hidden_channels),
                                ReLU(),
                                Linear(hidden_channels, 1),)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.head(x)
        return self.sigmoid(x)


class PointNet(torch.nn.Module):
    def __init__(self, train_with_norm=False):
        super().__init__()
        torch.manual_seed(12345)

        if train_with_norm:
            in_channels = 6
        else:
            in_channels = 3
        out_channels = 256
        self.train_with_normal = train_with_norm
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        mlp1 = Sequential(Linear(in_channels + 3, out_channels),
                          ReLU(),
                          Linear(out_channels, out_channels))
        self.conv1 = PointNetConv(local_nn=mlp1)

        mlp2 = Sequential(Linear(out_channels + 3, out_channels),
                          ReLU(),
                          Linear(out_channels, out_channels))
        self.conv2 = PointNetConv(local_nn=mlp2)

        mlp3 = Sequential(Linear(out_channels + 3, out_channels),
                          ReLU(),
                          Linear(out_channels, out_channels))
        self.conv3 = PointNetConv(local_nn=mlp3)

        self.classifier = Classifier(in_channels=out_channels,hidden_channels=128)

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
        return self.classifier(h)



# model = PointNet()
# print(model)
# dataset = Grasp_Dataset(root='./raw1/foo',train=True)
# print(len(dataset))
# loader = DataLoader(dataset,batch_size=1,shuffle=False)
# for batch in loader:
#     print(batch)
#     print(batch.batch)
#     print(len(batch))
#     print(batch.positive_mask)
#     print(batch.label)
#     y = model(batch.pos, batch.batch)
#     print(y.size())
#     break

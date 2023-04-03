import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
#sys.path.append('..')
from models.dataset_processor import Grasp_Dataset, GraspNormalization, GraspAugmentation, PreTransformBallBox,SubsampleBall
from models.edge_grasper import EdgeGrasper
from models.vn_edge_grasper import EdgeGrasper as VNEdgeGrasper
import torch
#from models.utils import write_test,write_training
import argparse
from torch_geometric.data import DataLoader
import warnings
from torch_geometric.transforms import Compose
import numpy as np
from torch.backends import cudnn
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='edge_grasper_ball')
parser.add_argument('--dataset_dir', type=str, default='./grasp_raw_data')
parser.add_argument('--save_dir', type=str, default='./edge_grasp_net_pretrained_para')
parser.add_argument('--vn_save_dir', type=str, default='./vn_edge_pretrained_para')
parser.add_argument('--load', type=int, default=0)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--sample_num', type=int, default=5)
parser.add_argument('--test_interval', type=int, default=1)
parser.add_argument('--save_interval', type=int, default=10)
parser.add_argument('--verbose', action="store_true", default=False)
parser.add_argument('--train', action="store_true", default=False)
parser.add_argument('--vn', action="store_true", default=False)
args = parser.parse_args()

def main(args):
    np.random.seed(1)
    torch.set_num_threads(1)
    torch.manual_seed(1)
    cudnn.benchmark = True
    cudnn.deterministic = True
    # set up the model
    if args.vn:
        tr_dataset = Grasp_Dataset(root=args.dataset_dir, pre_transform=PreTransformBallBox(),transform=Compose([GraspNormalization(),SubsampleBall()]), train=True)
        tr_loader = DataLoader(tr_dataset[:len(tr_dataset)], batch_size=1, shuffle=True)
        edge_grasper = VNEdgeGrasper(device=1, root_dir=args.vn_save_dir, sample_num=args.sample_num, lr=0.5 * 1e-4,
                                     load=args.load, ubn=False, normal=True, aggr='max')
    else:
        tr_dataset = Grasp_Dataset(root=args.dataset_dir, pre_transform=PreTransformBallBox(max_width=True),
                                   transform=Compose([GraspNormalization(), GraspAugmentation()]), train=True)
        tr_loader = DataLoader(tr_dataset[:], batch_size=1, shuffle=True)
        edge_grasper = EdgeGrasper(device=1,root_dir=args.save_dir, sample_num=args.sample_num, lr=1e-4, load=args.load)

    # set up the dataset
    tst_dataset = Grasp_Dataset(root=args.dataset_dir, transform=GraspNormalization(), train=False)
    tst_loader = DataLoader(tst_dataset[:], batch_size=1, shuffle=False)
    print('# training data:', len(tr_loader), '\n', '# testing data', len(tst_loader))
    #edge_grasper.test_draw(tr_dataset[3],learned=False)
    train = args.train
    if train:
        edge_grasper.train_test_save(tr_loader, tst_loader,tr_epoch=args.epoch,test_interval=args.test_interval,
                                     save_interval=args.save_interval,log=False,verbose=args.verbose)
if __name__ == "__main__":
    main(args)
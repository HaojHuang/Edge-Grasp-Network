import os.path
from dataset_pyg import Grasp_Dataset,GraspNormalization,GraspAugmentation
from edge_grasp_model_revised3 import EdgeGrasp
import torch
import time
from utils import write_test2,write_training
import argparse
from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from torch_geometric.transforms import Compose
from torch.backends import cudnn

class EdgeGrasper:
    def __init__(self,device,root_dir='./store',sample_num=32,position_emd=True,lr=1e-5, load=False):
        if device == 1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device('cpu')
        self.device = device
        self.position_emd = position_emd
        self.model = EdgeGrasp(device=self.device, sample_num=sample_num, position_emd=self.position_emd, lr=lr)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        self.root_dir = root_dir
        self.parameter_dir = os.path.join(root_dir,'checkpoints')
        if load != False:
            # print('load pretained model checkpoint at {} step'.format(load))
            self.load(load)
            self.epoch_num = load
        else:
            self.epoch_num = 1

    def train_test_save(self, train_dataset, test_dataset, tr_epoch=200, verbose=True, test_interval=1,save_interval=100,log=True):
        #time0 = time.time()
        for epoch_num in range(self.epoch_num,tr_epoch+1):
            step = 1
            for batch in train_dataset:
                res= self.model.train(batch.to(self.device))
                if res is not None:
                    loss_total, loss, accu, ba_acc, loss_dot2, accu_dot2, ba_acc_dot2 = res
                else:
                    continue
                if log:
                    write_training(self.root_dir,epoch_num,step,loss,accu,ba_acc)
                if verbose:
                    print("Epoch: {}/{}, Step {}, Tr Total: {:.5f} ,"
                          "Tr loss: {:.5f}, Tr Acc: {:.5f}, Tr Balanced Acc: {:.5f}, "
                          "Tr loss2: {:.5f}, Tr Acc2: {:.5f},Tr Balanced Acc2: {:.5f}"
                          .format(epoch_num,tr_epoch,step,loss_total,loss,accu,ba_acc,loss_dot2, accu_dot2, ba_acc_dot2))
                step = step + 1
            # todo check for later
            if epoch_num % test_interval ==0:
                self.test(test_dataset)
            if epoch_num % save_interval ==0:
                self.save()
            self.epoch_num += 1

    def test(self,test_dataset):
        total_losst = 0.
        total_loss = 0.
        total_accu = 0.
        total_ba_accu = 0.
        total_loss2 = 0.
        total_accu2 = 0.
        total_ba_accu2 = 0.
        tst_step = 0

        for batch in test_dataset:
            res = self.model.test(batch.to(self.device))
            if res is not None:
                tst_step+=1
                loss_total, loss, accu, ba_acc, loss2, accu2, ba_acc2 = res
                total_losst += loss_total
                total_loss += loss
                total_accu += accu
                total_ba_accu += ba_acc
                total_loss2 += loss2
                total_accu2 += accu2
                total_ba_accu2 += ba_acc2

        print("Test at Epoch {}, Tst avg loss totoal: {:.5f}, "
              "Tst avg loss: {:.5f}, Tst avg Acc: {:.5f},Tst avg Balanced Acc: {:.5f} "
              "Tst avg loss2: {:.5f}, Tst avg Acc2: {:.5f},Tst avg Balanced Acc2: {:.5f}"
              .format(self.epoch_num,total_losst/tst_step,total_loss/tst_step, total_accu/tst_step, total_ba_accu/tst_step,
                      total_loss2/tst_step, total_accu2/tst_step, total_ba_accu2/tst_step,))
        write_test2(self.root_dir,self.epoch_num,0,total_losst/tst_step, total_loss/tst_step, total_accu/tst_step, total_ba_accu/tst_step,
                    total_loss2/tst_step, total_accu2/tst_step, total_ba_accu2/tst_step,)

    def save(self,):
        if not os.path.exists(self.parameter_dir):
            os.makedirs(self.parameter_dir)
        fname1 = 'local_emd_model-ckpt-%d.pt' % self.epoch_num
        fname2 = 'global_emd_model-ckpt-%d.pt' % self.epoch_num
        fname3 = 'classifier_model-ckpt-%d.pt' % self.epoch_num
        fname4 = 'classifier_2-ckpt-%d.pt' % self.epoch_num

        fname1 = os.path.join(self.parameter_dir,fname1)
        fname2 = os.path.join(self.parameter_dir, fname2)
        fname3 = os.path.join(self.parameter_dir, fname3)
        fname4 = os.path.join(self.parameter_dir, fname4)
        self.model.save(fname1,fname2,fname3,fname4)
        print('save the parameters to' + fname1)

    def load(self,n_iter):
        fname1 = 'local_emd_model-ckpt-%d.pt' % n_iter
        fname2 = 'global_emd_model-ckpt-%d.pt' % n_iter
        fname3 = 'classifier_model-ckpt-%d.pt' % n_iter
        fname4 = 'classifier_2-ckpt-%d.pt' % n_iter
        fname1 = os.path.join(self.parameter_dir, fname1)
        fname2 = os.path.join(self.parameter_dir, fname2)
        fname3 = os.path.join(self.parameter_dir, fname3)
        fname4 = os.path.join(self.parameter_dir, fname4)
        self.model.load(fname1,fname2,fname3,fname4)
        print('Load the parameters from' + fname1)

parser = argparse.ArgumentParser(description='edge_grasper')
parser.add_argument('--dataset_dir', type=str, default='./raw/foo')
parser.add_argument('--load', type=int, default=0)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--sample_num', type=int, default=32)
parser.add_argument('--test_interval', type=int, default=10)
parser.add_argument('--save_interval', type=int, default=10)
args = parser.parse_args()

def main(args):
    np.random.seed(1)
    torch.set_num_threads(1)
    torch.manual_seed(1)
    cudnn.benchmark = True
    cudnn.deterministic = True
    #set up the dataset
    tr_dataset = Grasp_Dataset(root=args.dataset_dir, transform=Compose([GraspNormalization(),GraspAugmentation()]), train=True)
    tr_loader = DataLoader(tr_dataset[:], batch_size=1, shuffle=True)
    tst_dataset = Grasp_Dataset(root=args.dataset_dir, transform=GraspNormalization(), train=False)
    tst_loader = DataLoader(tst_dataset[:], batch_size=1, shuffle=False)
    print(len(tr_loader),len(tst_loader))
    # set up the model
    edge_grasper = EdgeGrasper(device=1,root_dir='./store16', sample_num=args.sample_num, lr=0.5*1e-4, load=args.load)
    #edge_grasper.test_draw(tr_dataset[3],learned=False)
    edge_grasper.train_test_save(tr_loader,tst_loader,tr_epoch=args.epoch,test_interval=args.test_interval,save_interval=args.save_interval,log=False)
if __name__ == "__main__":
    main(args)
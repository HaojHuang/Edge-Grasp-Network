import os.path
from dataset_pyg import Grasp_Dataset,GraspNormalization
from edge_grasp_model import EdgeGrasp
import torch
import time
from utils import write_test,write_training
import argparse
from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

class EdgeGrasper:
    def __init__(self,device,root_dir='./store',sample_num=32,position_emd=True,lr=1e-5, load=False):
        if device == 1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device('cpu')
        self.device = device
        self.position_emd = position_emd
        self.model = EdgeGrasp(device=self.device,sample_num=sample_num,position_emd=self.position_emd,lr=lr)
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
                    loss, accu, ba_acc = res
                else:
                    continue
                if log:
                    write_training(self.root_dir,epoch_num,step,loss,accu,ba_acc)
                if verbose:
                    print("Epoch: {}/{}, Step {}, Tr loss: {:.5f}, Tr Acc: {:.5f},Tr Balanced Acc: {:.5f}"
                          .format(epoch_num,tr_epoch,step,loss,accu,ba_acc))
                step = step + 1
            # todo check for later
            if epoch_num % test_interval ==0:
                self.test(test_dataset)
            if epoch_num % save_interval ==0:
                self.save()
            self.epoch_num += 1

    def test(self,test_dataset):
        total_loss = 0.
        total_accu = 0.
        total_ba_accu = 0.
        tst_step = len(test_dataset)
        for batch in test_dataset:
            loss, accu, ba_acc = self.model.test(batch.to(self.device))
            total_loss += loss
            total_accu += accu
            total_ba_accu += ba_acc
        print("Test at Epoch {}, Tst avg loss: {:.5f}, Tst avg Acc: {:.5f},Tst avg Balanced Acc: {:.5f}"
              .format(self.epoch_num,total_loss/tst_step, total_accu/tst_step, total_ba_accu/tst_step))
        write_test(self.root_dir,self.epoch_num,0,total_loss/tst_step, total_accu/tst_step, total_ba_accu/tst_step)

    def save(self,):
        if not os.path.exists(self.parameter_dir):
            os.makedirs(self.parameter_dir)
        fname1 = 'local_emd_model-ckpt-%d.pt' % self.epoch_num
        fname2 = 'global_emd_model-ckpt-%d.pt' % self.epoch_num
        fname3 = 'classifier_model-ckpt-%d.pt' % self.epoch_num
        fname1 = os.path.join(self.parameter_dir,fname1)
        fname2 = os.path.join(self.parameter_dir, fname2)
        fname3 = os.path.join(self.parameter_dir, fname3)
        self.model.save(fname1,fname2,fname3)
        print('save the parameters to' + fname1)

    def load(self,n_iter):
        fname1 = 'local_emd_model-ckpt-%d.pt' % n_iter
        fname2 = 'global_emd_model-ckpt-%d.pt' % n_iter
        fname3 = 'classifier_model-ckpt-%d.pt' % n_iter
        fname1 = os.path.join(self.parameter_dir, fname1)
        fname2 = os.path.join(self.parameter_dir, fname2)
        fname3 = os.path.join(self.parameter_dir, fname3)
        self.model.save(fname1,fname2,fname3)
        print('Load the parameters from' + fname1)


parser = argparse.ArgumentParser(description='edge_grasper')
parser.add_argument('--dataset_dir', type=str, default='./raw1/foo')
parser.add_argument('--load', type=int, default=0)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--sample_num', type=int, default=32)
parser.add_argument('--test_interval', type=int, default=10)
parser.add_argument('--save_interval', type=int, default=100)
args = parser.parse_args()

def main(args):
    #set up the dataset
    tr_dataset = Grasp_Dataset(root=args.dataset_dir, transform=GraspNormalization(), train=True)
    tr_loader = DataLoader(tr_dataset[:], batch_size=1, shuffle=False)
    tst_dataset = Grasp_Dataset(root=args.dataset_dir, transform=GraspNormalization(), train=False)
    tst_loader = DataLoader(tst_dataset[:], batch_size=1, shuffle=False)
    print(len(tr_loader),len(tst_loader))
    # set up the model
    edge_grasper = EdgeGrasper(device=1,root_dir='./store',sample_num=args.sample_num,lr=0.5*1e-4,load=args.load)
    #edge_grasper.test_draw(tr_dataset[3],learned=False)
    edge_grasper.train_test_save(tr_loader,tst_loader,tr_epoch=args.epoch,test_interval=args.test_interval,
                                 save_interval=args.save_interval,log=False)
if __name__ == "__main__":
    main(args)
import os.path
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
#sys.path.append('..')
from dataset_processor import Grasp_Dataset, GraspNormalization, GraspAugmentation, PreTransformBallBox
from edge_grasp_network import EdgeGrasp
import torch
from utils import write_test,write_training
import argparse
from torch_geometric.data import DataLoader
import warnings
from torch_geometric.transforms import Compose
import numpy as np
from torch.backends import cudnn
warnings.filterwarnings("ignore")


class EdgeGrasper:
    def __init__(self,device,root_dir='./store',sample_num=32,position_emd=True,lr=1e-5, load=False):
        if device == 1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device('cpu')
        self.device = device
        self.position_emd = position_emd
        self.model = EdgeGrasp(device=self.device, sample_num=sample_num, lr=lr)
        # set scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.model.optim, mode='min', factor=0.5,
                                                                    patience=6, verbose=True)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        self.root_dir = root_dir
        self.parameter_dir = os.path.join(root_dir,'checkpoints')
        if load != False:
            # print('load pretained model checkpoint at {} step'.format(load))
            self.load(load)
            self.epoch_num = load+1
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
                    print("Epoch: {}/{}, Step {},"
                          "Tr loss: {:.5f}, Tr Acc: {:.5f}, Tr Balanced Acc: {:.5f}, "
                          .format(epoch_num,tr_epoch,step,
                                  loss,accu,ba_acc,))
                step = step + 1
            # todo check for later
            if epoch_num % test_interval ==0:
                validation_loss = self.test(test_dataset)
                self.scheduler.step(validation_loss)
            if epoch_num % save_interval ==0:
                self.save()
            self.epoch_num += 1

    def test(self,test_dataset):
        total_loss = 0.
        total_accu = 0.
        total_ba_accu = 0.
        tst_step = 0

        for batch in test_dataset:
            res  = self.model.test(batch.to(self.device))
            if res is not None:
                loss, accu, ba_acc = res
                tst_step+=1
                total_loss += loss
                total_accu += accu
                total_ba_accu += ba_acc

        print("Test at Epoch {},"
              "Tst avg loss: {:.5f}, Tst avg Acc: {:.5f},Tst avg Balanced Acc: {:.5f} "
              .format(self.epoch_num,
                      total_loss/tst_step, total_accu/tst_step, total_ba_accu/tst_step,))

        write_test(self.root_dir,self.epoch_num, 0, total_loss/tst_step, total_accu/tst_step, total_ba_accu/tst_step,)
        return total_loss / tst_step

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
        self.model.load(fname1,fname2,fname3,)
        print('Load the parameters from' + fname1)
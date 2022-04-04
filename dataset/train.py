from dataset_pyg import Grasp_Dataset, GraspNormalization
from torch_geometric.data import DataLoader
from aggregation_model import PointNet
import torch
import torch.nn.functional as F
import os
import time
import torch.backends.cudnn as cudnn

def get_mask(positive_mask,positive_numbers,points_numbers):
    positive_numbers = torch.cat([torch.tensor([0],device=positive_numbers.device), positive_numbers])
    points_numbers = torch.cat([torch.tensor([0],device=points_numbers.device), points_numbers])
    positive_numbers_ptr = torch.cumsum(positive_numbers,dim=0)
    points_numbers_ptr = torch.cumsum(points_numbers,dim=0)
    mask_global = torch.cat([positive_mask[positive_numbers_ptr[i]:positive_numbers_ptr[i + 1]] + points_numbers_ptr[i] for i in
                             range(len(positive_numbers_ptr) - 1)])
    return mask_global


def train(model,train_loader,optimizer,device,unbalanced_weight=1.08):
    model.to(device)
    model.train()
    total_loss = 0
    total_correct_prediction = 0
    total_num_prediction = 0

    for idex,batch in enumerate(train_loader):
        batch = batch.to(device)
        score = model(pos=batch.pos,batch = batch.batch,normal=batch.normals)
        positive_mask = get_mask(batch.positive_mask, batch.positive_numbers, batch.points_numbers)
        negative_mask = get_mask(batch.negative_mask, batch.negative_numbers, batch.points_numbers)
        score_with_plabel = score[positive_mask,:]
        score_with_nlabel = score[negative_mask,:]
        score_with_label = torch.cat((score_with_plabel,score_with_nlabel),dim=0)
        label_positive = torch.ones(size=(len(positive_mask), 1), device=device)
        label_negative = torch.zeros(size=(len(negative_mask), 1), device=device)
        label = torch.cat((label_positive,label_negative),dim=0)
        ## the positive sample > negative sample
        weights = torch.ones_like(label)
        weights[:len(positive_mask),:] = unbalanced_weight
        loss = F.binary_cross_entropy(score_with_label,label,weight=weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pp = score_with_plabel > 0.5
        nn = score_with_nlabel < 0.5
        total_correct_prediction = total_correct_prediction + pp.sum().item() + nn.sum().item()
        total_num_prediction += len(batch.label)
    return total_loss/len(train_loader), total_correct_prediction/total_num_prediction

def test(model,test_loader,device):
    model.eval()
    total_loss = 0
    total_correct_prediction = 0
    total_num_prediction = 0

    for idex,batch in enumerate(test_loader):
        batch = batch.to(device)
        score = model(batch.pos,batch = batch.batch,normal=batch.normals)
        positive_mask = get_mask(batch.positive_mask, batch.positive_numbers, batch.points_numbers)
        negative_mask = get_mask(batch.negative_mask, batch.negative_numbers, batch.points_numbers)
        score_with_plabel = score[positive_mask,:]
        score_with_nlabel = score[negative_mask,:]
        score_with_label = torch.cat((score_with_plabel,score_with_nlabel),dim=0)
        label_positive = torch.ones(size=(len(positive_mask), 1), device=device)
        label_negative = torch.zeros(size=(len(negative_mask), 1), device=device)
        label = torch.cat((label_positive,label_negative),dim=0)
        ## the positive sample > negative sample
        loss = F.binary_cross_entropy(score_with_label,label)
        total_loss += loss.item()
        pp = score_with_plabel > 0.5
        nn = score_with_nlabel < 0.5
        total_correct_prediction = total_correct_prediction + pp.sum().item() + nn.sum().item()
        total_num_prediction += len(batch.label)
    return total_loss/len(test_loader), total_correct_prediction/total_num_prediction


def print_info(info,log_fp=None):
    message = ("Epoch: {}/{}, Duration: {:.3f}s, " \
              "Train Loss: {:.4f}, Train Accuracy: {:.4f}, " \
              "Test Loss: {:.4f}, Test Accuracy: {:.4f} \n").format(info['current_epoch'], info['epochs'], info['t_duration'],
                                                                info['tr_loss'],info['tr_acc'],info['tst_loss'],info['tst_acc'],)
    print(message)
    if log_fp:
        if not os.path.exists(log_fp):
            os.makedirs(log_fp)
        with open(log_fp+'/'+'log.txt', 'a') as log_file:
            log_file.write('{:s}\n'.format(message))


def run(model,train_loader,test_loader,epochs,optimizer,device=0,log_fp=None):

    if device == 0:
        device = torch.device('cpu')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    for epoch in range(1,epochs+1):
        t = time.time()
        train_loss,train_accuracy = train(model,train_loader,optimizer,device)
        t_duration = time.time() - t
        # Todo by haojie
        test_loss, test_accuracy = test(model,test_loader,device)
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            't_duration': t_duration,
            'tr_loss': train_loss,
            'tr_acc': train_accuracy,
            'tst_loss': test_loss,
            'tst_acc': test_accuracy,}
        print_info(info,log_fp)


model = PointNet(train_with_norm=True)
print(model)
torch.set_num_threads(1234)
torch.manual_seed(1234)
cudnn.benchmark = False
cudnn.deterministic = True
pytorch_total_params = sum(p.numel() for p in model.parameters())
train_dataset = Grasp_Dataset(root='./raw/foo',train=True, transform = GraspNormalization())
# remove the 10 below to use the entire dataset
train_loader = DataLoader(train_dataset[:10],batch_size=1,shuffle=False)
test_dataset = Grasp_Dataset(root='./raw/foo',train=False, transform = GraspNormalization())
test_loader = DataLoader(test_dataset[:10],batch_size=1,shuffle=False)
print(len(train_loader))
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4,weight_decay=1e-5)
run(model,train_loader,test_loader,500,optimizer,device=1)

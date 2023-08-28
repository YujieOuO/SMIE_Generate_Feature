from gen_config import *
from feeder import DataSet
from model import *
from logger import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm

from st_gcn import Model

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    
setup_seed(5277)

class BaseProcessor:

    @ex.capture
    def load_data(self,train_list,train_label,test_list,test_label,batch_size):
        self.dataset = dict()
        self.data_loader = dict()

        self.dataset['train'] = DataSet(train_list, train_label)
        self.dataset['test'] = DataSet(test_list, test_label)
        self.best_epoch = -1
        self.best_acc = -1
        self.loss = -1
        self.train_acc = -1
        self.test_acc = -1

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            num_workers=16,
            shuffle=True)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=batch_size,
            num_workers=16,
            shuffle=False)


    def initialize(self):
        self.load_data()
        self.load_model()
        self.load_optim()
        self.log = Log()
    
    @ex.capture
    def optimize(self, epoch_num):
        for epoch in range(epoch_num):
            self.epoch = epoch
            self.train_epoch()
            self.test_epoch()

    def save_model(self):
        
        pass

    def start(self):
        self.initialize()
        self.optimize()

class Processor(BaseProcessor):

    @ex.capture
    def load_model(self,in_channels,hidden_channels,hidden_dim,
                    dropout,graph_args,edge_importance_weighting):
        self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                            hidden_dim=hidden_dim,dropout=dropout, 
                            graph_args=graph_args,
                            edge_importance_weighting=edge_importance_weighting,
                            )
        self.encoder = self.encoder.cuda()
        self.classifier = Linear().cuda()
    
    @ex.capture
    def load_optim(self, lr, epoch):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.classifier.parameters()}],
            lr=lr,
            )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epoch)
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()

    @ex.capture
    def train_epoch(self):
        self.encoder.train()
        self.classifier.train()
        loader = self.data_loader['train']
        running_loss = []
        acc_list = []
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            acc, loss = self.train_batch(data, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss)
            acc_list.append(acc)
        
        self.loss = torch.tensor(running_loss).mean().item()
        self.train_acc = torch.tensor(acc_list).mean().item()
        self.scheduler.step()

    @ex.capture
    def train_batch(self, data, label):

        Z = self.encoder(data)
        predict = self.classifier(Z)
        _, pred = torch.max(predict, 1)
        acc = pred.eq(label.view_as(pred)).float().mean()
        loss = self.CrossEntropyLoss(predict, label)
        return acc, loss

    @ex.capture
    def test_epoch(self, epoch):
        self.encoder.eval()
        self.classifier.eval()
        acc_list = []

        loader = self.data_loader['test']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            with torch.no_grad():
                Z = self.encoder(data)
                predict = self.classifier(Z)

            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            acc_list.append(acc)
        
        self.test_acc = torch.tensor(acc_list).mean().item()
        if self.test_acc > self.best_acc:
            self.best_acc = self.test_acc
            self.best_epoch = epoch
            self.save_model(epoch)

    @ex.capture
    def optimize(self,epoch,_log):
        for ep in range(epoch):
            self.train_epoch()
            self.test_epoch(ep)
            print("epoch [{}] train loss: {}".format(ep,self.loss))
            print("epoch [{}] train acc: {}".format(ep,self.train_acc))
            print("epoch [{}] test acc: {}".format(ep,self.test_acc))
            print("epoch [{}] gets the best acc: {}".format(self.best_epoch,self.best_acc))
    
    @ex.capture
    def save_model(self,epoch,weight_path,_log):
        _log.info("model is saved.")
        torch.save(self.encoder.state_dict(), weight_path)

# %%
@ex.automain
def main():
    p = Processor()
    p.start()

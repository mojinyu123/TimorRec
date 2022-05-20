import numpy as np
import torch.nn as nn
from sklearn import metrics

softmax = nn.Softmax(dim = 1)

class auc_binary_class():
    def __init__(self):
        self.pred_list = []
        self.label_list = []
        self.metric = metrics.roc_auc_score
    
    def update(self, pred, label):
        pred = pred[:, 0].cpu().numpy()
        self.pred_list.extend(pred)
        self.label_list.extend(label.cpu().numpy())

    def cal(self):
        return self.metric(self.label_list, self.pred_list)
    
    def refresh(self):
        self.pred_list = []
        self.label_list = []

class auc_multi_class():
    def __init__(self):
        self.pred_list = []
        self.label_list = []
        self.metric = metrics.roc_auc_score
    
    def update(self, pred, label):
        pred = softmax(pred)
        pred = pred[:, 1].cpu().numpy()
        self.pred_list.extend(pred)
        self.label_list.extend(label.cpu().numpy())

    def cal(self):
        return round(self.metric(self.label_list, self.pred_list), 4)
    
    def refresh(self):
        self.pred_list = []
        self.label_list = []
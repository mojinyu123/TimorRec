from turtle import back
import torch
import torch.nn as nn
from .net import FMLayer
from utils.metric import *

class MultipleOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

class DygraphModel():
    def __init__(self, config) -> None:
        self.config = config

    def create_model(self):
        sparse_feature_number = self.config.sparse_feature_number
        sparse_feature_dim = self.config.sparse_feature_dim
        dense_feature_number = self.config.dense_feature_number
        model = FMLayer(sparse_feature_number, sparse_feature_dim, dense_feature_number)
        model.weight_init()
        return model
    
    def create_feeds(self, batch_data):
        label = batch_data[0].type(torch.float).to(self.config.device)
        sparse_tensor = torch.concat(batch_data[1:-1], axis=1).to(self.config.device)
        dense_tensor = batch_data[-1].reshape(-1, self.config.dense_feature_number).to(self.config.device)

        return sparse_tensor, dense_tensor, label

    def create_loss(self, pred, label):
        loss_func = nn.BCELoss()
        binary_loss = loss_func(pred, label)
        return binary_loss 

    def create_optimizer(self, model):
        lr = self.config.lr
        dense_param_list = []
        sparse_param_list = []
        for (name, param) in model.named_parameters():
            if 'sparse' in name: 
                sparse_param_list.append(param)
            else: 
                dense_param_list.append(param)

        optimizer_sparse = torch.optim.SparseAdam(sparse_param_list, lr=lr)
        optimizer_dense = torch.optim.Adam(dense_param_list, lr=lr)
        
        return MultipleOptimizer(optimizer_sparse, optimizer_dense)
    
    def create_metric(self):
        metrics_list_name = ["auc_binary"]
        auc_binary = auc_binary_class()
        metrics_list = [auc_binary]
        return metrics_list, metrics_list_name

    def dygraph_train(self, model, batch_data):
        sparse_tensor, dense_tensor, label = self.create_feeds(batch_data)
        pred = model(sparse_tensor, dense_tensor)
        binary_loss = self.create_loss(pred, label)
        return binary_loss
    
    def infer(self, model, batch_data, metrics_list):
        sparse_tensor, dense_tensor, label = self.create_feeds(batch_data)
        pred = model(sparse_tensor, dense_tensor)
        metrics_list[0].update(pred, label)
        return pred

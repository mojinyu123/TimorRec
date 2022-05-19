import torch
import torch.nn as nn
from .net import MMoELayer
from utils.metric import *


class DygraphModel():
    def create_model(self, config):
        feature_size = config.feature_size
        expert_num = config.expert_num
        expert_size = config.expert_size
        gate_num = config.gate_num
        tower_size = config.tower_size
        model = MMoELayer(feature_size, expert_num, expert_size, gate_num, tower_size)
        model.weight_init()
        return model
    
    def create_feeds(self, batch_data):
        input_data, label_income, label_marital = batch_data
        input_data = input_data.type(torch.float)
        label_income = label_income.type(torch.LongTensor)
        label_marital = label_marital.type(torch.LongTensor)
        return input_data, label_income, label_marital

    def create_loss(self, pred_income, pred_marital, label_income, label_marital):
        income_loss_func = nn.CrossEntropyLoss()
        marital_loss_func = nn.CrossEntropyLoss()
        income_loss = income_loss_func(pred_income, label_income)
        marital_loss = marital_loss_func(pred_marital, label_marital)

        return marital_loss + income_loss

    def create_optimizer(self, model, config):
        lr = config.lr
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return optimizer
    
    def create_metric(self,):
        metrics_list_name = ["auc_income", "auc_marital"]
        auc_income_metric = auc_class()
        auc_marital_metric = auc_class()
        metrics_list = [auc_income_metric, auc_marital_metric]
        return metrics_list, metrics_list_name

    def dygraph_train(self, model, batch_data):
        input_data, label_income, label_marital = self.create_feeds(batch_data)
        pred_income, pred_marital = model(input_data)
        loss = self.create_loss(pred_income, pred_marital, label_income, label_marital)
        return loss
    
    def infer(self, model, batch_data, metrics_list):
        input_data, label_income, label_marital = self.create_feeds(batch_data)
        pred_income, pred_marital = model(input_data)

        metrics_list[0].update(pred_income, label_income)
        metrics_list[1].update(pred_marital, label_marital)

        return pred_income, pred_marital,
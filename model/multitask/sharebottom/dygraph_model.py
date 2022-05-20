import torch
import torch.nn as nn
from .net import ShareBottomLayer
from utils.metric import *


class DygraphModel():
    def __init__(self, config) -> None:
        self.config = config

    def create_model(self):
        feature_size = self.config.feature_size
        task_num = self.config.task_num
        bottom_size = self.config.bottom_size
        tower_size = self.config.tower_size
        model = ShareBottomLayer(feature_size, task_num, bottom_size, tower_size)
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


    def create_optimizer(self, model):
        lr = self.config.lr
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return optimizer
    
    def create_metric(self):
        metrics_list_name = ["auc_income", "auc_marital"]
        auc_income_metric = auc_multi_class()
        auc_marital_metric = auc_multi_class()
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
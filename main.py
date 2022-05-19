import torch
import torch.nn as nn
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.args import *
from utils.tool import *
# from model.multitask.sharebottom.dygraph_model import *
# from model.multitask.sharebottom.census_reader import *
from model.multitask.mmoe.dygraph_model import *
from model.multitask.sharebottom.census_reader import *


if __name__ == "__main__":
    setup_seed(2022)

    # args = share_bottom_layer_args()
    args = mmoe_layer_args()


    data_path = "dataset/"
    train_data_path = data_path + args.dataset + "/train_data.csv"
    test_data_path = data_path + args.dataset + "/test_data.csv"

    train_data = CensusDataset(train_data_path)
    test_data = CensusDataset(test_data_path)
    train_data_loader = DataLoader(train_data, batch_size = args.bs, num_workers=2, drop_last=True, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size = args.bs, num_workers=2, shuffle=False)
    print("train data shape:%d" % len(train_data))
    print("val data shape:%d" % len(test_data))

    dy_model_class = DygraphModel()
    model = dy_model_class.create_model(args)
    model.to(args.device)
    
    optimizer = dy_model_class.create_optimizer(model, args)
    metrics_list, metrics_list_name = dy_model_class.create_metric()

    interval = 5
    # train 
    model.train()
    for epoch in range(args.epoch):
        loss_epoch = 0
        start_time = time.time()
        for idx, bacth in enumerate(train_data_loader):
            optimizer.zero_grad()
            loss = dy_model_class.dygraph_train(model, bacth)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.cpu().item()
        
        spend_time = time.time() - start_time

        # test
        if epoch % interval == 0:
            model.eval()
            with torch.no_grad():
                for idx, bacth in enumerate(train_data_loader):
                    dy_model_class.infer(model, bacth, metrics_list)
                
                print("Epoch:%d--------train data metric---------:"%epoch)
                for i, metric_name in enumerate(metrics_list_name):
                    print(metric_name + ":" + str(metrics_list[i].cal()), end='; ')
                    metrics_list[i].refresh()
                print("\n")

                for idx, bacth in enumerate(test_data_loader):
                    dy_model_class.infer(model, bacth, metrics_list)
                
                print("Epoch:%d--------val data metric---------:"%epoch)
                for i, metric_name in enumerate(metrics_list_name):
                    print(metric_name + ":" + str(metrics_list[i].cal()), end='; ')
                    metrics_list[i].refresh()
                print("\n")

                

        print("Epoch:{}, loss:{}, spend time: {}".
            format(epoch, loss_epoch / len(train_data_loader), spend_time))

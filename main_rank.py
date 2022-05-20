import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from utils.args import *
from utils.tool import *
from model.rank.fm.dygraph_model import *
from model.rank.fm.criteo_reader import *


if __name__ == "__main__":
    setup_seed(2022)
    args = fm_layer_args()
    data_path = "dataset/"
    train_data_path = data_path + args.dataset + "/train"
    test_data_path = data_path + args.dataset + "/test"    

    train_data = RecDataset(train_data_path)
    test_data = RecDataset(test_data_path)
    train_data_loader = DataLoader(train_data, batch_size = args.bs, num_workers=2, drop_last=True, shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size = args.bs, num_workers=2, shuffle=False)
    
    dy_model_class = DygraphModel(args)
    model = dy_model_class.create_model()
    model.to(args.device)
    
    optimizer = dy_model_class.create_optimizer(model)
    metrics_list, metrics_list_name = dy_model_class.create_metric()

    print("Begin run.......")
    interval = 2
    # train 
    model.train()
    for epoch in range(args.epoch):
        loss_epoch = 0
        batch_num = 0
        start_time = time.time()
        for idx, bacth in enumerate(train_data_loader):
            optimizer.zero_grad()
            loss = dy_model_class.dygraph_train(model, bacth)
            loss.backward()
            optimizer.step()
            
            loss_epoch += loss.cpu().item()
            batch_num += 1
        spend_time = time.time() - start_time

        # test
        if epoch % interval == 0:
            model.eval()
            with torch.no_grad():
                for idx, bacth in enumerate(train_data_loader):
                    dy_model_class.infer(model, bacth, metrics_list)
                
                print("--------train data metric Epoch:%d---------:"%epoch)
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

        
        print("Epoch:{}, loss:{:.4f}, spend time: {:.2f}".
            format(epoch, loss_epoch / batch_num, spend_time))

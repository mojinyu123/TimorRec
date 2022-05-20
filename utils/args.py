import argparse
from constant import *

def data_preparation_args():
    parser = argparse.ArgumentParser(description="data preparation")
    parser.add_argument('--train_path', type=str, default='./train_data.csv')
    parser.add_argument('--test_path', type=str, default='./test_data.csv')         
    parser.add_argument('--train_data_path', type=str, default='./')
    parser.add_argument('--test_data_path', type=str, default='./')
    return parser.parse_args()

def share_bottom_layer_args():
    # feature_size, task_num, bottom_size, tower_size
    parser = argparse.ArgumentParser(description="ShareBottomLayer") 
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--feature_size', type=int, default=499)
    parser.add_argument('--task_num', type=int, default=2)
    parser.add_argument('--bottom_size', type=int, default=117)
    parser.add_argument('--tower_size', type=int, default=8)

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--bs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dataset', type=str, default="census")

    return parser.parse_args()

def mmoe_layer_args():
    # feature_size, task_num, bottom_size, tower_size
    parser = argparse.ArgumentParser(description="mmoe") 
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--feature_size', type=int, default=499)
    parser.add_argument('--gate_num', type=int, default=2)
    parser.add_argument('--expert_num', type=int, default=8)
    parser.add_argument('--expert_size', type=int, default=16)
    parser.add_argument('--tower_size', type=int, default=8)

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--bs', type=int, default=100)

    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dataset', type=str, default="census")

    return parser.parse_args()

def fm_layer_args():
    # feature_size, task_num, bottom_size, tower_size
    parser = argparse.ArgumentParser(description="fm") 
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sparse_feature_number', type=int, default=10000001)
    parser.add_argument('--sparse_feature_dim', type=int, default=9)
    parser.add_argument('--dense_feature_number', type=int, default=13)

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--bs', type=int, default=4096)

    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dataset', type=str, default="criteo")

    return parser.parse_args()

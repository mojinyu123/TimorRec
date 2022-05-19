import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ShareBottomLayer(nn.Module):
    def __init__(self, feature_size, task_num, bottom_size, tower_size) -> None:
        super().__init__()
        self.task_num = task_num
        self._param_bottom = nn.Sequential(
            nn.Linear(feature_size, bottom_size),
            nn.ReLU()
        )

        self._param_tower = nn.ModuleList()
        self._param_tower_out = nn.ModuleList()
        for i in range(task_num):
            linear = nn.Sequential(
                nn.Linear(bottom_size, tower_size),
                nn.ReLU()
            )
            self._param_tower.append(linear)
        
            linear = nn.Sequential(
                nn.Linear(tower_size, 2)
            )

            self._param_tower_out.append(linear)


    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=0, b=1)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, input_data):
        bottom_out = self._param_bottom(input_data)
        output_layer = []
        for i in range(self.task_num):
            cur_tower = self._param_tower[i](bottom_out)
            out_tower = self._param_tower_out[i](cur_tower)
            output_layer.append(out_tower)
        return output_layer

    

if __name__ == "__main__":
    feature_size = 20
    task_num = 2
    bottom_size = 10 
    tower_size = 2

    a = ShareBottomLayer(feature_size, task_num, bottom_size, tower_size)
    a.weight_init()

    summary(a, input_size = (1, 20))

    test_data = torch.tensor([[1.0 for i in range(20)]])
    res = a.forward(test_data)
    print(res)

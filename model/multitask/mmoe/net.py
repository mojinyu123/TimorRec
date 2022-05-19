import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class MMoELayer(nn.Module):
    def __init__(self, feature_size, expert_num, expert_size, gate_num, tower_size) -> None:
        super().__init__()
        self.gate_num = gate_num
        self.expert_num = expert_num
        self.expert_size = expert_size
        self._param_expert = nn.ModuleList()
        self._param_gate = nn.ModuleList()

        
        for i in range(expert_num):
            linear = nn.Sequential(
                nn.Linear(feature_size, expert_size),
                nn.ReLU()
            )
            self._param_expert.append(linear)

        self._param_tower = nn.ModuleList()
        self._param_tower_out = nn.ModuleList()
        for i in range(gate_num):
            linear = nn.Sequential(
                nn.Linear(expert_size, tower_size),
                nn.ReLU()
            )
            self._param_tower.append(linear)
        
            linear = nn.Sequential(
                nn.Linear(tower_size, 2)
            )

            self._param_tower_out.append(linear)

            linear = nn.Sequential(
                nn.Linear(feature_size, expert_num),
                nn.Softmax(dim=1)
            )
            self._param_gate.append(linear)


    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, input_data):
        expert_out = []
        for i in range(self.expert_num):
            expert_tmp = self._param_expert[i](input_data)
            expert_out.append(expert_tmp)
        
        expert_concat = torch.concat(expert_out, axis=1)
        expert_concat = torch.reshape(expert_concat, (-1, self.expert_num, self.expert_size))
        output_layer = []
        for i in range(self.gate_num):
            gate_tmp = self._param_gate[i](input_data)
            gate_out = torch.reshape(gate_tmp, (-1, self.expert_num, 1))

            expert_gate_out = expert_concat.mul(gate_out)
            expert_gate_out = torch.sum(expert_gate_out, dim=1)
            
            cur_tower = self._param_tower[i](expert_gate_out)
            out_tower = self._param_tower_out[i](cur_tower)
            output_layer.append(out_tower)
        
        return output_layer

    

if __name__ == "__main__":
    feature_size = 20
    expert_num = 2
    expert_size = 8
    gate_num = 2

    tower_size = 5

    a = MMoELayer(feature_size, expert_num, expert_size, gate_num, tower_size)
    a.weight_init()

    # summary(a, input_size = (1, 20))

    test_data = torch.tensor([[1.0 for i in range(20)] for j in range(10)])
    res = a.forward(test_data)
    # print(res)

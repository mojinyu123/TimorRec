import numpy as np
import os
from torch.utils.data import IterableDataset, DataLoader


class RecDataset(IterableDataset):
    def __init__(self, file_path):
        super(RecDataset, self).__init__()
        self.file_path = file_path
        self.file_list = os.listdir(file_path)
        self.init()

    def init(self):
        padding = 0
        sparse_slots = "click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"
        self.sparse_slots = sparse_slots.strip().split(" ")
        self.dense_slots = ["dense_feature"]
        self.dense_slots_shape = 13
        self.slots = self.sparse_slots + self.dense_slots
        self.slot2index = {}
        self.visit = {}
        for i in range(len(self.slots)):
            self.slot2index[self.slots[i]] = i
            self.visit[self.slots[i]] = False
        self.padding = padding

    def __iter__(self):
        self.data = []
        for file in self.file_list:
            with open(os.path.join(self.file_path, file), "r") as rf:
                for l in rf:
                    line = l.strip().split(" ")
                    output = [(i, []) for i in self.slots]
                    for i in line:
                        slot_feasign = i.split(":")
                        slot = slot_feasign[0]
                        if slot not in self.slots:
                            continue
                        if slot in self.sparse_slots:
                            if slot_feasign[1] != " ":
                                feasign = int(slot_feasign[1])
                            else:
                                feasign = self.padding
                        else:
                            if slot_feasign[1] != " ":
                                feasign = float(slot_feasign[1])
                            else:
                                feasign = float(self.padding)
                        output[self.slot2index[slot]][1].append(feasign)
                        self.visit[slot] = True

                    # fill none value by zero
                    for i in self.visit:
                        slot = i
                        if not self.visit[slot]:
                            if i in self.dense_slots:
                                output[self.slot2index[i]][1].extend(
                                    [float(self.padding)] * self.dense_slots_shape)
                            else:
                                output[self.slot2index[i]][1].extend(
                                    [self.padding])
                        else:
                            self.visit[slot] = False

                    # sparse
                    output_list = []
                    for key, value in output[:-1]:
                        output_list.append(np.array(value).astype('int64'))
                    # dense
                    output_list.append(
                        np.array(output[-1][1]).astype("float32"))
                    # list
                    yield output_list



if __name__ == "__main__":
    file_path = "../../../dataset/criteo/train"
    dataset = RecDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=2, drop_last=True, num_workers=2)
    for i, data in enumerate(dataloader):
        print(i, end=" ")


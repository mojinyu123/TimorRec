import numpy as np
from torch.utils.data import IterableDataset, DataLoader


class RecDataset(IterableDataset):
    def __init__(self, file_list):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.init()

    def init(self):
        padding = 0
        sparse_slots = "click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"
        self.sparse_slots = sparse_slots.strip().split(" ")
        self.dense_slots = ["dense_feature"]
        self.dense_slots_shape = [13]
        self.slots = self.sparse_slots + self.dense_slots
        self.slot2index = {}
        self.visit = {}
        for i in range(len(self.slots)):
            self.slot2index[self.slots[i]] = i
            self.visit[self.slots[i]] = False
        self.padding = padding

    def __iter__(self):
        full_lines = []
        self.data = []
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    line = l.strip().split(" ")
                    output = [(i, []) for i in self.slots]
                    for i in line:
                        slot_feasign = i.split(":")
                        slot = slot_feasign[0]
                        if slot not in self.slots:
                            continue
                        if slot in self.sparse_slots:
                            feasign = int(slot_feasign[1])
                        else:
                            feasign = float(slot_feasign[1])
                        output[self.slot2index[slot]][1].append(feasign)
                        self.visit[slot] = True
                    for i in self.visit:
                        slot = i
                        if not self.visit[slot]:
                            if i in self.dense_slots:
                                output[self.slot2index[i]][1].extend(
                                    [self.padding] *
                                    self.dense_slots_shape[self.slot2index[i]])
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
    file_list = ["../../../dataset/criteo/test/part-220"]
    dataset = RecDataset(file_list)
    dataloader = DataLoader(dataset, batch_size=1, drop_last=True, num_workers=2)
    for data in dataloader:
        print(data)
        break
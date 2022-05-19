import pandas as pd
from torch.utils.data import Dataset, DataLoader

class CensusDataset(Dataset):
    def __init__(self, file):
        self.data = pd.read_csv(file, header=None).to_numpy()
        self.length = self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index][2:].astype("float32"), self.data[index][1].astype("int64"), self.data[index][0].astype("int64"), 

    def __len__(self):
        return self.length




if __name__ == "__main__":
    file_path = "../../../dataset/census/test_data.csv"
    dataset = CensusDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=3, drop_last=True, num_workers=2, shuffle=True)
    for data in dataloader:
        print(data)
        break
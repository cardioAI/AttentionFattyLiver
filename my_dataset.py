import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MyDataSet(Dataset):

    def __init__(self, impedance_path: list, impedance_class: list):
        self.impedance_path = impedance_path
        self.impedance_class = impedance_class

    def __len__(self):
        return len(self.impedance_path)

    def __getitem__(self, item):
        impedance = np.load(self.impedance_path[item], allow_pickle=True)
        impedance = impedance.astype(float)
        # np.expand_dims(impedance, axis=0)
        # impedance = impedance.transpose(2, 0, 1)
        trans = transforms.Compose([transforms.ToTensor()])
        trans(impedance)

        impedance = torch.from_numpy(impedance).unsqueeze(0)
        label = self.impedance_class[item]
        label = torch.as_tensor(int(label))
        print(label)
        return impedance, label


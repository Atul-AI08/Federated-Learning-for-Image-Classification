from PIL import Image
import torch
from torch.utils import data
import pandas as pd
import os


class MyDataset(data.Dataset):
    def __init__(
        self, root_dir, csv_path, dataset="tumor", transform=None, dataidxs=None
    ):
        super(MyDataset, self).__init__()

        data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.data = data["id_code"]
        self.labels = data["diagnosis"]

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

        self.dataidxs = dataidxs
        if dataset == "tumor":
            self.ext = ".jpg"
        elif dataset == "covid":
            self.ext = ".png"
        else:
            raise ("Invalid Dataset")

        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.labels = self.labels[self.dataidxs]
            self.data = self.data.reset_index(drop=True)
            self.labels = self.labels.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.data[idx] + self.ext)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]

        return image, torch.tensor(label)

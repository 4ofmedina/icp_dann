import os, io
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch


class BloodFlow(Dataset):
    """Blood Flow dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        df = pd.read_csv(os.path.join(root_dir, csv_file))
        self.dataframe = df.fillna(0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # column 5 corresponds to ICP,
        # columns 6 to end corresponds to blood flow
        icp = self.dataframe.iloc[idx, 5].astype(np.float32)  # icp = self.dataframe['ICP'].values[idx].astype(np.float32)
        flow = np.array([self.dataframe.iloc[idx, 6:]]).astype(np.float32)
        flow = (flow-flow.min())/(flow.max()-flow.min())

        sample = {'x': flow, 'y': icp}

        if self.transform:
            sample = self.transform(sample)

        return sample['x'], sample['y']


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        flow, icp = sample['x'], sample['y']
        return {'x': torch.from_numpy(flow),
                'y': icp
                }

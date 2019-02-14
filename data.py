import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CameraDataset(Dataset):
    """Camera position dataset."""

    def __init__(self, csv_file, img_dir, transform=None):
        """
        Parameters
        ----------
        csv_file : string
            Path to the csv file with annotations.
        root_dir : string
            Directory with all the images.
        transform : callable, optional
            Optional transform to be applied
            on a sample.
        """

        self.dataframe = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir,
                                self.dataframe['ImageFile'].at[idx])
        image = Image.open(img_name)
        coords = self.dataframe.iloc[idx, 3:10].values
        coords = np.array(coords, dtype=np.float32)

        if self.transform:
            image = self.transform(image)
            coords = torch.tensor(coords, dtype=torch.float)

        sample = {'image': image, 'coords': coords}

        return sample


class ConvertPILMode():
    """Converts PIL image to
        one of the supported modes:
        - `L` - 1 channel - (grayscale)
        - `RGB` - 3 channels
        - `RGBA` - 4 channels
        """

    def __init__(self, mode):
        self.mode = mode

    def __call__(self, pil_image):
        return pil_image.convert(mode=self.mode)


def train_val_holdout_split(dataset, ratios=[0.7, 0.2, 0.1]):
    """Return indices for subsets of the dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset made with class which inherits `torch.utils.data.Dataset`
    ratios : list of floats
        List of [train, val, holdout] ratios respectively. Note, that sum of 
        values must be equal to 1. (train + val + holdout = 1.0)

    """

    assert np.allclose(ratios[0] + ratios[1] + ratios[2], 1)
    train_ratio = ratios[0]
    val_ratio = ratios[1]
    test_ratio = ratios[2]

    df_size = len(dataset)
    train_inds = np.random.choice(range(df_size), 
                                  size=int(df_size*train_ratio),
                                  replace=False)
    val_test_inds = list(set(range(df_size)) - set(train_inds))
    val_inds = np.random.choice(val_test_inds,
                                size=int(len(val_test_inds)*val_ratio/(val_ratio+test_ratio)),
                                replace=False)

    test_inds = np.asarray(list(set(val_test_inds) - set(val_inds)), dtype='int')

    assert len(list(set(train_inds) - set(val_inds) - set(test_inds))) == len(train_inds)

    return train_inds, val_inds, test_inds

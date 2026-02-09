import torch
from torch.utils.data import Dataset
from MNIST_Dataset import MNIST_Dataset

class TorchMNIST(Dataset):
    def __init__(self, folder_name: str, train: bool = True):
        self.dataset = MNIST_Dataset(folder_name)
        self.images = (
            self.dataset.training_images
            if train
            else self.dataset.testing_images
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        # Convert to torch tensors
        x = torch.tensor(img.pixels, dtype=torch.float32)
        y = torch.tensor(img.label, dtype=torch.long)

        return x, y

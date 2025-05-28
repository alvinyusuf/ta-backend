# di file datasets.py misal:
from torch.utils.data import Dataset
from PIL import Image

class InMemoryDataset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        return self.transform(img), idx

    def __len__(self):
        return len(self.images)

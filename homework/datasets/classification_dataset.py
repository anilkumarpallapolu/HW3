import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from homework.models import INPUT_MEAN, INPUT_STD  # Import these from models.py

LABEL_NAMES = ["background", "kart", "pickup", "nitro", "bomb", "projectile"]


class SuperTuxDataset(Dataset):
    """
    SuperTux dataset for classification
    """

    def __init__(self, dataset_path: str, transform_pipeline: str = "default"):
        self.transform = self.get_transform(transform_pipeline)
        self.data = []
        print('dataset_path',dataset_path)
        with open(Path(dataset_path, "labels.csv"), newline="") as f:
            for fname, label, _ in csv.reader(f):
                if label in LABEL_NAMES:
                    img_path = Path(dataset_path, fname)
                    label_id = LABEL_NAMES.index(label)

                    self.data.append((img_path, label_id))

    def get_transform(self, transform_pipeline: str = "default"):
        if transform_pipeline == "default":
            xform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=INPUT_MEAN, std=INPUT_STD),
            ])
        elif transform_pipeline == "aug":
            xform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=INPUT_MEAN, std=INPUT_STD),
            ])
        else:
            raise ValueError(f"Invalid transform {transform_pipeline} specified!")

        return xform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label_id = self.data[idx]
        img = Image.open(img_path)
        data = (self.transform(img), label_id)
        return data


def load_data(
    dataset_path: str,
    transform_pipeline: str = "default",
    return_dataloader: bool = True,
    num_workers: int = 4,
    batch_size: int = 128,
    shuffle: bool = False,
) -> DataLoader | Dataset:
    """
    Constructs the dataset/dataloader.
    The specified transform_pipeline must be implemented in the SuperTuxDataset class.

    Args:
        transform_pipeline (str): 'default', 'aug', or other custom transformation pipelines
        return_dataloader (bool): returns either DataLoader or Dataset
        num_workers (int): data workers, set to 0 for VSCode debugging
        batch_size (int): batch size
        shuffle (bool): should be true for train and false for val

    Returns:
        DataLoader or Dataset
    """
    dataset = SuperTuxDataset(dataset_path, transform_pipeline=transform_pipeline)

    if not return_dataloader:
        return dataset

    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )

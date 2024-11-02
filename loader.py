from typing import Tuple
from torch import Generator
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from hyperparams import INPUT_SIZE, BATCH_SIZE


def load_data() -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Init variables and constants
    train_loader, val_loader, test_loader = None, None, None
    # Download dataset from: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
    SOURCE = "brain_tumor_mri_2d_4class\\{purpose}"

    # Prepare transforms
    transform_default = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize((0.5), (0.5)),
        ]
    )
    transform_horizontal_flip = transforms.Compose(
        [
            transform_default,
            transforms.RandomHorizontalFlip(p=1),
        ]
    )

    # Prepare training data
    trainset = ImageFolder(
        root=SOURCE.format(purpose="Training"),
        transform=transform_default,
    )
    trainset += ImageFolder(
        root=SOURCE.format(purpose="Training"),
        transform=transform_horizontal_flip,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    # Prepare validation and test data
    testing = ImageFolder(
        root=SOURCE.format(purpose="Testing"),
        transform=transform_default,
    )
    testing += ImageFolder(
        root=SOURCE.format(purpose="Testing"),
        transform=transform_horizontal_flip,
    )

    test_count = int(len(testing) * 0.3)
    val_count = len(testing) - test_count

    test_set, val_set = random_split(
        testing,
        [test_count, val_count],
        generator=Generator().manual_seed(42),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    return (train_loader, val_loader, test_loader)

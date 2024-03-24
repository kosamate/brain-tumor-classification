import os
import pickle
from typing import Tuple, List
from torch import Generator
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from hyperparams import INPUT_SIZE, BATCH_SIZE


def load_data(force_reread=False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Init variables and constants
    train_loader, val_loader, test_loader = None, None, None
    FILE_TEMPLATE = "D:\\Project\\Dipterv\\Cache\\{file}.pickle"
    OUTPUTS = {
        "train_dl": train_loader,
        "val_dl": val_loader,
        "test_dl": test_loader,
    }
    SOURCE = "D:\\Project\\Dipterv\\Datasets\\brain_tumor_mri_2d_4class\\{purpose}"

    # Check the saved obejects
    if force_reread is False and all(os.path.exists(FILE_TEMPLATE.format(file=f)) for f in OUTPUTS):
        result: List[DataLoader] = []
        for file in OUTPUTS:
            with open(FILE_TEMPLATE.format(file=file), "rb") as in_file:
                result.append(pickle.load(in_file))
        return result[0], result[1], result[2]

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

    # Save the results in pickle files
    for file, obj in OUTPUTS.items():
        f = FILE_TEMPLATE.format(file=file)
        with open(f, "wb") as out:
            pickle.dump(obj, out)
    return (train_loader, val_loader, test_loader)

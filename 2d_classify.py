import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from typing import Tuple

CLASSES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor",
]
BATCH_SIZE = 4


def main():
    load_data()
    pass


def load_data() -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.ImageFolder(
        root="D:\\Project\\Dipterv\\Datasets\\brain_tumor_mri_2d_4class\\Training\\",
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    testset = torchvision.datasets.ImageFolder(
        root="D:\\Project\\Dipterv\\Datasets\\brain_tumor_mri_2d_4class\\Testing",
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )


def show_img(img: torch.Tensor, label_idx):
    plt.imshow(img.permute(1, 2, 0))
    plt.title(CLASSES[label_idx])
    plt.show()


if __name__ == "__main__":
    main()

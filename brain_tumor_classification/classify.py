from typing import Tuple, List
import time
import pickle
import os

from torch import Generator, optim, Tensor, no_grad
from torch.nn import Module as CNN
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import TumorClassificationCNN
from hyperparams import BATCH_SIZE, LEARNING_RATE, EPOCHS
from visualize import show_sample


def main():
    train_dl, val_dl, test_dl = load_data()
    model = TumorClassificationCNN()
    train_h, val_h = train(
        model,
        train_dl,
        val_dl,
        BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
    )


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
    if force_reread is False and all(
        os.path.exists(FILE_TEMPLATE.format(file=f)) for f in OUTPUTS
    ):
        result: List[DataLoader] = []
        for file in OUTPUTS:
            with open(FILE_TEMPLATE.format(file=file), "rb") as in_file:
                result.append(pickle.load(in_file))
        return result[0], result[1], result[2]

    # Prepare transforms
    transform_default = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=1),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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


def create_loss_and_optimizer(net: CNN, learning_rate=0.001):
    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    return criterion, optimizer


def train(
    net: CNN,
    train_dl: DataLoader,
    val_dl: DataLoader,
    batch_size: int,
    epochs: int,
    learning_rate: float,
):
    print("----Hyperparameters----")
    print(f"batch_size = {batch_size}")
    print(f"epochs = {epochs}")
    print(f"learning_rate = {learning_rate}")
    print()

    batches = len(train_dl)
    val_batches = len(val_dl)
    criterion, optimizer = create_loss_and_optimizer(net, learning_rate)

    # Init variables used for plotting the loss
    train_history = []
    val_history = []
    training_start_time = int(time.time())
    for epoch in range(epochs):
        # loop over the dataset multiple times
        running_loss = 0.0
        log_period = batches // 10
        total_train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dl):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss: Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_train_loss += loss.item()
            # print every 10th of epoch
            if (i + 1) % (log_period + 1) == 0:
                delta_time = time.time() - training_start_time
                print(
                    "Epoch {epoch}, {progress:.0f}% \t train_loss: {loss:.2f} \t time: {m} minutes {s} seconds".format(
                        epoch=epoch + 1,
                        progress=i / batches,
                        loss=running_loss / log_period,
                        m=delta_time // 60,
                        s=delta_time % 60,
                    )
                )
                running_loss = 0.0
        train_history.append(total_train_loss / batches)
        total_val_loss = 0.0
        # Do a pass on the validation set# We don't need to compute gradient,
        # we save memory and computation using th.no_grad()
        with no_grad():
            for inputs, labels in val_dl:
                # Forward pass
                predictions = net(inputs)
                val_loss: Tensor = criterion(predictions, labels)
                total_val_loss += val_loss.item()
                # TODO Túltanulás kezelés: earliy stopping

        val_history.append(total_val_loss / val_batches)
        print(f"Validation loss = {total_val_loss/val_batches:.2f}")
    delta_time = time.time() - training_start_time
    print(
        f"Training Finished, took {delta_time // 60} minutes {delta_time % 60} seconds"
    )
    return train_history, val_history


if __name__ == "__main__":
    main()

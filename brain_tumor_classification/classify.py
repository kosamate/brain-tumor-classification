from torch import Generator, optim, Tensor, no_grad
from torch.nn import Module as CNN
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from typing import Tuple
import time
from model import TumorClassificationCNN
from hyperparams import BATCH_SIZE


def main():
    train_dl, val_dl, test_dl = load_data()
    model = TumorClassificationCNN()
    train(model, train_dl, val_dl, BATCH_SIZE, epochs=10, learning_rate=0.001)


def load_data() -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = ImageFolder(
        root="D:\\Project\\Dipterv\\Datasets\\brain_tumor_mri_2d_4class\\Training\\",
        transform=transform,
    )
    train_loader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    testing = ImageFolder(
        root="D:\\Project\\Dipterv\\Datasets\\brain_tumor_mri_2d_4class\\Testing",
        transform=transform,
    )
    test_count = len(testing) * 0.7
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


def createLossAndOptimizer(net: CNN, learning_rate=0.001):
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
    criterion, optimizer = createLossAndOptimizer(net, learning_rate)

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
            optimizer.step()  # print statistics
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
        # Do a pass on the validation set# We don't need to compute gradient,# we save memory and computation using th.no_grad()
        with no_grad():
            for inputs, labels in val_dl:
                # Forward pass
                predictions = net(inputs)
                val_loss: Tensor = criterion(predictions, labels)
                total_val_loss += val_loss.item()

        val_history.append(total_val_loss / val_batches)
        print(f"Validation loss = {total_val_loss/val_batches:.2f}")
    delta_time = time.time() - training_start_time
    print(
        f"Training Finished, took {delta_time // 60} minutes {delta_time % 60} seconds"
    )
    return train_history, val_history


if __name__ == "__main__":
    main()

import torch
import time
from torch import Tensor, no_grad, optim
from torch.utils.data.dataloader import DataLoader
import model


def create_loss_and_optimizer(net: model.TumorClassification, learning_rate=0.001):
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    return criterion, optimizer


def train(
    net: model.TumorClassification,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int,
    learning_rate: float,
) -> tuple[list[float], list[float]]:

    batches = len(train_dl)
    val_batches = len(val_dl)
    criterion, optimizer = create_loss_and_optimizer(net, learning_rate)

    # Init variables used for plotting the loss
    train_history = []
    val_history = []
    training_start_time = int(time.time())
    overfitting = False
    best = {"epoch": 0, "loss": 1000}

    total_val_loss = 0.0
    # Do a pass on the validation set# We don't need to compute gradient,
    # we save memory and computation using th.no_grad()
    with no_grad():
        net.eval()
        for inputs, labels in val_dl:
            # Forward pass
            predictions = net(inputs)
            val_loss: Tensor = criterion(predictions, labels)
            total_val_loss += val_loss.item()
    net.train()
    val_loss_h = total_val_loss / val_batches
    print(f"Validation loss = {val_loss_h:.4f}")

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
                    "Epoch {epoch}, {progress:.0f}% \t train_loss: {loss:.4f} \t time: {m:.0f} minutes {s:.0f} seconds".format(
                        epoch=epoch + 1,
                        progress=i / batches * 100,
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
            net.eval()
            for inputs, labels in val_dl:
                # Forward pass
                predictions = net(inputs)
                val_loss: Tensor = criterion(predictions, labels)
                total_val_loss += val_loss.item()
        net.train()
        val_loss_h = total_val_loss / val_batches

        val_history.append(val_loss_h)
        print(f"Validation loss = {val_loss_h:.4f}")

        torch.save(net, f"model{epoch}")

        if best["loss"] > val_loss_h:
            best["loss"] = val_loss_h
            best["epoch"] = epoch

        if epoch > 4:
            if (
                val_history[-1] > val_history[-2]
                and val_history[-1] > val_history[-3]
                and val_history[-1] > val_history[-4]
            ):
                overfitting = True
                break  # stop the training if overfitting
    if overfitting:
        net = torch.load(f"model{best["epoch"]}")
    delta_time = time.time() - training_start_time
    print(f"Training Finished, took {delta_time // 60:.0f} minutes {delta_time % 60:.0f} seconds")
    return train_history, val_history

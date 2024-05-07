import numpy as np
import torch

from torch.nn import Module as CNN
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader

from hyperparams import BATCH_SIZE, LEARNING_RATE, EPOCHS, CLASSES, MODEL, print_hyperparams
from visualize import show_sample, save_classification_plot, save_train_res_plot
from train import train
from loader import load_data
from redirect import Redirect


def main():
    with Redirect(bypass=True) as test_case_number:
        print_hyperparams()
        train_dl, val_dl, test_dl = load_data()
        print(f"Data has been loaded. Batches: train: {len(train_dl)} val: {len(val_dl)} test: {len(test_dl)}")
        model = MODEL
        train_h, val_h = train(
            model,
            train_dl,
            val_dl,
            BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
        )
        test(model, test_dl, test_case_number)
        save_train_res_plot(train_h, val_h, test_case_number)


def peek():
    train_dl, val_dl, test_dl = load_data()
    for batch in train_dl:
        for i in range(0, BATCH_SIZE):
            show_sample(batch[0][i], batch[1][i])


def test(model: CNN, test_loader: DataLoader, test_case_number: int):
    # tracking test loss
    print("Testing model")
    test_loss = 0.0
    class_correct = [0.0] * len(CLASSES)
    class_total = [0.0] * len(CLASSES)
    points = np.array([[0] * len(CLASSES) for _ in range(len(CLASSES))], np.int32)

    model.eval()  # Prepare model for testing
    criterion = CrossEntropyLoss()

    for data, target in test_loader:
        # forward pass
        output = model(data)
        # batch loss
        loss = criterion(output, target)
        # test loss update
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
            points[int(pred.data[i]), int(label)] += 1

    # average test loss
    test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    for i, label in enumerate(CLASSES):
        if class_total[i] > 0:
            print(f"Test Accuracy of {label}: {100*class_correct[i]/class_total[i]:.3f}%")
        else:
            print(f"Test Accuracy of {label}s: N/A (no training examples)")

    print(
        f"Full Test Accuracy: {round(100. * np.sum(class_correct) / np.sum(class_total), 2)}% {np.sum(class_correct)} out of {np.sum(class_total)}"
    )
    save_classification_plot(points, test_case_number)


if __name__ == "__main__":
    main()

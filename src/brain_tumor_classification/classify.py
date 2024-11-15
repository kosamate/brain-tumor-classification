import argparse
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from brain_tumor_classification import hyperparams
from brain_tumor_classification import draw
from brain_tumor_classification.train import train
from brain_tumor_classification.loader import load_data
from brain_tumor_classification.model import TC_Final
from brain_tumor_classification.redirect import Redirect


def main(params: hyperparams.Hyperparameter):
    with Redirect(params.result_path) as test_case_number:
        drawer = draw.Drawer(params)
        print(params)
        train_dl, val_dl, test_dl = load_data(params)
        print(f"Data has been loaded. Batches: train: {len(train_dl)} val: {len(val_dl)} test: {len(test_dl)}")
        train_h, val_h = train(
            params,
            train_dl,
            val_dl,
        )
        test(params, drawer, test_dl, test_case_number)
        drawer.save_train_res_plot(train_h, val_h, test_case_number)


def parse_args() -> tuple[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Path to the used dataset.")
    parser.add_argument("result_path", type=str, help="Path to the result folder.")
    args = parser.parse_args()
    return args.dataset_path, args.result_path


def peek(params: hyperparams.Hyperparameter, drawer: draw.Drawer):
    train_dl, val_dl, test_dl = load_data(params)
    for batch in train_dl:
        for i in range(0, params.batch_size):
            drawer.show_sample(batch[0][i], batch[1][i])


def test(params: hyperparams.Hyperparameter, drawer: draw.Drawer, test_loader: DataLoader, test_case_number: int):
    # tracking test loss
    print("Testing model")
    model = params.model
    test_loss = 0.0
    class_correct = [0.0] * params.class_count
    class_total = [0.0] * params.class_count
    points = np.array([[0] * params.class_count for _ in range(params.class_count)], np.int32)

    model.eval()  # Prepare model for testing
    criterion = torch.nn.CrossEntropyLoss()

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

    for i, label in enumerate(params.classes):
        if class_total[i] > 0:
            print(f"Test Accuracy of {label}: {100*class_correct[i]/class_total[i]:.3f}%")
        else:
            print(f"Test Accuracy of {label}s: N/A (no training examples)")

    print(
        f"Full Test Accuracy: {round(100. * np.sum(class_correct) / np.sum(class_total), 2)}% {np.sum(class_correct)} out of {np.sum(class_total)}"
    )
    drawer.save_classification_plot(points, test_case_number)


if __name__ == "__main__":
    dataset_path, result_path = parse_args()
    hp = hyperparams.Hyperparameter.build(
        model=TC_Final,
        classes=[
            "glioma_tumor",
            "meningioma_tumor",
            "no_tumor",
            "pituitary_tumor",
        ],
        batch_size=64,
        learing_rate=2e-5,
        epochs=30,
        input_size=148,
        dataset_path=dataset_path,
        result_path=result_path,
    )
    main(hp)

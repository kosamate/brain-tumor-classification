from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns
from hyperparams import CLASSES
from typing import List


def show_sample(img: Tensor, label_idx):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(img.permute(1, 2, 0), cmap="gray")
    plt.title(CLASSES[label_idx])
    plt.show()


def show_classification(data):
    sns.heatmap(data, linewidths=1, annot=True, xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("ground truth")
    plt.ylabel("predicted")
    plt.show()


def plot_train_res(train_h: List[float], val_h: List[float]):
    assert len(train_h) == len(val_h)
    plt.plot(range(1, len(val_h) + 1), train_h, "b", label="train")
    plt.plot(range(1, len(val_h) + 1), val_h, "g", label="validation")
    plt.subplots_adjust(left=0.36, bottom=0.36)
    plt.legend(loc="upper left")
    plt.show()

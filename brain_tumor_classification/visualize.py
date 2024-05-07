from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns
from hyperparams import CLASSES
from typing import List

_CLASSIFICATION_PATH_TEMPLATE = "D:/Project/Dipterv/Results/class_result_CNN{test_case}.png"
_TRAINING_PATH_TEMPLATE = "D:/Project/Dipterv/Results/class_training_CNN{test_case}.png"


def show_sample(img: Tensor, label_idx: int):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(img.permute(1, 2, 0), cmap="gray")
    plt.title(CLASSES[label_idx])
    plt.show()


def save_classification_plot(data, test_case_number: int):
    plt.figure(1)
    sns.heatmap(data, linewidths=1, annot=True, xticklabels=CLASSES, yticklabels=CLASSES)
    plt.subplots_adjust(left=0.36, bottom=0.36)
    plt.xlabel("ground truth")
    plt.ylabel("predicted")
    plt.savefig(_CLASSIFICATION_PATH_TEMPLATE.format(test_case=test_case_number))
    plt.close(1)


def save_train_res_plot(train_h: List[float], val_h: List[float], test_case_number: int):
    assert len(train_h) == len(val_h)
    # plt.subplots_adjust(left=0.36, bottom=0.36)
    plt.figure(2)
    plt.plot(range(1, len(val_h) + 1), train_h, "b", label="train")
    plt.plot(range(1, len(val_h) + 1), val_h, "g", label="validation")
    plt.legend(loc="upper left")
    plt.savefig(_TRAINING_PATH_TEMPLATE.format(test_case=test_case_number))
    plt.close(2)

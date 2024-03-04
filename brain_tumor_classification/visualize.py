from torch import Tensor
import matplotlib.pyplot as plt
from hyperparams import CLASSES


def show_sample(img: Tensor, label_idx):
    plt.imshow(img.permute(1, 2, 0))
    plt.title(CLASSES[label_idx])
    plt.show()

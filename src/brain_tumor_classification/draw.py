import torch
import matplotlib.pyplot as plt
import seaborn as sns
from brain_tumor_classification import hyperparams


class Drawer:
    def __init__(self, params: hyperparams.Hyperparameter):
        self._params = params
        self._classification_path_template = params.result_path.name + "/class_result_CNN{test_case}.png"
        self._training_path_template = params.result_path.name + "/class_training_CNN{test_case}.png"

    def show_sample(self, img: torch.Tensor, label_idx: int) -> None:
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(img.permute(1, 2, 0), cmap="gray")
        plt.title(self._params.classes[label_idx])
        plt.show()

    def save_classification_plot(self, data, test_case_number: int, accuracy: int | None = None) -> None:
        plt.figure(1)
        sns.heatmap(data, linewidths=1, annot=True, xticklabels=self._params.classes, yticklabels=self._params.classes)
        plt.subplots_adjust(left=0.36, bottom=0.36)
        plt.xlabel("ground truth")
        plt.xticks(rotation=-30, ha="left", rotation_mode="anchor")
        plt.ylabel("predicted")
        plt.title(f"Test #{test_case_number} accuracy")
        plt.text(0.1, 0.3, f"accuracy={accuracy}")
        plt.savefig(self._classification_path_template.format(test_case=test_case_number))
        plt.close(1)

    def save_train_res_plot(self, train_h: list[float], val_h: list[float], test_case_number: int) -> None:
        assert len(train_h) == len(val_h)
        # plt.subplots_adjust(left=0.36, bottom=0.36)
        plt.figure(2)
        plt.plot(range(1, len(val_h) + 1), train_h, "b", label="train")
        plt.plot(range(1, len(val_h) + 1), val_h, "g", label="validation")
        plt.legend(loc="upper left")
        plt.title(f"Training #{test_case_number} loss")
        plt.savefig(self._training_path_template.format(test_case=test_case_number))
        plt.close(2)

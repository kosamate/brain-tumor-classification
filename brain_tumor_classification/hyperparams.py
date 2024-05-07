from torch.nn import Module
from torchsummary import summary

CLASSES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor",
]
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
EPOCHS = 12
INPUT_SIZE = 128

from model import (
    TumorClassificationCNN_Conv,
    TumorClassificationCNN_FC,
    TumorClassificationCNN_Mixed,
    TumorClassificationCNN_Conv_Norm,
    DogClassificationCNN,
    TC_CNN_Big,
)

MODEL: Module = TC_CNN_Big()


def print_hyperparams():
    print("----Hyperparameters----")
    print(f"batch_size = {BATCH_SIZE}")
    print(f"epochs = {EPOCHS}")
    print(f"learning_rate = {LEARNING_RATE}")
    print(f"input_size = {INPUT_SIZE}")
    print(f"model = {MODEL.__class__.__name__}")
    summary(MODEL, (1, INPUT_SIZE, INPUT_SIZE))
    print()

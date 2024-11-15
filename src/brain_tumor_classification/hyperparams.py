import dataclasses
import pathlib
import textwrap
import torchsummary
import torch
from brain_tumor_classification.model import TumorClassification


@dataclasses.dataclass
class Hyperparameter:
    model: TumorClassification
    classes: list[str]
    batch_size: int
    learing_rate: float
    epochs: int
    input_size: int
    result_path: pathlib.Path
    dataset_path: pathlib.Path
    device: str

    @classmethod
    def build(
        cls,
        model: type[TumorClassification],
        classes: list[str],
        batch_size: int,
        learing_rate: float,
        epochs: int,
        input_size: int,
        result_path: str,
        dataset_path: str,
    ) -> "Hyperparameter":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return cls(
            model(input_size, len(classes)).to(device),
            classes,
            batch_size,
            learing_rate,
            epochs,
            input_size,
            pathlib.Path(result_path),
            pathlib.Path(dataset_path),
            device=device,
        )

    def __repr__(self) -> str:
        s = textwrap.dedent(
            f"""\
            ----Hyperparameters----
            batch_size = {self.batch_size}
            epochs = {self.epochs}
            learning_rate = {self.learing_rate}
            input_size = {self.input_size}
            model = {self.model.__class__.__name__}
            """
        )
        s += str(torchsummary.summary(self.model, (1, self.input_size, self.input_size), verbose=0))
        return s

    @property
    def class_count(self) -> int:
        return len(self.classes)

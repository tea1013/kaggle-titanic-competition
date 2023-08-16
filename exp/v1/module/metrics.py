import numpy as np
from numpy import ndarray
from ocha.models.metrics import Metrics
from sklearn.metrics import accuracy_score


class Accuracy(Metrics):
    def __init__(self) -> None:
        name = "Accuracy"
        super().__init__(name)

    def execute(self, y_true: ndarray, y_pred: ndarray) -> float:
        _y_pred = np.where(y_pred < 0.5, 0, 1)
        return accuracy_score(y_true, _y_pred)

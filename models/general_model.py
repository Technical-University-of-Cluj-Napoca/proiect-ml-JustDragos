
from abc import ABC, abstractmethod

import numpy as np


class GeneralModel(ABC):
    
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train

    @abstractmethod
    def fit(self) -> None:
        pass

   
    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        pass
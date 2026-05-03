
import numpy as np

from models.general_model import GeneralModel


class RegressionModel(GeneralModel):
   def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        super().__init__(X_train, y_train)
    
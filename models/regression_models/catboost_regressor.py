# models/regressors/catboost_regressor.py
import numpy as np
from catboost import CatBoostRegressor
from models.regression_models.regression_model import RegressionModel

class CatBoostRegressorModel(RegressionModel):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        super().__init__(X_train, y_train)
        self.model = CatBoostRegressor(verbose=0, random_state=42)

    def fit(self) -> None:
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)
# models/regressors/xgboost_regressor.py
import numpy as np
from xgboost import XGBRegressor
from models.regression_models.regression_model import RegressionModel

class XGBoostRegressorModel(RegressionModel):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        super().__init__(X_train, y_train)
        self.model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

    def fit(self) -> None:
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)
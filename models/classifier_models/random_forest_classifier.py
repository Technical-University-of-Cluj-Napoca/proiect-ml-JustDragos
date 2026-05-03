# models/random_forest.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from models.classifier_models.classifer_model import ClassifierModel

class RandomForestModel(ClassifierModel):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        super().__init__(X_train, y_train)
        self.model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)

    def fit(self) -> None:
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)
# models/ebm_classifier.py
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from models.classifier_models.classifer_model import ClassifierModel

class EBMClassifier(ClassifierModel):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        super().__init__(X_train, y_train)
        self.model = ExplainableBoostingClassifier(random_state=42)

    def fit(self) -> None:
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)
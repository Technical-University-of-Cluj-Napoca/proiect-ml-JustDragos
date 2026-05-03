import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from models.classifier_models.classifer_model import ClassifierModel

class LogisticRegressionClassifier(ClassifierModel):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        super().__init__(X_train, y_train)
        self.scaler = StandardScaler()
        self.model  = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

    def fit(self) -> None:
        X_scaled = self.scaler.fit_transform(self.X_train)  
        self.model.fit(X_scaled, self.y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_test)             
        return self.model.predict(X_scaled)
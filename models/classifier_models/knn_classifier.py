# models/knn.py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from models.classifier_models.classifer_model import ClassifierModel

class KNNClassifier(ClassifierModel):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        super().__init__(X_train, y_train) 
        self.scaler = StandardScaler()
        self.model  = KNeighborsClassifier(n_neighbors=5)

    def fit(self) -> None:
        X_scaled = self.scaler.fit_transform(self.X_train)  

        self.model.fit(X_scaled, self.y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
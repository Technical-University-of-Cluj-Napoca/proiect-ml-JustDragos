import numpy as np

from sklearn.naive_bayes import GaussianNB
from models.classifier_models.classifer_model import ClassifierModel


class NaiveBayesClassifier(ClassifierModel):
    
    
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
            super().__init__(X_train, y_train)
            self.model = GaussianNB()

    def fit(self) -> None:
            self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
            return self.model.predict(X_test)
    
    
    
    
    
    
    
    
    
    
    # How it would work if we were to implement it directly in this class without inheriting from Scikit-learn's BaseEstimator and ClassifierMixin:
    
    # def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
    #     super().__init__(X_train, y_train)

    # def fit(self) -> None:
    #     self.classes = np.unique(self.y_train)
    #     self.mean     = np.zeros((len(self.classes), self.X_train.shape[1]))
    #     self.variance = np.zeros((len(self.classes), self.X_train.shape[1]))
    #     self.priors   = np.zeros(len(self.classes))

    #     for idx, c in enumerate(self.classes):
    #         X_c = self.X_train[self.y_train == c]
    #         self.mean[idx, :]     = X_c.mean(axis=0)
    #         self.variance[idx, :] = X_c.var(axis=0)
    #         self.priors[idx]      = X_c.shape[0] / self.X_train.shape[0]

    # def likelihood(self, X: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    #     exponent = np.exp(-0.5 * ((X - mean) ** 2) / (variance + 1e-9))          # ← removed extra )
    #     return (1 / np.sqrt(2 * np.pi * (variance + 1e-9))) * exponent            # ← 1e-9 inside sqrt too

    # def predict(self, X_test: np.ndarray) -> np.ndarray:
    #     posteriors = []
    #     for idx, c in enumerate(self.classes):                                     # ← was missing c
    #         prior      = np.log(self.priors[idx])
    #         likelihood = np.sum(np.log(self.likelihood(X_test, self.mean[idx], self.variance[idx])), axis=1)
    #         posterior  = prior + likelihood
    #         posteriors.append(posterior)
    #     posteriors = np.array(posteriors).T
    #     return self.classes[np.argmax(posteriors, axis=1)]
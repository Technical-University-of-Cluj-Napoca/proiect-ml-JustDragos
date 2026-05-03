
import numpy as np

from models.general_model import GeneralModel


class ClassifierModel(GeneralModel):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        super().__init__(X_train, y_train)
    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        result: float = (y_true == y_pred).mean().item()
        return result
    
    # # Out of all predicted, how many were correct
    # def precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    #     true_positive = np.sum((y_true == 1) & (y_pred == 1))
    #     false_positive = np.sum((y_true == 0) & (y_pred == 1))
    #     return true_positive / (true_positive + false_positive + 1e-9)
    # # Out of all positive, how many were correct
    # def recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    #     true_positive = np.sum((y_true == 1) & (y_pred == 1))  # removed stray backtick
    #     false_negative = np.sum((y_true == 1) & (y_pred == 0))
    #     return true_positive / (true_positive + false_negative + 1e-9)
    
    # def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    #     prec = self.precision(y_true, y_pred)
    #     rec = self.recall(y_true, y_pred)
    #     return 2 * (prec * rec) / (prec + rec + 1e-9)

    def roc_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_pred))

    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred)
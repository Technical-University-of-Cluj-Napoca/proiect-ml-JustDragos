# models/model_comparison.py

import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from comparison_programs.AOC_and_CM_Run import plot_roc_and_confusion
from models.classifier_models.classifer_model import ClassifierModel
from models.classifier_models.naive_bayes_classifier import NaiveBayesClassifier
from models.classifier_models.logistic_regression_classifier import LogisticRegressionClassifier
from models.classifier_models.decision_tree_classifier import DecisionTreeModel
from models.classifier_models.random_forest_classifier import RandomForestModel
from models.classifier_models.svm_classifier import SVMClassifier
from models.classifier_models.knn_classifier import KNNClassifier
from models.classifier_models.xgboost_classifier import XGBoostModel
from models.classifier_models.catboost_model_classifier import CatBoostModel
from models.classifier_models.explainable_boosting_classifier import EBMClassifier



def compare_classifiers(df: DataFrame, features: list):
    X = df[features].values
    y = df["Target_Cls_float"].values.astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, shuffle=False)
    X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    models: dict[str, ClassifierModel] = {
        "Naive Bayes":         NaiveBayesClassifier(X_train, y_train),
        "Logistic Regression": LogisticRegressionClassifier(X_train, y_train),
        "Decision Tree":       DecisionTreeModel(X_train, y_train),
        "Random Forest":       RandomForestModel(X_train, y_train),
        "SVM":                 SVMClassifier(X_train, y_train),
        "KNN":                 KNNClassifier(X_train, y_train),
        "XGBoost":             XGBoostModel(X_train, y_train),
        "CatBoost":            CatBoostModel(X_train, y_train),
        "EBM":                 EBMClassifier(X_train, y_train),
    }

    results = {}
    for name, model in models.items():
        model.fit()

        y_val_pred  = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        results[name] = {
            "val_accuracy":   accuracy_score(y_val,  y_val_pred),  # weighted not macro
            "val_f1":         f1_score(y_val,  y_val_pred,  average="weighted"),  # weighted not macro
            "val_precision":  precision_score(y_val,  y_val_pred, average="weighted"),  # weighted not macro
            "val_recall":     recall_score(y_val,  y_val_pred, average="weighted"),  # weighted not macro
            "test_accuracy":  accuracy_score(y_test, y_test_pred ),  # weighted not macro
            "test_f1":        f1_score(y_test, y_test_pred, average="weighted"),  # weighted not macro
            "test_precision": precision_score(y_test, y_test_pred, average="weighted"),  # weighted not macro
            "test_recall":    recall_score(y_test, y_test_pred, average="weighted"),  # weighted not macro
        }

        r = results[name]
        print(f" {name}")
        print(f"   val  → acc={r['val_accuracy']:.3f}  prec={r['val_precision']:.3f}  rec={r['val_recall']:.3f}  f1={r['val_f1']:.3f}")
        print(f"   test → acc={r['test_accuracy']:.3f}  prec={r['test_precision']:.3f}  rec={r['test_recall']:.3f}  f1={r['test_f1']:.3f}")
    plot_roc_and_confusion(models, X_val, y_val, X_test, y_test)
    
    _plot_results(results)
    return results


def _plot_results(results: dict):
    names  = list(results.keys())
    x      = np.arange(len(names))
    width  = 0.2                        # 4 bars so narrower

    metrics = ["test_accuracy", "test_f1", "test_precision", "test_recall"]
    labels  = ["Accuracy", "F1 Weighted", "Precision", "Recall"]
    colors  = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        scores = [results[n][metric] for n in names]
        offset = (i - 1.5) * width      # center the 4 bars around each tick
        bars   = ax.bar(x + offset, scores, width, label=label, color=color)

        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=7
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Classifier Comparison — Test Set")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()
    return results
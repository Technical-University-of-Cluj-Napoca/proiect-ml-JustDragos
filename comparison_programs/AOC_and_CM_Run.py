import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_roc_and_confusion(models: dict, X_val, y_val, X_test, y_test):
    """
    For each model plots side by side:
    - Left  : ROC curves (val vs test) for each class
    - Right : Confusion matrices (val vs test)
    """
    classes     = [0, 1, 2]
    class_names = ["scadere", "stabil", "crestere"]
    colors      = ["steelblue", "coral", "mediumseagreen"]

    y_val_bin  = label_binarize(y_val,  classes=classes)
    y_test_bin = label_binarize(y_test, classes=classes)

    for name, model in models.items():
        y_val_pred  = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        y_val_pred_bin  = label_binarize(y_val_pred,  classes=classes)
        y_test_pred_bin = label_binarize(y_test_pred, classes=classes)

        # 1 row, 4 cols: ROC val | ROC test | CM val | CM test
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.suptitle(f"{name}", fontsize=14, fontweight="bold")

        # ── ROC val ───────────────────────────────────────────────────────────
        for i, (cls_name, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_val_pred_bin[:, i])
            roc_auc     = auc(fpr, tpr)
            axes[0].plot(fpr, tpr, color=color, label=f"{cls_name} (AUC={roc_auc:.2f})")

        axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axes[0].set_title("ROC Curve — Validation")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].legend(loc="lower right", fontsize=8)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1.05])

        # ── ROC test ──────────────────────────────────────────────────────────
        for i, (cls_name, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_pred_bin[:, i])
            roc_auc     = auc(fpr, tpr)
            axes[1].plot(fpr, tpr, color=color, label=f"{cls_name} (AUC={roc_auc:.2f})")

        axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axes[1].set_title("ROC Curve — Test")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend(loc="lower right", fontsize=8)
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1.05])

        # ── CM val ────────────────────────────────────────────────────────────
        cm_val = model.confusion_matrix(y_val, y_val_pred)
        sns.heatmap(
            cm_val,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[2]
        )
        axes[2].set_title("Confusion Matrix — Validation")
        axes[2].set_ylabel("Actual")
        axes[2].set_xlabel("Predicted")

        # ── CM test ───────────────────────────────────────────────────────────
        cm_test = model.confusion_matrix(y_test, y_test_pred)
        sns.heatmap(
            cm_test,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[3]
        )
        axes[3].set_title("Confusion Matrix — Test")
        axes[3].set_ylabel("Actual")
        axes[3].set_xlabel("Predicted")

        plt.tight_layout()
        plt.show()
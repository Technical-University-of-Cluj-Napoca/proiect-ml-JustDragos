# models/regressor_comparison.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from models.regression_models.linear_regression import LinearRegressionModel
from models.regression_models.decision_tree_regressor import DecisionTreeRegressorModel
from models.regression_models.random_forest_regressor import RandomForestRegressorModel
from models.regression_models.gaussian_process_regressor import GaussianProcessRegressorModel

from models.regression_models.knn_regressor import KNNRegressorModel
from models.regression_models.svr_regressor import SVRModel
from models.regression_models.xgboost_regressor import XGBoostRegressorModel
from models.regression_models.catboost_regressor import CatBoostRegressorModel
from models.regression_models.ebm_regressor import EBMRegressorModel





def compare_regressors(df, features: list):
    X = df[features].values
    y = df["Target_Reg"].values

    # 75 / 12.5 / 12.5 split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, shuffle=False)
    X_val,  X_test,  y_val,  y_test  = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    models = {
        "Linear Regression":  LinearRegressionModel(X_train, y_train),
        "Decision Tree":      DecisionTreeRegressorModel(X_train, y_train),
        "Random Forest":      RandomForestRegressorModel(X_train, y_train),
        "SVR":                SVRModel(X_train, y_train),
        "KNN":                KNNRegressorModel(X_train, y_train),
        #"Gaussian Process":   GaussianProcessRegressorModel(X_train, y_train),
        "XGBoost":            XGBoostRegressorModel(X_train, y_train),
        "CatBoost":           CatBoostRegressorModel(X_train, y_train),
        "EBM":                EBMRegressorModel(X_train, y_train),
    }

    results = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit()

        for split, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
            y_pred = model.predict(X_s)
            results.setdefault(name, {})[split] = {
                "mse":  model.mean_squared_error(y_s, y_pred),
                "mae":  model.mean_absolute_error(y_s, y_pred),
                "rmse": model.rmse(y_s, y_pred),
                "r2":   model.r2_score(y_s, y_pred),
            }

        print(
            f"    val  → MAE={results[name]['val']['mae']:.4f}  "
            f"RMSE={results[name]['val']['rmse']:.4f}  "
            f"R2={results[name]['val']['r2']:.4f}"
        )
        print(
            f"    test → MAE={results[name]['test']['mae']:.4f}  "
            f"RMSE={results[name]['test']['rmse']:.4f}  "
            f"R2={results[name]['test']['r2']:.4f}"
        )

    _plot_regressor_results(results, models, X_val, y_val, X_test, y_test)
    return results


def _plot_regressor_results(results, models, X_val, y_val, X_test, y_test):
    names  = list(results.keys())
    colors = {"val": "steelblue", "test": "coral"}
    x      = np.arange(len(names))
    width  = 0.35

    metrics = ["mae", "rmse", "r2"]
    titles  = ["MAE (lower is better)", "RMSE (lower is better)", "R² (higher is better)"]

    # ── Plot 1: bar charts for all 3 metrics ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Regressor Comparison — Validation vs Test", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(axes, metrics, titles):
        val_scores  = [results[n]["val"][metric]  for n in names]
        test_scores = [results[n]["test"][metric] for n in names]

        bars1 = ax.bar(x - width/2, val_scores,  width, label="Validation", color=colors["val"])
        bars2 = ax.bar(x + width/2, test_scores, width, label="Test",       color=colors["test"])

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.legend(fontsize=8)

        for bar in bars1 + bars2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=6
            )

    plt.tight_layout()
    plt.show()

    # ── Plot 2: actual vs predicted scatter per model ─────────────────────────
    cols    = 3
    rows    = (len(models) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
    axes    = axes.flatten()
    fig.suptitle("Actual vs Predicted — Test Set", fontsize=14, fontweight="bold")

    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)

        axes[i].scatter(y_test, y_pred, alpha=0.4, color="steelblue", s=10)
        # perfect prediction line
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        axes[i].plot([mn, mx], [mn, mx], "r--", linewidth=1)
        axes[i].set_title(f"{name}\nR²={results[name]['test']['r2']:.3f}  MAE={results[name]['test']['mae']:.3f}")
        axes[i].set_xlabel("Actual")
        axes[i].set_ylabel("Predicted")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
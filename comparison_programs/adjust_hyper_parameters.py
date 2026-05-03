# models/hyperparameter_tuning.py
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from models.classifier_models.naive_bayes_classifier import NaiveBayesClassifier
from models.classifier_models.logistic_regression_classifier import LogisticRegressionClassifier
from models.classifier_models.decision_tree_classifier import DecisionTreeModel
from models.classifier_models.random_forest_classifier import RandomForestModel
from models.classifier_models.svm_classifier import SVMClassifier
from models.classifier_models.knn_classifier import KNNClassifier
from models.classifier_models.xgboost_classifier import XGBoostModel
from models.classifier_models.catboost_model_classifier import CatBoostModel
from models.classifier_models.explainable_boosting_classifier import EBMClassifier


# ── param grids for GridSearchCV ──────────────────────────────────────────────
GRID_PARAMS = {
    "Naive Bayes":         {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "saga"]},
    "Decision Tree":       {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
    "Random Forest":       {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
    "SVM":                 {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "KNN":                 {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
    "XGBoost":             {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.01, 0.1]},
    "CatBoost":            {"depth": [4, 6, 8], "learning_rate": [0.01, 0.1], "iterations": [100, 200]},
    "EBM":                 {"max_bins": [128, 256], "learning_rate": [0.01, 0.05]},
}

# ── param spaces for BayesSearchCV ────────────────────────────────────────────
BAYES_PARAMS = {
    "Naive Bayes":         {"var_smoothing": Real(1e-11, 1e-5, prior="log-uniform")},
    "Logistic Regression": {"C": Real(0.001, 100, prior="log-uniform")},
    "Decision Tree":       {"max_depth": Integer(2, 20), "min_samples_split": Integer(2, 20)},
    "Random Forest":       {"n_estimators": Integer(50, 300), "max_depth": Integer(3, 20)},
    "SVM":                 {"C": Real(0.01, 100, prior="log-uniform"), "kernel": Categorical(["rbf", "linear"])},
    "KNN":                 {"n_neighbors": Integer(2, 20), "weights": Categorical(["uniform", "distance"])},
    "XGBoost":             {"n_estimators": Integer(50, 200), "max_depth": Integer(2, 8), "learning_rate": Real(0.01, 0.3, prior="log-uniform")},
    "CatBoost":            {"depth": Integer(3, 10), "learning_rate": Real(0.01, 0.3, prior="log-uniform"), "iterations": Integer(50, 300)},
    "EBM":                 {"max_bins": Integer(64, 512), "learning_rate": Real(0.005, 0.1, prior="log-uniform")},
}

# ── sklearn-compatible wrappers ───────────────────────────────────────────────
# GridSearchCV/BayesSearchCV need sklearn estimators, not our custom classes
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.pipeline import Pipeline

# models that need scaling
NEEDS_SCALING = {"Logistic Regression", "SVM", "KNN"}

SKLEARN_MODELS = {
    "Naive Bayes":         GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree":       DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest":       RandomForestClassifier(class_weight="balanced", random_state=42),
    "SVM":                 SVC(class_weight="balanced", random_state=42),
    "KNN":                 KNeighborsClassifier(),
    "XGBoost":             XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42, verbosity=0),
    "CatBoost":            CatBoostClassifier(verbose=0, random_state=42),
    "EBM":                 ExplainableBoostingClassifier(random_state=42),
}


def _get_estimator(name):
    """Wrap model in a pipeline with scaler if needed."""
    model = SKLEARN_MODELS[name]
    if name in NEEDS_SCALING:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return model


def _prefix_params(params: dict, name: str, needs_pipeline: bool) -> dict:
    """Prefix param keys with 'model__' if inside a pipeline."""
    if needs_pipeline:
        return {f"model__{k}": v for k, v in params.items()}
    return params


def tune_top5(results: dict, df: DataFrame, features: list):
    """
    1. Pick top 5 models by test_f1 from compare_classifiers results
    2. Run GridSearchCV and BayesSearchCV for each
    3. Plot comparison
    """
    # ── step 1: pick top 5 ────────────────────────────────────────────────────
    sorted_models = sorted(results.items(), key=lambda x: x[1]["test_f1"], ascending=False)
    top5          = [name for name, _ in sorted_models[:5]]

    print("Top 5 models by F1:")
    for i, name in enumerate(top5, 1):
        print(f"  {i}. {name} — F1={results[name]['test_f1']:.3f}")

    # ── step 2: prepare data ──────────────────────────────────────────────────
    X = df[features].values
    y = df["Target_Cls_float"].values.astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, shuffle=False)
    X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    scorer = make_scorer(f1_score, average="weighted")
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tuning_results = {}

    for name in top5:
        print(f"\nTuning {name}...")
        needs_pipeline = name in NEEDS_SCALING
        estimator      = _get_estimator(name)

        # ── GridSearchCV ──────────────────────────────────────────────────────
        grid_params = _prefix_params(GRID_PARAMS[name], name, needs_pipeline)
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=grid_params,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        grid_pred    = grid_search.best_estimator_.predict(X_test)
        grid_f1      = f1_score(y_test, grid_pred, average="weighted")
        grid_val_f1  = f1_score(y_val, grid_search.best_estimator_.predict(X_val), average="weighted")

        print(f"  GridSearch  best params : {grid_search.best_params_}")
        print(f"  GridSearch  val_f1={grid_val_f1:.3f}  test_f1={grid_f1:.3f}")

        # ── BayesSearchCV ─────────────────────────────────────────────────────
        bayes_params = _prefix_params(BAYES_PARAMS[name], name, needs_pipeline)
        bayes_search = BayesSearchCV(
            estimator=estimator,
            search_spaces=bayes_params,
            scoring=scorer,
            cv=cv,
            n_iter=30,              # number of parameter settings sampled
            n_jobs=-1,
            verbose=0,
            random_state=42,
        )
        bayes_search.fit(X_train, y_train)
        bayes_pred   = bayes_search.best_estimator_.predict(X_test)
        bayes_f1     = f1_score(y_test, bayes_pred, average="weighted")
        bayes_val_f1 = f1_score(y_val, bayes_search.best_estimator_.predict(X_val), average="weighted")

        print(f"  BayesSearch best params : {bayes_search.best_params_}")
        print(f"  BayesSearch val_f1={bayes_val_f1:.3f}  test_f1={bayes_f1:.3f}")

        tuning_results[name] = {
            "baseline_f1":  results[name]["test_f1"],
            "grid_val_f1":  grid_val_f1,
            "grid_f1":      grid_f1,
            "bayes_val_f1": bayes_val_f1,
            "bayes_f1":     bayes_f1,
        }

    _plot_tuning_results(tuning_results)
    return tuning_results


def _plot_tuning_results(tuning_results: dict):
    names    = list(tuning_results.keys())
    x        = np.arange(len(names))
    width    = 0.25

    baseline = [tuning_results[n]["baseline_f1"]  for n in names]
    grid_f1  = [tuning_results[n]["grid_f1"]      for n in names]
    bayes_f1 = [tuning_results[n]["bayes_f1"]     for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Hyperparameter Tuning — Baseline vs GridSearch vs BayesSearch", fontsize=13, fontweight="bold")

    # ── Plot 1: test F1 comparison ────────────────────────────────────────────
    ax = axes[0]
    b1 = ax.bar(x - width, baseline, width, label="Baseline",    color="steelblue")
    b2 = ax.bar(x,         grid_f1,  width, label="GridSearchCV", color="coral")
    b3 = ax.bar(x + width, bayes_f1, width, label="BayesSearchCV", color="mediumseagreen")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Weighted (Test)")
    ax.set_title("Test F1 — Baseline vs Tuned")
    ax.legend()

    for bar in b1 + b2 + b3:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=7
        )

    # ── Plot 2: val vs test F1 for tuned models (overfitting check) ───────────
    ax2   = axes[1]
    x2    = np.arange(len(names) * 2)
    width2 = 0.35

    grid_pairs  = []
    bayes_pairs = []
    xlabels     = []

    for name in names:
        grid_pairs.extend([tuning_results[name]["grid_val_f1"],  tuning_results[name]["grid_f1"]])
        bayes_pairs.extend([tuning_results[name]["bayes_val_f1"], tuning_results[name]["bayes_f1"]])
        xlabels.extend([f"{name}\nVal", f"{name}\nTest"])

    ax2.bar(x2 - width2/2, grid_pairs,  width2, label="GridSearchCV",  color="coral",          alpha=0.8)
    ax2.bar(x2 + width2/2, bayes_pairs, width2, label="BayesSearchCV", color="mediumseagreen",  alpha=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=7)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("F1 Weighted")
    ax2.set_title("Val vs Test F1 — Overfitting Check")
    ax2.legend()

    plt.tight_layout()
    plt.show()
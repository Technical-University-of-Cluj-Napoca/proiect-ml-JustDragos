# models/learning_curves.py
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier


NEEDS_SCALING = {"Logistic Regression", "SVM", "KNN"}

SKLEARN_MODELS = {
    "Naive Bayes":         GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree":       DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest":       RandomForestClassifier(class_weight="balanced", random_state=42),
    "SVM":                 SVC(class_weight="balanced", random_state=42),
    "KNN":                 KNeighborsClassifier(),
    "XGBoost":             XGBClassifier(eval_metric="mlogloss", random_state=42, verbosity=0),
    "CatBoost":            CatBoostClassifier(verbose=0, random_state=42),
    "EBM":                 ExplainableBoostingClassifier(random_state=42),
}


def _get_estimator(name: str):
    model = SKLEARN_MODELS[name]
    if name in NEEDS_SCALING:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return model


def plot_learning_curves(results: dict, df: DataFrame, features: list):
    """
    1. Pick top 5 models by test_f1
    2. Plot learning curves for each
    3. Print overfitting/underfitting analysis
    """
    # ── step 1: pick top 5 ────────────────────────────────────────────────────
    top5 = [
        name for name, _ in
        sorted(results.items(), key=lambda x: x[1]["test_f1"], reverse=True)[:5]
    ]

    print("Top 5 models selected for learning curves:")
    for i, name in enumerate(top5, 1):
        print(f"  {i}. {name} — test_f1={results[name]['test_f1']:.3f}")

    # ── step 2: prepare data ──────────────────────────────────────────────────
    X = df[features].values
    y = df["Target_Cls_float"].values.astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, shuffle=False)
    X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    scorer = make_scorer(f1_score, average="weighted")
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── step 3: plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes      = axes.flatten()
    fig.suptitle("Learning Curves — Top 5 Models", fontsize=14, fontweight="bold")

    analyses = {}

    for i, name in enumerate(top5):
        print(f"\nComputing learning curve for {name}...")
        estimator = _get_estimator(name)

        train_sizes, train_scores, val_scores = learning_curve(
            estimator      = estimator,
            X              = X_train,
            y              = y_train,
            cv             = cv,
            scoring        = scorer,
            train_sizes    = np.linspace(0.1, 1.0, 10),  # 10 points from 10% to 100%
            n_jobs         = -1,
            shuffle        = True,
            random_state   = 42,
        )

        # mean and std across folds
        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        # ── plot ──────────────────────────────────────────────────────────────
        ax = axes[i]
        ax.plot(train_sizes, train_mean, "o-", color="steelblue", label="Training score")
        ax.plot(train_sizes, val_mean,   "o-", color="coral",     label="Validation score")

        # shaded std bands
        ax.fill_between(train_sizes,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.15, color="steelblue")
        ax.fill_between(train_sizes,
                        val_mean - val_std,
                        val_mean + val_std,
                        alpha=0.15, color="coral")

        ax.set_title(f"{name}\ntest_f1={results[name]['test_f1']:.3f}")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("F1 Weighted")
        ax.set_ylim(0, 1.1)
        ax.legend(loc="lower right", fontsize=8)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
        ax.grid(True, alpha=0.3)

        # ── diagnose ──────────────────────────────────────────────────────────
        final_train = train_mean[-1]
        final_val   = val_mean[-1]
        gap         = final_train - final_val

        if final_train < 0.7 and final_val < 0.7:
            diagnosis = "UNDERFITTING — both scores low, model too simple"
        elif gap > 0.15:
            diagnosis = "OVERFITTING — large gap between train and val"
        elif gap < 0.05:
            diagnosis = "GOOD FIT — train and val scores are close"
        else:
            diagnosis = "SLIGHT OVERFITTING — acceptable gap"

        analyses[name] = {
            "final_train": final_train,
            "final_val":   final_val,
            "gap":         gap,
            "diagnosis":   diagnosis,
        }

    # hide unused subplot
    for j in range(len(top5), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    # ── step 4: print analysis ────────────────────────────────────────────────
    print("\n" + "="*65)
    print("LEARNING CURVE ANALYSIS")
    print("="*65)
    for name, a in analyses.items():
        print(f"\n{name}")
        print(f"  Train score : {a['final_train']:.3f}")
        print(f"  Val score   : {a['final_val']:.3f}")
        print(f"  Gap         : {a['gap']:.3f}")
        print(f"  Diagnosis   : {a['diagnosis']}")

    return analyses
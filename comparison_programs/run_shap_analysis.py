# shap_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline


def get_shap_explainer(model, X_background):
    """Pick the right explainer based on model type."""
    # unwrap pipeline if needed
    inner = model.named_steps["model"] if isinstance(model, Pipeline) else model
    name  = type(inner).__name__

    tree_based = (
        "RandomForest", "DecisionTree", "XGB", "CatBoost",
        "ExplainableBoosting", "GradientBoosting"
    )

    if any(t in name for t in tree_based):
        return shap.TreeExplainer(inner)
    else:
        # model-agnostic fallback — works for any model
        background = shap.sample(X_background, min(100, len(X_background)))
        return shap.KernelExplainer(model.predict, background)


def get_shap_values(explainer, X, model, is_classifier: bool):
    """Get shap values handling multiclass and regression."""
    inner = model.named_steps["model"] if isinstance(model, Pipeline) else model

    if isinstance(explainer, shap.TreeExplainer):
        shap_vals = explainer.shap_values(X)
        # multiclass returns list — take class with highest avg absolute shap
        if isinstance(shap_vals, list):
            # pick class 2 (crestere) as most interesting for trading
            shap_vals = shap_vals[2]
    else:
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[2]

    return shap_vals


def run_shap_analysis(
    results:    dict,
    models:     dict,
    X_train:    np.ndarray,
    X_test:     np.ndarray,
    features:   list,
    is_classifier: bool = True,
    top_n:      int = 3,
):
    """
    Full SHAP analysis for top N tuned models:
    - Global: summary plot + bar plot
    - Local:  waterfall plot + force plot for one prediction
    - Scatter plots for top 3 features
    """
    # ── pick top 3 by test_f1 (classifier) or test_r2 (regressor) ────────────
    metric = "test_f1" if is_classifier else "r2"
    if is_classifier:
        top = [
            name for name, _ in
            sorted(results.items(), key=lambda x: x[1]["test_f1"], reverse=True)[:top_n]
        ]
    else:
        top = [
            name for name, _ in
            sorted(results.items(), key=lambda x: x[1]["test"]["r2"], reverse=True)[:top_n]
        ]

    print(f"Running SHAP for top {top_n} models: {top}")

    for model_name in top:
        model = models[model_name]
        print(f"\n{'='*60}")
        print(f"SHAP Analysis — {model_name}")
        print(f"{'='*60}")

        # use a sample for speed
        X_sample = X_test[:300]

        # get inner model for tree explainer
        inner = model.named_steps["model"] if isinstance(model, Pipeline) else model

        # transform X if pipeline
        if isinstance(model, Pipeline):
            X_transformed = model.named_steps["scaler"].transform(X_sample)
            X_bg          = model.named_steps["scaler"].transform(X_train[:200])
        else:
            X_transformed = X_sample
            X_bg          = X_train[:200]

        explainer = get_shap_explainer(model, X_train)
        shap_vals = get_shap_values(explainer, X_transformed, model, is_classifier)

        shap_df = pd.DataFrame(X_sample, columns=features)

        # ── 1. Global summary plot ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.summary_plot(
            shap_vals, shap_df,
            feature_names=features,
            show=False
        )
        plt.title(f"SHAP Summary Plot — {model_name}", fontsize=12)
        plt.tight_layout()
        plt.show()

        # ── 2. Global bar plot ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.summary_plot(
            shap_vals, shap_df,
            feature_names=features,
            plot_type="bar",
            show=False
        )
        plt.title(f"SHAP Feature Importance (Bar) — {model_name}", fontsize=12)
        plt.tight_layout()
        plt.show()

        # ── 3. Local waterfall plot (first prediction) ────────────────────────
        try:
            explainer_obj = shap.Explainer(model.predict, X_bg)
            shap_exp      = explainer_obj(X_sample[:1])
            fig, ax = plt.subplots(figsize=(9, 4))
            shap.waterfall_plot(shap_exp[0], show=False)
            plt.title(f"SHAP Waterfall (Local) — {model_name}", fontsize=12)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"  Waterfall skipped: {e}")

        # ── 4. Local force plot (first prediction) ────────────────────────────
        try:
            shap.initjs()
            force = shap.force_plot(
                explainer.expected_value if not isinstance(explainer.expected_value, list)
                else explainer.expected_value[2],
                shap_vals[0],
                shap_df.iloc[0],
                matplotlib=True,
                show=False
            )
            plt.title(f"SHAP Force Plot (Local) — {model_name}", fontsize=10)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"  Force plot skipped: {e}")

        # ── 5. SHAP scatter plots for top 3 features ─────────────────────────
        mean_abs   = np.abs(shap_vals).mean(axis=0)
        top3_idx   = np.argsort(mean_abs)[::-1][:3]
        top3_feats = [features[i] for i in top3_idx]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(f"SHAP Scatter — Top 3 Features — {model_name}", fontsize=12)

        for ax, idx, feat in zip(axes, top3_idx, top3_feats):
            ax.scatter(
                X_sample[:, idx],
                shap_vals[:, idx],
                alpha=0.4, color="steelblue", s=10
            )
            ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
            ax.set_xlabel(feat)
            ax.set_ylabel("SHAP value")
            ax.set_title(feat)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # ── 6. Print interpretation ───────────────────────────────────────────
        print(f"\nTop 3 most important features for {model_name}:")
        for rank, (idx, feat) in enumerate(zip(top3_idx, top3_feats), 1):
            avg_shap = mean_abs[idx]
            direction = "↑ pushes prediction UP" if shap_vals[:, idx].mean() > 0 else "↓ pushes prediction DOWN"
            print(f"  {rank}. {feat:20s} avg|SHAP|={avg_shap:.4f}  {direction}")

        # ── 7. Concrete example ───────────────────────────────────────────────
        print(f"\nConcrete example — first test sample:")
        pred = model.predict(X_sample[:1])[0]
        label = {0: "scadere", 1: "stabil", 2: "crestere"}.get(int(pred), f"{pred:.4f}%")
        print(f"  Prediction : {label}")
        print(f"  Feature values:")
        for feat, val in zip(features, X_sample[0]):
            print(f"    {feat:20s} = {val:.4f}")
        print(f"  Top contributor: {top3_feats[0]} (SHAP={shap_vals[0, top3_idx[0]]:.4f})")
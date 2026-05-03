# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.metrics import make_scorer

st.set_page_config(page_title="ETF ML App", layout="wide")

# ── load everything ───────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    meta = joblib.load("saved_models/metadata.pkl")
    cls_models, reg_models = {}, {}
    for name in meta["top5_cls"]:
        cls_models[name] = joblib.load(f"saved_models/cls_{name.replace(' ', '_')}.pkl")
    for name in meta["top5_reg"]:
        reg_models[name] = joblib.load(f"saved_models/reg_{name.replace(' ', '_')}.pkl")
    return meta, cls_models, reg_models

meta, cls_models, reg_models = load_all()

FEATURES    = meta["features"]
LABEL_MAP   = {0: "scadere", 1: "stabil", 2: "crestere"}
X_test      = meta["X_test"]
yc_test     = meta["yc_test"]
yr_test     = meta["yr_test"]
X_val       = meta["X_val"]
yc_val      = meta["yc_val"]
yr_val      = meta["yr_val"]
X_train     = meta["X_train"]
yc_train    = meta["yc_train"]
yr_train    = meta["yr_train"]

# ── helpers ───────────────────────────────────────────────────────────────────
def plot_confusion_matrix(model, X, y_true, title):
    y_pred = model.predict(X)
    cm     = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["scadere", "stabil", "crestere"],
                yticklabels=["scadere", "stabil", "crestere"], ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    return fig


def plot_learning_curve(model, X, y, scorer, title):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring=scorer,
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, train_mean, "o-", color="steelblue", label="Train")
    ax.plot(train_sizes, val_mean,   "o-", color="coral",     label="Validation")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="steelblue")
    ax.fill_between(train_sizes, val_mean   - val_std,   val_mean   + val_std,   alpha=0.15, color="coral")
    ax.set_title(title)
    ax.set_xlabel("Training Size")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_shap(model, X, feature_names, model_name):
    try:
        # use a small background sample for speed
        background = shap.sample(X, min(100, len(X)))
        explainer  = shap.Explainer(model.predict, background)
        shap_vals  = explainer(X[:200])

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_vals, X[:200],
                          feature_names=feature_names,
                          show=False, plot_type="bar")
        plt.title(f"SHAP — {model_name}")
        return fig
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"SHAP not available:\n{e}", ha="center", va="center")
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def classification_page():
    st.title("📈 ETF Classification")
    st.markdown("""
    **Dataset:** Top 20 ETFs selected by Sharpe ratio, combined into a single dataframe.  
    **Problem:** Predict the next-day price movement as one of:  
    - 🟢 **Crestere** — price increases more than 2%  
    - 🟡 **Stabil** — price moves between -2% and +2%  
    - 🔴 **Scadere** — price drops more than 2%  
    """)

    # ── EDA ───────────────────────────────────────────────────────────────────
    st.subheader("📊 Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        labels  = [LABEL_MAP[i] for i in sorted(np.unique(yc_test))]
        counts  = [np.sum(yc_test == i) for i in sorted(np.unique(yc_test))]
        ax.bar(labels, counts, color=["coral", "steelblue", "mediumseagreen"])
        ax.set_title("Class Distribution (Test Set)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        corr = pd.DataFrame(X_test, columns=FEATURES).corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, cbar=False)
        ax.set_title("Feature Correlation")
        st.pyplot(fig)

    # ── model selector ────────────────────────────────────────────────────────
    st.subheader("🤖 Model Selection")
    selected = st.selectbox("Select a model:", list(cls_models.keys()))
    model    = cls_models[selected]

    # ── metrics ───────────────────────────────────────────────────────────────
    st.subheader("📋 Model Metrics")
    y_pred_val  = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{accuracy_score(yc_test, y_pred_test):.3f}")
    col2.metric("F1 Weighted", f"{f1_score(yc_test, y_pred_test, average='weighted'):.3f}")
    col3.metric("Precision", f"{precision_score(yc_test, y_pred_test, average='weighted', zero_division=0):.3f}")
    col4.metric("Recall",    f"{recall_score(yc_test, y_pred_test, average='weighted', zero_division=0):.3f}")

    # ── hyperparameters ───────────────────────────────────────────────────────
    st.subheader("⚙️ Best Hyperparameters")
    best_params = meta["cls_tuning"][selected].get("best_params", {})
    st.json(best_params)

    # ── plots row ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_confusion_matrix(model, X_test, yc_test, f"Confusion Matrix — {selected}"))
    with col2:
        scorer = make_scorer(f1_score, average="weighted")
        st.pyplot(plot_learning_curve(
            model, X_train, yc_train, scorer,
            f"Learning Curve — {selected}"
        ))

    # ── SHAP ──────────────────────────────────────────────────────────────────
    st.subheader("🔍 SHAP Feature Importance")
    with st.spinner("Computing SHAP values..."):
        st.pyplot(plot_shap(model, X_test, FEATURES, selected))

    # ── prediction input ──────────────────────────────────────────────────────
    st.subheader("🔮 Make a Prediction")
    st.markdown("Enter feature values to predict the next-day movement:")

    cols   = st.columns(len(FEATURES))
    inputs = []
    for j, feat in enumerate(FEATURES):
        val = cols[j].number_input(feat, value=float(np.mean(X_test[:, j])), format="%.4f")
        inputs.append(val)

    if st.button("Predict", key="cls_predict"):
        x_in   = np.array(inputs).reshape(1, -1)
        pred   = model.predict(x_in)[0]
        label  = LABEL_MAP[pred]
        emoji  = {"crestere": "🟢", "stabil": "🟡", "scadere": "🔴"}
        st.success(f"Prediction: {emoji[label]} **{label.upper()}**")

        # SHAP for this single prediction
        try:
            background = shap.sample(X_test, 50)
            explainer  = shap.Explainer(model.predict, background)
            shap_vals  = explainer(x_in)
            fig, ax    = plt.subplots(figsize=(8, 3))
            shap.waterfall_plot(shap_vals[0], show=False)
            st.pyplot(fig)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
def regression_page():
    st.title("📉 ETF Regression")
    st.markdown("""
    **Dataset:** Top 20 ETFs selected by Sharpe ratio, combined into a single dataframe.  
    **Problem:** Predict the **exact next-day % return** of the ETF price.  
    A positive value means price is expected to rise, negative means fall.
    """)

    # ── EDA ───────────────────────────────────────────────────────────────────
    st.subheader("📊 Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(yr_test, bins=50, color="steelblue", edgecolor="white")
        ax.set_title("Target Distribution (Next-Day % Return)")
        ax.set_xlabel("Return %")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        corr    = pd.DataFrame(
            np.column_stack([X_test, yr_test]),
            columns=FEATURES + ["Target_Reg"]
        ).corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar=False)
        ax.set_title("Feature + Target Correlation")
        st.pyplot(fig)

    # ── model selector ────────────────────────────────────────────────────────
    st.subheader("🤖 Model Selection")
    selected = st.selectbox("Select a model:", list(reg_models.keys()))
    model    = reg_models[selected]

    # ── metrics ───────────────────────────────────────────────────────────────
    st.subheader("📋 Model Metrics")
    y_pred_test = model.predict(X_test)
    y_pred_val  = model.predict(X_val)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE  (test)",  f"{mean_absolute_error(yr_test, y_pred_test):.4f}")
    col2.metric("RMSE (test)",  f"{np.sqrt(mean_squared_error(yr_test, y_pred_test)):.4f}")
    col3.metric("R²   (test)",  f"{r2_score(yr_test, y_pred_test):.4f}")
    col4.metric("MAE  (val)",   f"{mean_absolute_error(yr_val, y_pred_val):.4f}")

    # ── hyperparameters ───────────────────────────────────────────────────────
    st.subheader("⚙️ Best Hyperparameters")
    best_params = meta["reg_tuning"][selected].get("best_params", {})
    st.json(best_params)

    # ── plots row ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(yr_test, y_pred_test, alpha=0.3, color="steelblue", s=10)
        mn, mx = yr_test.min(), yr_test.max()
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
        ax.set_title(f"Actual vs Predicted — {selected}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

    with col2:
        scorer = make_scorer(r2_score)
        st.pyplot(plot_learning_curve(
            model, X_train, yr_train, scorer,
            f"Learning Curve — {selected}"
        ))

    # ── SHAP ──────────────────────────────────────────────────────────────────
    st.subheader("🔍 SHAP Feature Importance")
    with st.spinner("Computing SHAP values..."):
        st.pyplot(plot_shap(model, X_test, FEATURES, selected))

    # ── prediction input ──────────────────────────────────────────────────────
    st.subheader("🔮 Make a Prediction")
    st.markdown("Enter feature values to predict the next-day % return:")

    cols   = st.columns(len(FEATURES))
    inputs = []
    for j, feat in enumerate(FEATURES):
        val = cols[j].number_input(feat, value=float(np.mean(X_test[:, j])), format="%.4f")
        inputs.append(val)

    if st.button("Predict", key="reg_predict"):
        x_in  = np.array(inputs).reshape(1, -1)
        pred  = model.predict(x_in)[0]
        color = "green" if pred > 0 else "red"
        st.markdown(f"Predicted next-day return: :{color}[**{pred:.4f}%**]")

        try:
            background = shap.sample(X_test, 50)
            explainer  = shap.Explainer(model.predict, background)
            shap_vals  = explainer(x_in)
            fig, ax    = plt.subplots(figsize=(8, 3))
            shap.waterfall_plot(shap_vals[0], show=False)
            st.pyplot(fig)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
page = st.sidebar.radio(
    "Navigate",
    ["📈 Classification", "📉 Regression"]
)

if page == "📈 Classification":
    classification_page()
else:
    regression_page()
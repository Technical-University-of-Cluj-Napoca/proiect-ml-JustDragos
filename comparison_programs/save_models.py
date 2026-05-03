# save_models.py
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from interpret.glassbox import ExplainableBoostingRegressor

FEATURES = ["Adj Close", "Volume", "daily_range", "volume_zscore", "range_zscore"]
NEEDS_SCALING = {"Logistic Regression", "SVM", "KNN",
                 "Linear Regression", "SVR", "KNN Regressor"}
os.makedirs("saved_models", exist_ok=True)


def get_sklearn_classifier(name: str, params: dict):
    base = {
        "Naive Bayes":         GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree":       DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest":       RandomForestClassifier(class_weight="balanced", random_state=42),
        "SVM":                 SVC(class_weight="balanced", random_state=42, probability=True),
        "KNN":                 KNeighborsClassifier(),
        "XGBoost":             XGBClassifier(eval_metric="mlogloss", random_state=42, verbosity=0),
        "CatBoost":            CatBoostClassifier(verbose=0, random_state=42),
        "EBM":                 ExplainableBoostingClassifier(random_state=42),
    }
    model = base[name]
    # strip pipeline prefix from tuned params
    clean_params = {k.replace("model__", ""): v for k, v in params.items()}
    model.set_params(**clean_params)
    if name in NEEDS_SCALING:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return model


def get_sklearn_regressor(name: str, params: dict):
    base = {
        "Linear Regression":  LinearRegression(),
        "Decision Tree":      DecisionTreeRegressor(random_state=42),
        "Random Forest":      RandomForestRegressor(random_state=42),
        "SVR":                SVR(),
        "KNN Regressor":      KNeighborsRegressor(),
        "XGBoost":            XGBRegressor(random_state=42, verbosity=0),
        "CatBoost":           CatBoostRegressor(verbose=0, random_state=42),
        "EBM":                ExplainableBoostingRegressor(random_state=42),
    }
    model = base[name]
    clean_params = {k.replace("model__", ""): v for k, v in params.items()}
    model.set_params(**clean_params)
    if name in NEEDS_SCALING:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return model


def save_all(df, cls_results, reg_results, cls_tuning, reg_tuning):
    X = df[FEATURES].values
    y_cls = df["Target_Cls_float"].values.astype(int)
    y_reg = df["Target_Reg"].values

    # 75/12.5/12.5 split
    X_train, X_temp, yc_train, yc_temp = train_test_split(X, y_cls, test_size=0.25, shuffle=False)
    X_val,   X_test, yc_val,   yc_test = train_test_split(X_temp, yc_temp, test_size=0.5, shuffle=False)
    _,       _,      yr_train, yr_temp = train_test_split(y_reg, y_reg, test_size=0.25, shuffle=False)  
    yr_train = y_reg[:len(X_train)]
    yr_val   = y_reg[len(X_train):len(X_train) + len(X_val)]
    yr_test  = y_reg[len(X_train) + len(X_val):]

    # SMOTE on classification training only
    smote = SMOTE(random_state=42)
    X_train_bal, yc_train_bal = smote.fit_resample(X_train, yc_train)

    # ── top 5 classifiers ─────────────────────────────────────────────────────
    top5_cls = [
        name for name, _ in
        sorted(cls_results.items(), key=lambda x: x[1]["test_f1"], reverse=True)[:5]
    ]

    cls_models = {}
    for name in top5_cls:
        print(f"  Fitting classifier: {name}")
        best_params = cls_tuning[name]["best_params"]  # from tune_top5
        model = get_sklearn_classifier(name, best_params)
        model.fit(X_train_bal, yc_train_bal)
        cls_models[name] = model
        joblib.dump(model, f"saved_models/cls_{name.replace(' ', '_')}.pkl")

    # ── top 5 regressors ──────────────────────────────────────────────────────
    top5_reg = [
        name for name, _ in
        sorted(reg_results.items(), key=lambda x: x[1]["test_f1"], reverse=True)[:5]
    ]

    reg_models = {}
    for name in top5_reg:
        print(f"  Fitting regressor: {name}")
        best_params = reg_tuning[name]["best_params"]
        model = get_sklearn_regressor(name, best_params)
        model.fit(X_train, yr_train)
        reg_models[name] = model
        joblib.dump(model, f"saved_models/reg_{name.replace(' ', '_')}.pkl")

    # ── save metadata ─────────────────────────────────────────────────────────
    joblib.dump({
        "features":    FEATURES,
        "top5_cls":    top5_cls,
        "top5_reg":    top5_reg,
        "cls_results": cls_results,
        "reg_results": reg_results,
        "cls_tuning":  cls_tuning,
        "reg_tuning":  reg_tuning,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "yc_train": yc_train, "yc_val": yc_val, "yc_test": yc_test,
        "yr_train": yr_train, "yr_val": yr_val, "yr_test": yr_test,
        "X_train_bal": X_train_bal, "yc_train_bal": yc_train_bal,
    }, "saved_models/metadata.pkl")

    print("\nAll models and metadata saved to saved_models/")
    return cls_models, reg_models
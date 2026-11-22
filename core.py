"""Fungsi bersama: data, split, model, dan metrik (di-cache agar cepat)."""
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

RANDOM_STATE = 42
BASE = os.path.dirname(os.path.abspath(__file__))
DASAR = ["radius", "texture", "perimeter", "area", "smoothness",
         "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"]
FITUR = ([f"{f}_mean" for f in DASAR] + [f"{f}_se" for f in DASAR]
         + [f"{f}_worst" for f in DASAR])
WARNA_B, WARNA_M = "#4C9F70", "#D1495B"
NAMA_MODEL = ["Logistic Regression", "KNN", "SVM (RBF)",
              "Decision Tree", "Random Forest", "Gradient Boosting"]


def judul(teks, ikon=""):
    st.title(f"{ikon} {teks}".strip())


@st.cache_data
def muat_data():
    df = pd.read_csv(os.path.join(BASE, "data", "wdbc.csv"))
    return df


@st.cache_data
def siapkan():
    df = muat_data()
    X = df[FITUR]
    y = (df["diagnosis"] == "M").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    return X, y, X_train, X_test, y_train, y_test


def _scaler(m):
    return Pipeline([("scaler", StandardScaler()), ("model", m)])


def buat_model():
    return {
        "Logistic Regression": _scaler(LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
        "KNN": _scaler(KNeighborsClassifier(n_neighbors=7, weights="distance")),
        "SVM (RBF)": _scaler(SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


@st.cache_resource(show_spinner="Melatih 6 model (sekali saja)...")
def model_terlatih():
    """Latih semua model pada data latih; kembalikan model + tabel evaluasi."""
    _, _, X_train, X_test, y_train, y_test = siapkan()
    models = buat_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_rows, test_rows, prob, pred = {}, [], {}, {}
    for nama, m in models.items():
        s = cross_validate(m, X_train, y_train, cv=cv, scoring=scoring)
        cv_rows[nama] = {k: (s[f"test_{k}"].mean(), s[f"test_{k}"].std()) for k in scoring}
        m.fit(X_train, y_train)
        p, pb = m.predict(X_test), m.predict_proba(X_test)[:, 1]
        pred[nama], prob[nama] = p, pb
        tn, fp, fn, tp = confusion_matrix(y_test, p).ravel()
        test_rows.append({"Model": nama, "Accuracy": accuracy_score(y_test, p),
                          "Precision": precision_score(y_test, p),
                          "Recall": recall_score(y_test, p),
                          "F1-Score": f1_score(y_test, p),
                          "ROC-AUC": roc_auc_score(y_test, pb),
                          "TP": tp, "FN": fn, "FP": fp, "TN": tn})
    tabel_test = pd.DataFrame(test_rows).set_index("Model").sort_values(
        ["Recall", "F1-Score"], ascending=False)
    return models, tabel_test, cv_rows, pred, prob


def vif_standar(X):
    """VIF terstandardisasi: 1/(1-R2) tiap fitur terhadap fitur lain."""
    from sklearn.linear_model import LinearRegression
    Z = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    out = {}
    for c in Z.columns:
        lr = LinearRegression().fit(Z.drop(columns=c), Z[c])
        r2 = lr.score(Z.drop(columns=c), Z[c])
        out[c] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(out)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
from core import siapkan, model_terlatih, NAMA_MODEL

st.title("🤖 Modelling & Evaluasi")
X, y, X_train, X_test, y_train, y_test = siapkan()
models, tabel, cv_rows, pred, prob = model_terlatih()

st.markdown("""
| Algoritma | Alasan | Scaling |
|---|---|---|
| Logistic Regression (L2) | Baseline linear, stabil terhadap VIF tinggi | Ya |
| KNN (k = 7) | Berbasis jarak, pola lokal | Ya |
| SVM (RBF) | Batas non-linear | Ya |
| Decision Tree | Mudah diinterpretasi | Tidak |
| Random Forest | Menurunkan varians pohon | Tidak |
| Gradient Boosting | Andal untuk data tabular | Tidak |
""")
st.write(f"Split 80:20 terstratifikasi ({len(X_train)} latih : {len(X_test)} uji), Stratified 5-Fold CV, "
         "`class_weight='balanced'` pada LR, SVM, DT, RF. Kelas positif = **ganas (M)**. "
         "*False Negative* (ganas terprediksi jinak) paling berbahaya, jadi **Recall** adalah metrik utama.")

t1, t2, t3, t4 = st.tabs(["5-Fold CV", "Data uji", "Confusion matrix & ROC", "Model terbaik"])
with t1:
    cv = pd.DataFrame({m: {k: f"{v[0]:.4f} ± {v[1]:.4f}" for k, v in r.items()} for m, r in cv_rows.items()}).T
    cv.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    st.dataframe(cv, width="stretch")
    st.caption("Rata-rata ± simpangan baku pada data latih.")
with t2:
    n_g, n_j = int((y_test == 1).sum()), int((y_test == 0).sum())
    st.write(f"Data uji: **{len(y_test)} pasien** ({n_g} ganas, {n_j} jinak). Diurutkan berdasarkan Recall lalu F1.")
    st.dataframe(tabel.style.format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]})
                 .background_gradient(subset=["Recall", "F1-Score"], cmap="Blues"), width="stretch")
    st.plotly_chart(px.bar(tabel.reset_index(), x="Model", y=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                           barmode="group", range_y=[.85, 1.005], height=420), width="stretch")
with t3:
    pilih = st.selectbox("Model", NAMA_MODEL, index=2)
    cm = confusion_matrix(y_test, pred[pilih])
    a, b = st.columns(2)
    a.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred B", "Pred M"], y=["Aktual B", "Aktual M"],
                             color_continuous_scale="Blues", title=f"Confusion matrix: {pilih}"), width="stretch")
    fig = go.Figure()
    for nama in NAMA_MODEL:
        fpr, tpr, _ = roc_curve(y_test, prob[nama])
        fig.add_scatter(x=fpr, y=tpr, name=f"{nama} ({roc_auc_score(y_test, prob[nama]):.3f})",
                        line=dict(width=4 if nama == pilih else 1.5))
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="Acak")
    fig.update_layout(title="Kurva ROC", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    b.plotly_chart(fig, width="stretch")
    st.text(classification_report(y_test, pred[pilih], target_names=["Jinak (B)", "Ganas (M)"]))
with t4:
    terbaik = tabel.index[0]
    r = tabel.loc[terbaik]
    st.success(f"Model terbaik pada data uji (Recall lalu F1): **{terbaik}**")
    c = st.columns(4)
    c[0].metric("Recall", f"{r['Recall']:.3f}"); c[1].metric("Accuracy", f"{r['Accuracy']:.3f}")
    c[2].metric("ROC-AUC", f"{r['ROC-AUC']:.3f}"); c[3].metric("Ganas terlewat (FN)", int(r["FN"]))
    t = tabel.reset_index()
    fig = go.Figure()
    fig.add_bar(x=t["Model"], y=t["FN"], name="FN (ganas terlewat)", marker_color="#D1495B")
    fig.add_bar(x=t["Model"], y=t["FP"], name="FP (jinak salah ganas)", marker_color="#6C8EBF")
    fig.update_layout(barmode="group", title="Kesalahan per model (data uji)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Angka dihitung ulang di aplikasi ini; hasil bisa sedikit bergeser antar versi scikit-learn, terutama pada model pohon.")

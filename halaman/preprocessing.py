import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from core import muat_data, siapkan, FITUR, RANDOM_STATE

st.title("🛠️ Preprocessing")
df = muat_data()
X, y, X_train, X_test, y_train, y_test = siapkan()

st.markdown("""
Langkah persiapan data yang dipakai semua model:
1. **Buang kolom `id`** (unik 100%, berisiko kebocoran data).
2. **Encode target**: B = 0 (jinak), M = 1 (ganas).
3. **Split 80:20 terstratifikasi**, `random_state = 42`.
4. **StandardScaler** dibungkus *Pipeline* agar tidak bocor ke data uji.
""")
c = st.columns(4)
c[0].metric("Data latih", len(X_train)); c[1].metric("Data uji", len(X_test))
c[2].metric("% ganas (latih)", f"{y_train.mean() * 100:.1f}%"); c[3].metric("% ganas (uji)", f"{y_test.mean() * 100:.1f}%")

st.subheader("Efek penskalaan")
f = st.selectbox("Pilih fitur", FITUR, index=3)
sc = StandardScaler().fit(X_train[[f]])
z = sc.transform(X_train[[f]]).ravel()
a, b = st.columns(2)
a.plotly_chart(px.histogram(X_train, x=f, nbins=30, title="Sebelum (skala asli)"), width="stretch")
b.plotly_chart(px.histogram(x=z, nbins=30, title="Sesudah StandardScaler", labels={"x": f}), width="stretch")
st.caption("Scaler dipelajari (fit) hanya dari data latih. Bentuk distribusi tetap, tetapi skala menjadi mean 0 dan std 1.")

st.subheader("Seleksi fitur dengan RFE")
n_fitur = st.slider("Jumlah fitur yang dipertahankan", 2, 30, 10)


@st.cache_data
def jalankan_rfe(k):
    s = StandardScaler().fit(X_train)
    Xtr, Xte = s.transform(X_train), s.transform(X_test)
    base = LogisticRegression(max_iter=5000).fit(Xtr, y_train)
    acc0 = accuracy_score(y_test, base.predict(Xte))
    rfe = RFE(LogisticRegression(max_iter=5000), n_features_to_select=k).fit(Xtr, y_train)
    m = LogisticRegression(max_iter=5000).fit(rfe.transform(Xtr), y_train)
    p = m.predict(rfe.transform(Xte))
    return acc0, accuracy_score(y_test, p), [f for f, s_ in zip(FITUR, rfe.support_) if s_], confusion_matrix(y_test, p)


acc0, acc1, terpilih, cm = jalankan_rfe(n_fitur)
c = st.columns(3)
c[0].metric("Akurasi 30 fitur", f"{acc0 * 100:.2f}%")
c[1].metric(f"Akurasi {n_fitur} fitur", f"{acc1 * 100:.2f}%", f"{(acc1 - acc0) * 100:+.2f} poin")
c[2].metric("Fitur dibuang", 30 - n_fitur)
st.write("**Fitur terpilih:**", ", ".join(f"`{t}`" for t in terpilih))
st.caption("Metode: Logistic Regression + Recursive Feature Elimination pada data latih terskala (model akhir di halaman Modelling tetap memakai 30 fitur).")

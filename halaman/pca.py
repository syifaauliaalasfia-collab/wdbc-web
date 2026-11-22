import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from core import muat_data, siapkan, FITUR, WARNA_B, WARNA_M

st.title("🔵 Reduksi Dimensi (PCA)")
df = muat_data()
X, y, X_train, X_test, y_train, y_test = siapkan()
st.write("PCA mengubah 30 fitur yang saling berkorelasi menjadi komponen baru yang tidak berkorelasi. "
         "**Standarisasi wajib** sebelum PCA karena PCA sensitif terhadap skala.")

Xs = StandardScaler().fit_transform(X)
pca = PCA().fit(Xs)
ev = pca.explained_variance_ratio_
cum = np.cumsum(ev)

t1, t2, t3, t4 = st.tabs(["Scree & varians kumulatif", "Scatter PC1-PC2", "Loading", "Dampak ke model"])
with t1:
    fig = go.Figure()
    fig.add_bar(x=list(range(1, 31)), y=ev, name="Per komponen")
    fig.add_scatter(x=list(range(1, 31)), y=cum, name="Kumulatif", mode="lines+markers")
    fig.add_hline(y=.95, line_dash="dash", line_color="red", annotation_text="95%")
    fig.update_layout(xaxis_title="Principal Component", yaxis_title="Explained variance ratio", height=420)
    st.plotly_chart(fig, width="stretch")
    c = st.columns(4)
    for col, thr in zip(c, [.8, .9, .95, .99]):
        col.metric(f"Varians {int(thr * 100)}%", f"{int(np.argmax(cum >= thr)) + 1} komponen")
with t2:
    Z = PCA(n_components=2).fit_transform(Xs)
    p2 = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "Diagnosis": df["diagnosis"]})
    st.plotly_chart(px.scatter(p2, x="PC1", y="PC2", color="Diagnosis", opacity=.75,
                               color_discrete_map={"B": WARNA_B, "M": WARNA_M}, height=480), width="stretch")
    st.write(f"PC1 + PC2 menjelaskan **{(ev[0] + ev[1]) * 100:.1f}%** varians. Kelas M dan B terpisah cukup baik dengan sedikit tumpang tindih.")
with t3:
    load = pd.DataFrame(pca.components_[:2].T, columns=["PC1", "PC2"], index=FITUR)
    pc = st.radio("Komponen", ["PC1", "PC2"], horizontal=True)
    top = load[pc].abs().sort_values(ascending=False).head(10).index
    st.plotly_chart(px.bar(load.loc[top, pc].sort_values(), orientation="h", title=f"10 fitur paling berpengaruh pada {pc}"),
                    width="stretch")
with t4:
    def skor(pipe):
        return accuracy_score(y_test, pipe.fit(X_train, y_train).predict(X_test))
    full = skor(Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000))]))
    p95 = Pipeline([("s", StandardScaler()), ("p", PCA(n_components=.95)), ("c", LogisticRegression(max_iter=1000))])
    a95 = skor(p95); k95 = int(p95.named_steps["p"].n_components_)
    a2 = skor(Pipeline([("s", StandardScaler()), ("p", PCA(n_components=2)), ("c", LogisticRegression(max_iter=1000))]))
    tab = pd.DataFrame({"Skenario": ["30 fitur asli", f"PCA 95% ({k95} komponen)", "PCA 2 komponen"],
                        "Jumlah fitur": [30, k95, 2], "Akurasi": [full, a95, a2]})
    st.dataframe(tab.style.format({"Akurasi": "{:.4f}"}), hide_index=True, width="stretch")
    st.caption("Logistic Regression pada split 80:20 yang sama. PCA dipelajari di dalam Pipeline, hanya dari data latih.")

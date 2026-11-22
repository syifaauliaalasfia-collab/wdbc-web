import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.stats import zscore
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from core import muat_data, siapkan, vif_standar, FITUR, RANDOM_STATE, WARNA_B, WARNA_M

st.title("📊 Eksplorasi Data (EDA)")
df = muat_data()
X, y, X_train, X_test, y_train, y_test = siapkan()
CMAP = {"B": WARNA_B, "M": WARNA_M}

t1, t2, t3, t4, t5 = st.tabs(["Distribusi", "Outlier", "Korelasi", "Asosiasi dengan target", "10 masalah kritis"])

with t1:
    f = st.selectbox("Pilih fitur", FITUR, index=3, key="dist")
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x=f, nbins=30, title=f"Histogram {f}"), width="stretch")
    b.plotly_chart(px.histogram(df, x=f, color="diagnosis", nbins=30, barmode="overlay", opacity=.6,
                                color_discrete_map=CMAP, title="Per diagnosis"), width="stretch")
    st.write(f"Skewness {f}: **{X[f].skew():.3f}**")
    sk = X.skew().sort_values(ascending=False).round(3).rename("skewness")
    with st.expander("Skewness semua fitur"):
        st.dataframe(sk)

with t2:
    Q1, Q3 = X.quantile(.25), X.quantile(.75)
    IQR = Q3 - Q1
    iqr_out = ((X < Q1 - 1.5 * IQR) | (X > Q3 + 1.5 * IQR)).sum()
    z_out = (np.abs(X.apply(zscore)) > 3).sum()
    tab = pd.DataFrame({"Z-Score (|Z|>3)": z_out, "IQR": iqr_out}).sort_values("IQR", ascending=False)
    a, b = st.columns([1, 1])
    a.dataframe(tab, height=420)
    f = b.selectbox("Boxplot fitur", FITUR, index=13, key="box")
    b.plotly_chart(px.box(df, x="diagnosis", y=f, color="diagnosis", color_discrete_map=CMAP), width="stretch")
    st.caption("Outlier tidak dihapus: pada pasien dengan nilai ekstrem, proporsi kasus ganas biasanya tinggi (lihat EDA 10).")

with t3:
    kor = X.corr()
    st.plotly_chart(px.imshow(kor, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, height=700), width="stretch")
    atas = kor.where(np.triu(np.ones(kor.shape), 1).astype(bool)).stack().dropna()
    c = st.columns(3)
    c[0].metric("Pasangan |r| > 0,8", f"{int((atas.abs() > .8).sum())} / {len(atas)}")
    c[1].metric("Pasangan |r| > 0,9", f"{int((atas.abs() > .9).sum())} / {len(atas)}")
    c[2].metric("Pasangan |r| > 0,95", f"{int((atas.abs() > .95).sum())} / {len(atas)}")
    vif = vif_standar(X).sort_values(ascending=False)
    st.write(f"**VIF terstandardisasi:** {int((vif > 10).sum())} dari 30 fitur ber-VIF > 10 (multikolinearitas tinggi).")
    with st.expander("Tabel VIF"):
        st.dataframe(vif.round(1).rename("VIF"))

with t4:
    pr = [stats.pearsonr(X[c], y)[0] for c in FITUR]
    sp = [stats.spearmanr(X[c], y)[0] for c in FITUR]
    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    asos = pd.DataFrame({"Pearson": pr, "Spearman": sp, "Mutual Info": mi}, index=FITUR)
    urut = asos["Pearson"].abs().sort_values(ascending=False).index
    st.dataframe(asos.loc[urut].round(4), height=420)
    top = asos.loc[urut[:10], "Pearson"].sort_values()
    st.plotly_chart(px.bar(top, orientation="h", title="10 fitur dengan korelasi Pearson tertinggi terhadap target (M=1)"),
                    width="stretch")

with t5:
    st.write("Adaptasi panduan *10 Masalah Kritis EDA* ke WDBC. Angka di bawah dihitung langsung dari data.")
    data = X.copy(); data["target"] = y.values; data["diagnosis"] = df["diagnosis"].values
    hasil = []
    # 1 leakage
    Xb = StandardScaler().fit_transform(X)
    a_, b_, c_, d_ = train_test_split(Xb, y, test_size=.2, random_state=RANDOM_STATE, stratify=y)
    acc_bocor = LogisticRegression(max_iter=5000).fit(a_, c_).score(b_, d_)
    pipe = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=5000))])
    acc_iso = pipe.fit(X_train, y_train).score(X_test, y_test)
    hasil.append(("1. Data leakage", f"Akurasi uji {acc_bocor:.4f} (bocor)", f"{acc_iso:.4f} (Pipeline)", "Bungkus scaler dalam Pipeline"))
    # 2 skew
    sk0 = X["area_mean"].skew(); sk1 = np.log1p(X["area_mean"]).skew()
    hasil.append(("2. Skewness area_mean", f"{sk0:.3f}", f"{sk1:.3f}", "Log transform + RobustScaler"))
    # 3 missing MNAR
    rng = np.random.default_rng(RANDOM_STATE)
    sim = data.copy()
    mask = rng.random(len(sim)) < np.where(sim["target"] == 1, .35, .05)
    sim.loc[mask, "concave_points_worst"] = np.nan
    n_drop = len(sim.dropna())
    hasil.append(("3. Missing non-random (simulasi)", f"dropna: {n_drop} dari {len(sim)} baris", f"KNNImputer: {len(sim)} baris", "Indikator missing + imputasi"))
    # 4 VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor as vf
    xv = data[["radius_mean", "perimeter_mean", "area_mean", "texture_mean", "smoothness_mean"]].copy(); xv["c"] = 1
    v0 = vf(xv.values, 0)
    pilih = ["radius_mean", "texture_mean", "smoothness_mean", "compactness_mean"]
    xs = pd.DataFrame(StandardScaler().fit_transform(data[pilih]), columns=pilih)
    v1 = max(vf(xs.values, i) for i in range(xs.shape[1]))
    hasil.append(("4. Multikolinearitas", f"VIF radius_mean {v0:.1f}", f"VIF maks {v1:.2f}", "Buang fitur redundan"))
    # 5 imbalance
    dt_ = pd.concat([data[data.target == 0], data[data.target == 1].sample(20, random_state=RANDOM_STATE)])
    Xi = dt_[["radius_mean", "texture_mean", "smoothness_mean", "concavity_mean"]]; yi = dt_["target"]
    mb = LogisticRegression(class_weight="balanced", max_iter=5000).fit(Xi, yi)
    hasil.append(("5. Paradoks akurasi (90:10)", f"Dummy akurasi {(1 - yi.mean()) * 100:.1f}%, F1 makro {f1_score(yi, np.zeros(len(yi)), average='macro', zero_division=0):.3f}",
                  f"F1 makro {f1_score(yi, mb.predict(Xi), average='macro'):.3f}; AUC {roc_auc_score(yi, mb.predict_proba(Xi)[:, 1]):.3f}", "class_weight + F1/ROC-AUC"))
    # 6 Pearson
    hasil.append(("6. Bias linearitas Pearson", "Pearson saja bisa meremehkan fitur", "Spearman + Mutual Information", "Lihat tab Asosiasi"))
    # 7 ID
    imp = pd.Series(DecisionTreeClassifier(random_state=RANDOM_STATE).fit(
        pd.DataFrame({"id": df["id"], "radius_mean": X["radius_mean"], "texture_mean": X["texture_mean"]}), y).feature_importances_,
        index=["id", "radius_mean", "texture_mean"])
    hasil.append(("7. ID berkardinalitas tinggi", f"ID mengambil {imp['id'] * 100:.0f}% importance", "Dibuang dari fitur", "Hapus kolom ID"))
    # 8 masking
    am = pd.concat([X["area_mean"], pd.Series([6000.0])]).reset_index(drop=True)
    nz = int((np.abs(zscore(am)) > 3).sum())
    med = np.median(am); mad = np.median(np.abs(am - med))
    nmz = int((0.6745 * np.abs(am - med) / mad > 3.5).sum())
    hasil.append(("8. Masking effect", f"Z-Score: {nz} pencilan", f"Modified Z-Score: {nmz} pencilan", "Median + MAD"))
    # 9 multimodal
    tp = stats.ttest_ind(X.loc[y == 0, "radius_mean"], X.loc[y == 1, "radius_mean"])[1]
    hasil.append(("9. Subpopulasi tersembunyi", f"Mean global radius {X['radius_mean'].mean():.2f}", f"B {X.loc[y == 0, 'radius_mean'].mean():.2f} vs M {X.loc[y == 1, 'radius_mean'].mean():.2f} (p={tp:.1e})", "Dekomposisi per kelas"))
    # 10 outlier valid
    q1, q3 = X["area_worst"].quantile([.25, .75]); ub = q3 + 1.5 * (q3 - q1)
    o = data[data["area_worst"] > ub]
    hasil.append(("10. Outlier valid", f"{len(o)} outlier area_worst, {o.target.mean() * 100:.1f}% ganas", f"Populasi {y.mean() * 100:.1f}% ganas", "Yeo-Johnson, jangan dihapus"))
    st.dataframe(pd.DataFrame(hasil, columns=["Masalah", "Sebelum mitigasi", "Sesudah / pembanding", "Mitigasi"]),
                 hide_index=True, width="stretch")
    st.caption("Catatan: poin 3 dan 5 memakai simulasi karena WDBC asli tanpa missing value dan kelasnya tidak 90:10.")

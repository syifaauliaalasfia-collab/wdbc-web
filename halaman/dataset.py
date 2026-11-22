import pandas as pd
import plotly.express as px
import streamlit as st
from core import muat_data, FITUR, DASAR, WARNA_B, WARNA_M

st.title("🗃️ Tentang Dataset")
df = muat_data()
st.write("**Sumber:** UCI Machine Learning Repository, *Breast Cancer Wisconsin (Diagnostic)*. "
         "Fitur dihitung dari citra *Fine Needle Aspirate* (FNA) massa payudara, menggambarkan inti sel.")

st.markdown("""
Setiap pasien punya **1 ID + 1 diagnosis + 30 fitur**. Ke-30 fitur adalah **10 karakteristik inti sel**
dalam **3 versi**: *mean* (rata-rata), *se* (standard error), dan *worst* (rata-rata tiga nilai terbesar).
""")
st.table(pd.DataFrame({
    "Karakteristik": [d.replace("_", " ") for d in DASAR],
    "Keterangan": ["Rata-rata jarak pusat ke tepi", "Simpangan baku nilai abu-abu", "Keliling inti",
                   "Luas inti", "Variasi lokal panjang radius", "Keliling² / luas − 1",
                   "Tingkat keparahan bagian cekung kontur", "Jumlah bagian cekung kontur",
                   "Simetri inti", "Aproksimasi garis pantai − 1"]}))

tab1, tab2, tab3 = st.tabs(["Cuplikan data", "Statistik deskriptif", "Kualitas data"])
with tab1:
    st.dataframe(df.head(20), width="stretch")
    st.caption(f"Ukuran data: {df.shape[0]} baris × {df.shape[1]} kolom.")
with tab2:
    st.dataframe(df[FITUR].describe().T.round(3), width="stretch")
with tab3:
    c = st.columns(4)
    c[0].metric("Missing value", int(df.isna().sum().sum()))
    c[1].metric("Baris duplikat", int(df.duplicated().sum()))
    c[2].metric("ID unik", df["id"].nunique())
    c[3].metric("Rasio N/d", round(len(df) / 30, 1))
    dist = df["diagnosis"].value_counts().rename_axis("Diagnosis").reset_index(name="Jumlah")
    dist["Persen"] = (dist["Jumlah"] / len(df) * 100).round(2)
    a, b = st.columns([1, 2])
    a.dataframe(dist, hide_index=True)
    fig = px.bar(dist, x="Diagnosis", y="Jumlah", color="Diagnosis",
                 color_discrete_map={"B": WARNA_B, "M": WARNA_M}, text="Persen")
    fig.update_layout(showlegend=False, height=300, margin=dict(t=10))
    b.plotly_chart(fig, width="stretch")
    st.markdown("""
- Tidak ada missing value maupun duplikat.
- Outlier ada (metode IQR) tetapi **tidak langsung dihapus**, karena bisa jadi nilai valid.
- Skala antarfitur sangat berbeda (`area_mean` hingga 2501, `smoothness_mean` maks 0,163), jadi **scaling wajib**.
- Kelas tidak seimbang secara moderat (B 62,7% : M 37,3%), jadi akurasi saja tidak cukup.
""")

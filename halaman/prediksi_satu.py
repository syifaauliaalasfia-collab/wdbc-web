import pandas as pd
import streamlit as st
from core import muat_data, siapkan, model_terlatih, FITUR, NAMA_MODEL

st.title("🧑‍⚕️ Prediksi Satu Pasien")
st.write("Isi 30 nilai fitur hasil pemeriksaan FNA, lalu klik **Prediksi**. Gunakan tombol contoh untuk mengisi otomatis.")
df = muat_data()
models, tabel, *_ = model_terlatih()
_, y, *_rest = siapkan()

nama = st.selectbox("Model", NAMA_MODEL, index=NAMA_MODEL.index(tabel.index[0]))
rata = df.groupby("diagnosis")[FITUR].mean()
tengah = df[FITUR].median()

c1, c2, c3 = st.columns(3)
if c1.button("Contoh pasien jinak (rata-rata B)"):
    for f in FITUR: st.session_state[f"in_{f}"] = float(rata.loc["B", f])
if c2.button("Contoh pasien ganas (rata-rata M)"):
    for f in FITUR: st.session_state[f"in_{f}"] = float(rata.loc["M", f])
if c3.button("Reset (median)"):
    for f in FITUR: st.session_state[f"in_{f}"] = float(tengah[f])

kol = st.columns(3)
nilai = {}
for i, grup in enumerate(["mean", "se", "worst"]):
    with kol[i]:
        st.markdown(f"**Fitur _{grup}_**")
        for f in [x for x in FITUR if x.endswith(f"_{grup}")]:
            lo, hi = float(df[f].min()), float(df[f].max())
            st.session_state.setdefault(f"in_{f}", float(tengah[f]))
            nilai[f] = st.number_input(f.replace("_", " "), min_value=0.0, max_value=hi * 3,
                                       step=(hi - lo) / 200 or 0.001, format="%.5f", key=f"in_{f}",
                                       help=f"Rentang data: {lo:g} sampai {hi:g}")

if st.button("Prediksi", type="primary"):
    x = pd.DataFrame([nilai], columns=FITUR)
    m = models[nama]
    p_ganas = float(m.predict_proba(x)[0, 1])
    if p_ganas >= .5:
        st.error(f"Prediksi: **Ganas (Malignant)**, probabilitas {p_ganas * 100:.1f}%")
    else:
        st.success(f"Prediksi: **Jinak (Benign)**, probabilitas ganas {p_ganas * 100:.1f}%")
    st.progress(p_ganas, text=f"Probabilitas ganas {p_ganas * 100:.1f}%")
    st.caption("Ini hasil model untuk keperluan belajar, bukan diagnosis medis. Konsultasikan dengan tenaga kesehatan.")

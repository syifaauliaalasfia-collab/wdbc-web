import streamlit as st
from core import muat_data

st.title("🎗️ Prediksi Kanker Payudara")
st.subheader("Klasifikasi tumor jinak vs ganas dengan Machine Learning")
st.write("Aplikasi ini merangkum seluruh alur proyek pada dataset **Breast Cancer Wisconsin (Diagnostic)**: "
         "memahami data, EDA, preprocessing, reduksi dimensi, pemodelan, hingga prediksi untuk pasien baru.")

df = muat_data()
c = st.columns(4)
c[0].metric("Jumlah pasien", len(df))
c[1].metric("Jumlah fitur", 30)
c[2].metric("Jinak (B)", int((df.diagnosis == "B").sum()))
c[3].metric("Ganas (M)", int((df.diagnosis == "M").sum()))

st.markdown("""
### Alur aplikasi
1. **Tentang Dataset**: asal data, arti fitur, dan kondisi data.
2. **EDA**: distribusi, outlier, korelasi, dan 10 masalah kritis EDA.
3. **Preprocessing**: pembuangan ID, penskalaan, split 80:20, seleksi fitur RFE.
4. **PCA**: berapa komponen yang cukup mewakili 30 fitur.
5. **Modelling & Evaluasi**: 6 algoritma, 5-Fold CV, Accuracy, Precision, Recall, F1, ROC-AUC.
6. **Prediksi**: satu pasien lewat form isian.
7. **Kesimpulan & Saran**.
""")
st.info("Hasil prediksi bersifat alat bantu belajar, bukan diagnosis medis.", icon=":material/info:")
st.caption("Anggota: Fortuna Yuli Astuti, Sabrina Reinkana Aulia W Al Hasbi, Syifa Aulia Al Asfia")

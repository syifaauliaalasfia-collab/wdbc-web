"""Aplikasi Streamlit Klasifikasi Kanker Payudara (WDBC) - Kelompok 6.
Jalankan: streamlit run app.py"""
import os
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)  # halaman Bab 2 membaca wdbc.csv dan korelasi.png secara relatif

st.set_page_config(page_title="Prediksi Kanker Payudara (WDBC)", page_icon="🎗️", layout="wide")

BAB2 = [
    ("Hello World", "01_main.py", "bab2-hello"),
    ("st.write & DataFrame", "02_main1.py", "bab2-write"),
    ("Elemen Teks", "03_teks.py", "bab2-teks"),
    ("Menampilkan Data", "04_data.py", "bab2-data"),
    ("Menampilkan Grafik", "05_grafik.py", "bab2-grafik"),
    ("Input Form", "06_input.py", "bab2-input"),
    ("Media", "07_media.py", "bab2-media"),
    ("Layout & Container", "08_layout.py", "bab2-layout"),
    ("Control Flow", "09_control.py", "bab2-control"),
]


def jalankan_bab2(nama_file):
    def _run():
        path = os.path.join(BASE, "bab2", nama_file)
        with open(path, encoding="utf-8") as f:
            kode = f.read()
        exec(compile(kode, path, "exec"), {"__name__": "__main__"})
    _run.__name__ = "bab2_" + nama_file.replace(".py", "").replace("-", "_")
    return _run


halaman = {
    "Pengantar": [
        st.Page("halaman/beranda.py", title="Beranda", icon=":material/home:", default=True, url_path="beranda"),
        st.Page("halaman/dataset.py", title="Tentang Dataset", icon=":material/database:", url_path="dataset"),
        st.Page("halaman/eda.py", title="Eksplorasi Data (EDA)", icon=":material/bar_chart:", url_path="eda"),
    ],
    "Pengolahan & Model": [
        st.Page("halaman/preprocessing.py", title="Preprocessing", icon=":material/tune:", url_path="preprocessing"),
        st.Page("halaman/pca.py", title="Reduksi Dimensi (PCA)", icon=":material/bubble_chart:", url_path="pca"),
        st.Page("halaman/modelling.py", title="Modelling & Evaluasi", icon=":material/model_training:", url_path="modelling"),
    ],
    "Penerapan": [
        st.Page("halaman/prediksi_satu.py", title="Prediksi Satu Pasien", icon=":material/person_search:", url_path="prediksi"),
        st.Page("halaman/kesimpulan.py", title="Kesimpulan & Saran", icon=":material/task_alt:", url_path="kesimpulan"),
    ],
    "Praktik Streamlit (Bab 2)": [
        st.Page(jalankan_bab2(f), title=t, icon=":material/code:", url_path=u) for t, f, u in BAB2
    ],
}

nav = st.navigation(halaman)
with st.sidebar:
    st.divider()
    st.caption("Tugas Machine Learning - Kelompok 6")
    st.caption("Fortuna Yuli Astuti  \nSabrina Reinkana Aulia W Al Hasbi  \nSyifa Aulia Al Asfia")
nav.run()

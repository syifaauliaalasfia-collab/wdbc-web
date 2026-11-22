# Bab 2.7 - Layout dan container
import streamlit as st
import pandas as pd

df = pd.read_csv("wdbc.csv")
st.sidebar.title("Menu")                         # notasi objek
st.sidebar.selectbox("Pilih fitur", df.columns[1:])
with st.sidebar:                                  # notasi with
    st.radio("Pilih kelas", ["B", "M"])

col1, col2 = st.columns(2)
col1.metric("Benign", int((df.diagnosis == "B").sum()))
col2.metric("Malignant", int((df.diagnosis == "M").sum()))

tab1, tab2 = st.tabs(["Grafik", "Data"])
tab1.bar_chart(df["diagnosis"].value_counts())
tab2.dataframe(df.head(10))

with st.expander("Lihat penjelasan dataset"):
    st.write("WDBC berisi 569 pasien dengan 30 fitur hasil citra FNA massa payudara.")

with st.container():
    st.write("Ini di dalam container")
    st.line_chart(df["radius_mean"].head(30))
st.write("Ini di luar container")

import time
tempat = st.empty()
for i in range(3, 0, -1):
    tempat.write(f"Menghitung mundur: {i}"); time.sleep(1)
tempat.success("Selesai!")

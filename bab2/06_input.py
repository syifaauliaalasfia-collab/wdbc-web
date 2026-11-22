# Bab 2.5 - Input form
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, time

df = pd.read_csv("wdbc.csv")
st.header("Button"); 
if st.button("Tampilkan jumlah data"): st.write("Jumlah data:", len(df))

st.header("Data editor")  # di buku: st.experimental_data_editor
edit = st.data_editor(df.head(5).iloc[:, :5], num_rows="dynamic", use_container_width=True)

st.header("Download button")
st.download_button("Unduh wdbc.csv", df.to_csv(index=False).encode("utf-8"), "wdbc.csv", "text/csv")

st.header("Checkbox, radio, selectbox")
if st.checkbox("Tampilkan 5 baris awal"): st.write(df.head())
kelas = st.radio("Pilih kelas", ["B", "M"]); st.write("Kelas dipilih:", kelas)
fitur = st.selectbox("Pilih fitur", df.columns[1:]); st.write("Fitur:", fitur)

st.header("Multiselect")
pil = st.multiselect("Pilih beberapa fitur", df.columns[1:], default=["radius_mean", "area_mean"])
st.write(df[pil].head())

st.header("Slider")
lo, hi = st.slider("Rentang radius_mean", 6.0, 29.0, (10.0, 20.0))
st.write("Data dalam rentang:", int(df.radius_mean.between(lo, hi).sum()))

st.header("Select slider")
ukuran = st.select_slider("Ukuran sampel", options=[10, 50, 100, 200, 569], value=100)
st.write("Ukuran:", ukuran)

st.header("Text, number, text area")
nama = st.text_input("Nama pasien"); st.write("Nama:", nama)
umur = st.number_input("Umur", min_value=0, max_value=100, step=1); st.write("Umur:", umur)
cat = st.text_area("Catatan dokter"); st.write(cat)

st.header("Date dan time")
st.write("Tanggal periksa:", st.date_input("Tanggal", date.today()))
st.write("Jam periksa:", st.time_input("Jam", time(8, 30)))

st.header("File uploader")
up = st.file_uploader("Upload CSV", type="csv", accept_multiple_files=False)
if up is not None: st.write(pd.read_csv(up).head())

st.header("Camera input")
foto = st.camera_input("Ambil foto")
if foto: st.image(foto)

st.header("Color picker")
st.write("Warna:", st.color_picker("Pilih warna", "#D1495B"))

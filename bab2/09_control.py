# Bab 2.8 - Control flow
import streamlit as st

st.title("Control Flow Streamlit")
if st.button("Stop Aplikasi"):
    st.warning("Aplikasi dihentikan!")
    st.stop()
st.write("Baris ini tidak dieksekusi jika aplikasi dihentikan.")

st.subheader("st.form dan st.form_submit_button")
with st.form("Formulir Data Pasien"):
    c1, c2 = st.columns(2)
    with c1:
        nama = st.text_input("Nama")
        umur = st.number_input("Umur", 0, 100, step=1)
        alamat = st.text_area("Alamat")
    with c2:
        radius = st.number_input("radius_mean", 0.0, 40.0, 14.0)
        tekstur = st.number_input("texture_mean", 0.0, 50.0, 19.0)
        area = st.number_input("area_mean", 0.0, 3000.0, 650.0)
    kirim = st.form_submit_button("Submit")
if kirim:
    st.write("Nama:", nama, "| Umur:", umur, "| Alamat:", alamat)
    st.write("Rata-rata tiga fitur:", round((radius + tekstur + area) / 3, 2))

st.subheader("st.rerun")  # di buku: st.experimental_rerun
if st.button("Rerun"): st.rerun()

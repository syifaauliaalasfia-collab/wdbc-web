# Bab 2.2 - Elemen teks
import streamlit as st

st.markdown("Diagnosis **:red[Malignant]** dan :green[Benign] pada dataset *WDBC* :pencil:")
st.markdown("Contoh rumus akar kuadrat: $\\sqrt{x^2+y^2}$")
st.title("Breast Cancer Wisconsin (Diagnostic)")
st.header("Data Understanding")
st.subheader("569 pasien, 30 fitur, 2 kelas")
st.caption("Sumber data: UCI Machine Learning Repository")
st.code("df = pd.read_csv('wdbc.csv')\nprint(df.shape)", language="python")
st.text("Teks biasa: 212 Malignant dan 357 Benign")
st.latex(r"z = \frac{x - \mu}{\sigma}")
st.divider()
st.write("Teks setelah garis pemisah (divider).")

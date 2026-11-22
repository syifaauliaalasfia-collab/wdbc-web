# Bab 2.3 - Menampilkan data
import streamlit as st
import pandas as pd

df = pd.read_csv("wdbc.csv")
st.subheader("st.dataframe (interaktif)")
st.dataframe(df.sample(10, random_state=1).iloc[:, :6])
st.subheader("st.table (statis)")
st.table(df.head(5).iloc[:, :5])
st.subheader("st.metric")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Sampel", len(df))
c2.metric("Malignant (M)", int((df.diagnosis == "M").sum()), "37,3%")
c3.metric("Benign (B)", int((df.diagnosis == "B").sum()), "62,7%")
st.subheader("st.json")
st.json({"sampel": len(df), "fitur": 30, "kelas": ["B", "M"], "missing_value": 0})

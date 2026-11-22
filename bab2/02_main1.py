# Bab 2.1 - st.write() untuk angka dan DataFrame
import streamlit as st
import pandas as pd

df = pd.read_csv("wdbc.csv")
st.write("Jumlah data:", len(df))
st.write(1234)
st.write(df.head(10))

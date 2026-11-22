# Bab 2.6 - Media. Gambar dibuat dari data WDBC (heatmap korelasi).
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv("wdbc.csv")
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(df.iloc[:, 1:11].corr(), cmap="coolwarm"); fig.colorbar(im)
ax.set_title("Korelasi fitur mean"); fig.savefig("korelasi.png", dpi=100)

st.header("st.image")
st.image(Image.open("korelasi.png"), caption="Heatmap korelasi WDBC")
st.header("st.audio dan st.video")
st.caption("Opsional: letakkan file audio/video sendiri, lalu aktifkan baris di bawah.")
# st.audio("penjelasan.mp3")
# st.video("penjelasan.mp4")

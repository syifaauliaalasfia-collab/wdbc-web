# Bab 2.4 - Menampilkan grafik
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px

df = pd.read_csv("wdbc.csv")
cols = ["radius_mean", "texture_mean", "area_mean"]
cd = df[cols].head(50)

st.subheader("st.line_chart"); st.line_chart(cd[["radius_mean", "texture_mean"]])
st.subheader("st.area_chart"); st.area_chart(cd[["radius_mean", "texture_mean"]])
st.subheader("st.bar_chart"); st.bar_chart(df["diagnosis"].value_counts())

st.subheader("st.altair_chart")
st.altair_chart(alt.Chart(df).mark_circle().encode(
    x="radius_mean", y="area_mean", color="diagnosis", tooltip=["radius_mean", "area_mean"]
).interactive(), use_container_width=True)

st.subheader("st.vega_lite_chart")
st.vega_lite_chart(df, {"mark": {"type": "circle", "tooltip": True},
    "encoding": {"x": {"field": "radius_mean", "type": "quantitative"},
                 "y": {"field": "texture_mean", "type": "quantitative"},
                 "color": {"field": "diagnosis", "type": "nominal"}}},
    use_container_width=True)

st.subheader("st.plotly_chart")
st.plotly_chart(px.scatter(df, x="radius_mean", y="area_mean", color="diagnosis"), use_container_width=True)

st.subheader("st.pyplot")
fig, ax = plt.subplots()
df["area_mean"].plot.hist(bins=30, ax=ax); ax.set_title("Distribusi area_mean")
st.pyplot(fig)

st.subheader("st.map (data WDBC berasal dari University of Wisconsin, Madison)")
st.map(pd.DataFrame({"lat": [43.0731], "lon": [-89.4012]}))

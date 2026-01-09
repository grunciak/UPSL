import streamlit as st
import plotly.express as px

from utils import load_data, infer_target, get_dataset_info, make_filters

st.set_page_config(page_title="Wizualizacje", page_icon="📊", layout="wide")
st.title("📊 Wizualizacje i zależności")

df = load_data("WineQT.csv")
target_col = infer_target(df)
info = get_dataset_info(df, target_col)
filtered = make_filters(df, info.numeric_cols)

st.subheader("Macierz korelacji (numeryczne)")
corr = filtered[info.numeric_cols].corr(numeric_only=True)
fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Korelacje Pearsona")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Scatter: 2 cechy + kolor = target")
x = st.selectbox("Oś X", options=[c for c in info.numeric_cols if c != target_col], index=0)
y_opts = [c for c in info.numeric_cols if c != target_col]
y = st.selectbox("Oś Y", options=y_opts, index=1 if len(y_opts) > 1 else 0)
opacity = st.slider("Przezroczystość", 0.1, 1.0, 0.7)
fig2 = px.scatter(filtered, x=x, y=y, color=target_col, opacity=opacity, title=f"{x} vs {y} (kolor: {target_col})")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Boxplot cech względem jakości")
feat = st.selectbox("Cecha", options=[c for c in info.numeric_cols if c != target_col], index=0)
fig3 = px.box(filtered, x=target_col, y=feat, points="outliers", title=f"{feat} względem {target_col}")
st.plotly_chart(fig3, use_container_width=True)

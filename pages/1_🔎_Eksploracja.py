import streamlit as st
import plotly.express as px

from utils import load_data, infer_target, get_dataset_info, make_filters

st.set_page_config(page_title="Eksploracja danych", page_icon="🔎", layout="wide")
st.title("🔎 Eksploracja danych")

df = load_data("WineQT.csv")
target_col = infer_target(df)
info = get_dataset_info(df, target_col)

st.sidebar.header("Ustawienia")
filtered = make_filters(df, info.numeric_cols)

st.subheader("Tabela (po filtrach)")
st.dataframe(filtered, use_container_width=True, height=420)

c1, c2, c3 = st.columns(3)
c1.metric("Wiersze po filtrach", f"{len(filtered):,}".replace(",", " "))
c2.metric("Braki danych", int(filtered.isna().sum().sum()))
c3.metric("Unikalne wartości targetu", int(filtered[target_col].nunique()) if target_col in filtered.columns else 0)

st.subheader("Statystyki opisowe")
cols = st.multiselect("Wybierz kolumny", options=info.numeric_cols, default=info.numeric_cols)
if cols:
    st.dataframe(filtered[cols].describe().T, use_container_width=True)

st.subheader("Rozkłady (histogram)")
col = st.selectbox("Kolumna", options=info.numeric_cols, index=0)
bins = st.slider("Liczba koszyków (bins)", 10, 100, 30)
fig = px.histogram(filtered, x=col, nbins=bins, marginal="box", title=f"Histogram: {col}")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Pobierz dane po filtrach")
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Pobierz CSV", data=csv, file_name="WineQT_filtered.csv", mime="text/csv")

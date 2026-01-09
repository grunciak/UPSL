import streamlit as st
import plotly.express as px

from utils import load_data, infer_target, get_dataset_info

st.set_page_config(page_title="Jakość danych", page_icon="✅", layout="wide")
st.title("✅ Jakość danych")

df = load_data("WineQT.csv")
target_col = infer_target(df)
info = get_dataset_info(df, target_col)

st.subheader("Braki danych")
miss = df.isna().sum().sort_values(ascending=False)
miss = miss[miss > 0]
if miss.empty:
    st.success("Brak braków danych w datasetcie ✅")
else:
    fig = px.bar(miss.reset_index(), x="index", y=0, title="Liczba braków danych w kolumnach",
                 labels={"index": "Kolumna", 0: "Braki"})
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Duplikaty")
dup = int(df.duplicated().sum())
if dup == 0:
    st.success("Brak zduplikowanych wierszy ✅")
else:
    st.warning(f"Wykryto duplikaty: {dup}. Możesz je usunąć do modelowania.")
    if st.button("Usuń duplikaty (podgląd)"):
        st.dataframe(df.drop_duplicates(), use_container_width=True)

st.subheader("Outliery — szybki przegląd IQR")
feat = st.selectbox("Wybierz cechę", options=[c for c in info.numeric_cols if c != target_col], index=0)
q1 = df[feat].quantile(0.25)
q3 = df[feat].quantile(0.75)
iqr = q3 - q1
lo = q1 - 1.5 * iqr
hi = q3 + 1.5 * iqr
out = df[(df[feat] < lo) | (df[feat] > hi)]

c1, c2, c3 = st.columns(3)
c1.metric("Q1", f"{q1:.3f}")
c2.metric("Q3", f"{q3:.3f}")
c3.metric("Outliery (IQR)", f"{len(out):,}".replace(",", " "))

st.caption("Uwaga: outlier nie zawsze oznacza błąd. W danych laboratoryjnych może być realnym przypadkiem.")
st.dataframe(out[[feat, target_col]].head(50), use_container_width=True)

import streamlit as st

from utils import load_data, infer_target, get_dataset_info

st.set_page_config(
    page_title="WineQT • Analiza jakości wina",
    page_icon="🍷",
    layout="wide",
)

st.title("🍷 WineQT — dashboard analityczny (Streamlit)")
st.caption("Repozytorium gotowe do uruchomienia lokalnie lub na Streamlit Community Cloud (GitHub).")

with st.sidebar:
    st.header("Źródło danych")
    data_path = st.text_input("Ścieżka do pliku CSV", value="WineQT.csv", help="W repozytorium domyślnie jest WineQT.csv")
    st.divider()
    st.markdown("**Nawigacja:** użyj stron w menu po lewej (multipage).")

try:
    df = load_data(data_path)
except Exception as e:
    st.error(f"Nie mogę wczytać pliku: {e}")
    st.stop()

target_col = infer_target(df)
info = get_dataset_info(df, target_col)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Wiersze", f"{info.n_rows:,}".replace(",", " "))
col2.metric("Kolumny", f"{info.n_cols}")
col3.metric("Braki danych", f"{info.missing_total}")
col4.metric("Duplikaty", f"{info.duplicates}")

st.subheader("Podgląd danych")
st.dataframe(df.head(25), use_container_width=True)

st.subheader("Szybkie wnioski")
c1, c2 = st.columns(2)

with c1:
    st.markdown(
        f"""
- Wykryta kolumna celu (target): **`{target_col}`**
- Liczba kolumn numerycznych: **{len(info.numeric_cols)}**
- Jeśli dataset ma kolumnę `Id`, została automatycznie usunięta (to zwykle tylko identyfikator wiersza).
        """
    )

with c2:
    if target_col in df.columns:
        st.markdown("Rozkład targetu (pierwsze wartości):")
        vc = df[target_col].value_counts().sort_index()
        st.bar_chart(vc)

st.info("Przejdź do zakładek po lewej: **Eksploracja**, **Wizualizacje**, **Jakość danych**, **Modelowanie**.")

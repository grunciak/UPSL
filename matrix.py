import streamlit as st
import pandas as pd

st.set_page_config(page_title="Analiza ryzyka", layout="wide")
st.title("🔐 Analiza ryzyka systemów teleinformatycznych")

# Klasyfikacja poziomu ryzyka
def klasyfikuj_ryzyko(poziom):
    if poziom <= 6:
        return "Niskie"
    elif poziom <= 14:
        return "Średnie"
    else:
        return "Wysokie"

# Domyślna lista zagrożeń
default_risks = [
    {"Zagrożenie": "Awaria serwera", "Prawdopodobieństwo": 4, "Wpływ": 5},
    {"Zagrożenie": "Atak DDoS", "Prawdopodobieństwo": 3, "Wpływ": 4},
    {"Zagrożenie": "Błąd ludzki", "Prawdopodobieństwo": 5, "Wpływ": 3},
    {"Zagrożenie": "Utrata zasilania", "Prawdopodobieństwo": 2, "Wpływ": 2}
]

# Dane w sesji
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(default_risks)

# ➕ Dodawanie nowego ryzyka
st.subheader("➕ Dodaj nowe zagrożenie")
with st.form("add_risk_form"):
    name = st.text_input("Opis zagrożenia")
    prob = st.slider("Prawdopodobieństwo (1-5)", 1, 5, 3)
    impact = st.slider("Wpływ (1-5)", 1, 5, 3)
    submitted = st.form_submit_button("Dodaj")

    if submitted and name.strip() != "":
        st.session_state.df.loc[len(st.session_state.df)] = {
            "Zagrożenie": name,
            "Prawdopodobieństwo": prob,
            "Wpływ": impact
        }
        st.success("Zagrożenie dodane!")

# ✏️ Edycja istniejących danych
st.subheader("✏️ Edytuj istniejące ryzyka")
edited_df = st.data_editor(
    st.session_state.df[["Zagrożenie", "Prawdopodobieństwo", "Wpływ"]],
    num_rows="dynamic",
    use_container_width=True,
    key="editor"
)

# Oblicz poziom ryzyka i klasyfikację
edited_df["Poziom ryzyka"] = edited_df["Prawdopodobieństwo"] * edited_df["Wpływ"]
edited_df["Klasyfikacja"] = edited_df["Poziom ryzyka"].apply(klasyfikuj_ryzyko)

# 📋 Filtrowanie
st.subheader("📋 Filtrowanie według poziomu ryzyka")
filtr = st.radio("Pokaż tylko:", ["Wszystkie", "Niskie", "Średnie", "Wysokie"], horizontal=True)

if filtr != "Wszystkie":
    df_filtered = edited_df[edited_df["Klasyfikacja"] == filtr]
else:
    df_filtered = edited_df

# 🎨 Kolorowanie
def koloruj(val):
    if val == "Niskie":
        return "background-color: #d4edda"
    elif val == "Średnie":
        return "background-color: #fff3cd"
    elif val == "Wysokie":
        return "background-color: #f8d7da"
    return ""

st.subheader("📊 Macierz ryzyka")
st.dataframe(df_filtered.style.applymap(koloruj, subset=["Klasyfikacja"]), use_container_width=True)

# Aktualizuj sesję
st.session_state.df = edited_df.drop(columns=["Poziom ryzyka", "Klasyfikacja"])

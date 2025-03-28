import streamlit as st
import pandas as pd

st.set_page_config(page_title="Analiza ryzyka", layout="wide")
st.title("🔐 Analiza ryzyka systemów teleinformatycznych")

# Funkcja klasyfikująca poziom ryzyka
def klasyfikuj_ryzyko(poziom):
    if poziom <= 6:
        return "Niskie"
    elif poziom <= 14:
        return "Średnie"
    else:
        return "Wysokie"

# Domyślne dane
default_risks = [
    {"Zagrożenie": "Awaria serwera", "Prawdopodobieństwo": 4, "Wpływ": 5},
    {"Zagrożenie": "Atak DDoS", "Prawdopodobieństwo": 3, "Wpływ": 4},
    {"Zagrożenie": "Błąd ludzki", "Prawdopodobieństwo": 5, "Wpływ": 3},
    {"Zagrożenie": "Utrata zasilania", "Prawdopodobieństwo": 2, "Wpływ": 2}
]

# Przechowywanie danych w sesji
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(default_risks)

# Oblicz poziom ryzyka
df = st.session_state.df.copy()
df["Poziom ryzyka"] = df["Prawdopodobieństwo"] * df["Wpływ"]
df["Klasyfikacja"] = df["Poziom ryzyka"].apply(klasyfikuj_ryzyko)

# 🎨 Kolorowanie
def koloruj_komorki(val):
    if isinstance(val, (int, float)):
        return ""
    if val == "Niskie":
        return "background-color: #d4edda"  # zielony
    elif val == "Średnie":
        return "background-color: #fff3cd"  # żółty
    elif val == "Wysokie":
        return "background-color: #f8d7da"  # czerwony
    return ""

# 📋 Filtrowanie
st.subheader("📋 Filtruj ryzyka według poziomu")
wybor = st.radio("Wybierz poziom ryzyka", ["Wszystkie", "Niskie", "Średnie", "Wysokie"], horizontal=True)

if wybor != "Wszystkie":
    filtrowany_df = df[df["Klasyfikacja"] == wybor]
else:
    filtrowany_df = df

# Wyświetl
st.dataframe(filtrowany_df.style.applymap(koloruj_komorki, subset=["Klasyfikacja"]), use_container_width=True)

# ➕ Formularz dodawania
st.subheader("➕ Dodaj nowe zagrożenie")
with st.form("add_form"):
    name = st.text_input("Opis zagrożenia")
    prob = st.slider("Prawdopodobieństwo (1-5)", 1, 5, 3)
    impact = st.slider("Wpływ (1-5)", 1, 5, 3)
    submit = st.form_submit_button("Dodaj")

    if submit and name.strip() != "":
        new_row = {"Zagrożenie": name, "Prawdopodobieństwo": prob, "Wpływ": impact}
        st.session_state.df.loc[len(st.session_state.df)] = new_row
        st.success("Zagrożenie dodane. Odśwież tabelę wybierając inny filtr lub zmieniając poziom.")


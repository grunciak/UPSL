import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="System ICT - Ocena i ryzyko", layout="wide")

st.title("📊 Zintegrowana ocena systemu ICT")

# --- Sidebar navigation ---
menu = st.sidebar.radio("Wybierz moduł:", [
    "Macierz ryzyka",
    "Bezpieczeństwo sieci",
    "Ocena wg ISO 9126"
])

# --- Macierz ryzyka ---
if menu == "Macierz ryzyka":
    st.header("🔐 Macierz ryzyka systemów teleinformatycznych")

    def klasyfikuj_ryzyko(poziom):
        if poziom <= 6:
            return "Niskie"
        elif poziom <= 14:
            return "Średnie"
        else:
            return "Wysokie"

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame([
            {"Zagrożenie": "Awaria serwera", "Prawdopodobieństwo": 4, "Wpływ": 5},
            {"Zagrożenie": "Atak DDoS", "Prawdopodobieństwo": 3, "Wpływ": 4},
            {"Zagrożenie": "Błąd ludzki", "Prawdopodobieństwo": 5, "Wpływ": 3},
        ])

    st.subheader("Dodaj nowe zagrożenie")
    with st.form("add_risk_form"):
        name = st.text_input("Opis zagrożenia")
        prob = st.slider("Prawdopodobieństwo (1-5)", 1, 5, 3)
        impact = st.slider("Wpływ (1-5)", 1, 5, 3)
        submitted = st.form_submit_button("Dodaj")

        if submitted and name.strip():
            new_row = {"Zagrożenie": name, "Prawdopodobieństwo": prob, "Wpływ": impact}
            st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
            st.success("Zagrożenie dodane.")

    edited_df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True,
        key="editor"
    )

    edited_df["Poziom ryzyka"] = edited_df["Prawdopodobieństwo"] * edited_df["Wpływ"]
    edited_df["Klasyfikacja"] = edited_df["Poziom ryzyka"].apply(klasyfikuj_ryzyko)

    filt = st.radio("Pokaż:", ["Wszystkie", "Niskie", "Średnie", "Wysokie"], horizontal=True)
    if filt != "Wszystkie":
        df_filtered = edited_df[edited_df["Klasyfikacja"] == filt]
    else:
        df_filtered = edited_df

    def koloruj(val):
        if val == "Niskie": return "background-color: #d4edda"
        elif val == "Średnie": return "background-color: #fff3cd"
        elif val == "Wysokie": return "background-color: #f8d7da"
        return ""

    st.subheader("📊 Macierz ryzyka")
    st.dataframe(df_filtered.style.applymap(koloruj, subset=["Klasyfikacja"]), use_container_width=True)
    st.session_state.df = edited_df.drop(columns=["Poziom ryzyka", "Klasyfikacja"])

# --- Moduł bezpieczeństwa sieci ---
elif menu == "Bezpieczeństwo sieci":
    st.header("🛡️ Ocena bezpieczeństwa komponentów sieci")

    komponenty = {
        "Firewall": [
            ("Błędna konfiguracja", "Audyt reguł, segmentacja sieci"),
            ("Zbyt otwarte porty", "Wdrożenie polityki najmniejszych uprawnień")
        ],
        "Serwer": [
            ("Nieaktualny system", "Automatyczne aktualizacje, backup"),
            ("Brak kontroli dostępu", "Wdrożenie RBAC/2FA")
        ],
        "Router": [
            ("Hasło domyślne", "Zmiana hasła, hardening"),
            ("Brak logów", "Centralny monitoring logów")
        ]
    }

    wybor = st.selectbox("Wybierz komponent:", list(komponenty.keys()))
    zagrozenia = komponenty[wybor]

    st.subheader("Możliwe zagrożenia")
    for z, s in zagrozenia:
        with st.expander(z):
            p = st.slider(f"Prawdopodobieństwo - {z}", 1, 5, 3, key=f"p_{z}")
            i = st.slider(f"Wpływ - {z}", 1, 5, 3, key=f"i_{z}")
            poziom = p * i
            klasyf = klasyfikuj_ryzyko(poziom)
            st.write(f"**Ryzyko:** {poziom} ({klasyf})")
            st.write(f"💡 Środek zaradczy: {s}")

# --- ISO 9126 ---
elif menu == "Ocena wg ISO 9126":
    st.header("📐 Ocena jakości systemu wg ISO/IEC 9126")

    cechy = [
        "Funkcjonalność", "Niezawodność", "Użyteczność",
        "Wydajność", "Możliwość konserwacji", "Przenośność"
    ]

    oceny = {}
    for cecha in cechy:
        oceny[cecha] = st.slider(f"{cecha}", 1, 5, 3)

    df_iso = pd.DataFrame({"Cecha": list(oceny.keys()), "Ocena": list(oceny.values())})

    st.subheader("📈 Wyniki oceny")
    st.dataframe(df_iso.set_index("Cecha"))

    if st.button("Zinterpretuj wynik"):
        srednia = np.mean(list(oceny.values()))
        st.write(f"🔍 Średnia jakość systemu: **{srednia:.2f}/5**")
        if srednia >= 4:
            st.success("System spełnia wysokie wymagania jakościowe.")
        elif srednia >= 3:
            st.warning("System ma średnią jakość — warto poprawić niektóre aspekty.")
        else:
            st.error("System nie spełnia minimalnych norm jakości wg ISO 9126.")


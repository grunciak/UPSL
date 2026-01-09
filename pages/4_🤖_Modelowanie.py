import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import (
    load_data, infer_target, get_dataset_info, make_filters, target_mode,
    train_model_cached, evaluate_model, feature_importance, cross_val_predictions
)

st.set_page_config(page_title="Modelowanie", page_icon="🤖", layout="wide")
st.title("🤖 Modelowanie: predykcja jakości wina")

df = load_data("WineQT.csv")
target_col = infer_target(df)
info = get_dataset_info(df, target_col)

st.sidebar.header("Dane i target")
filtered = make_filters(df, info.numeric_cols)

drop_dups = st.sidebar.checkbox("Usuń duplikaty", value=True)
if drop_dups:
    filtered = filtered.drop_duplicates()

y, _ = target_mode(filtered, target_col)
X = filtered.drop(columns=[target_col]).select_dtypes(include=[np.number]).copy()
feature_names = list(X.columns)

st.sidebar.header("Walidacja i model")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
seed = st.sidebar.number_input("Seed", min_value=0, max_value=10_000, value=42, step=1)

model_kind = st.sidebar.selectbox("Model", ["LogisticRegression", "RandomForest"])

params = {}
if model_kind == "LogisticRegression":
    params["C"] = st.sidebar.slider("C (siła regularyzacji)", 0.01, 10.0, 1.0)
    params["max_iter"] = st.sidebar.slider("max_iter", 200, 5000, 2000, 200)
    cw = st.sidebar.selectbox("class_weight", ["None", "balanced"], index=1 if y.nunique() > 2 else 0)
    params["class_weight"] = None if cw == "None" else "balanced"

if model_kind == "RandomForest":
    params["n_estimators"] = st.sidebar.slider("n_estimators", 100, 1200, 400, 50)
    params["max_depth"] = st.sidebar.slider("max_depth (0=brak limitu)", 0, 40, 0, 1)
    params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 20, 2, 1)
    params["min_samples_leaf"] = st.sidebar.slider("min_samples_leaf", 1, 20, 1, 1)
    cw = st.sidebar.selectbox("class_weight", ["None", "balanced"], index=1)
    params["class_weight"] = None if cw == "None" else "balanced"
    params["random_state"] = int(seed)

params_json = json.dumps(params, sort_keys=True)

run = st.button("🚀 Trenuj model", type="primary")
if not run:
    st.info("Ustaw parametry w panelu po lewej i kliknij **Trenuj model**.")
    st.stop()

with st.spinner("Trening modelu..."):
    model, (X_train, X_test, y_train, y_test) = train_model_cached(
        X, y, model_kind, params_json, float(test_size), int(seed)
    )

results = evaluate_model(model, X_test, y_test)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{results['accuracy']:.3f}")
c2.metric("F1", f"{results['f1']:.3f}")
c3.metric("Precision", f"{results['precision']:.3f}")
c4.metric("Recall", f"{results['recall']:.3f}")

st.subheader("Raport klasyfikacji")
st.code(results["report"], language="text")

st.subheader("Macierz pomyłek")
cm = results["cm"]
fig = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion matrix")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Ważność cech / wpływ cech")
imp = feature_importance(model, feature_names)
if imp is None:
    st.info("Ten model nie udostępnia ważności cech wprost.")
else:
    topk = st.slider("Pokaż top K cech", 5, min(30, len(imp)), min(15, len(imp)))
    st.dataframe(imp.head(topk), use_container_width=True)
    fig2 = px.bar(imp.head(topk)[::-1], x="importance", y="feature", orientation="h", title=f"Top {topk} cech")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Predykcja dla pojedynczej próbki (manual input)")
with st.expander("Wprowadź wartości cech", expanded=False):
    sample = {}
    cols = st.columns(3)
    for i, f in enumerate(feature_names):
        col = cols[i % 3]
        vmin = float(X[f].min())
        vmax = float(X[f].max())
        default = float(X[f].median())
        sample[f] = col.number_input(f, value=default, min_value=vmin, max_value=vmax)
    sample_df = pd.DataFrame([sample])
    pred = model.predict(sample_df)[0]
    st.success(f"Predykcja: **{pred}**")

st.subheader("Walidacja krzyżowa (opcjonalnie)")
do_cv = st.checkbox("Policz predykcje cross-val (wolniejsze)", value=False)
if do_cv:
    folds = st.slider("Liczba foldów", 3, 10, 5)
    with st.spinner("Cross-val..."):
        preds = cross_val_predictions(X, y, model_kind, params, folds=int(folds), seed=int(seed))
    avg = "binary" if y.nunique() == 2 else "weighted"
    from sklearn.metrics import accuracy_score, f1_score
    st.write({"cv_accuracy": float(accuracy_score(y, preds)), "cv_f1": float(f1_score(y, preds, average=avg))})

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import os
import time
from streamlit_shap import st_shap

st.set_page_config(page_title="Platformă ML: Clasificare & Regresie", layout="wide")

@st.cache_resource
def load_all_assets():
    hr_assets = joblib.load('models_data.pkl')
    reg_assets = joblib.load('regresie_data.pkl')
    hr_params = joblib.load('clas_hyperparameters.pkl')
    reg_params = joblib.load('reg_hyperparameters.pkl')
    return hr_assets, reg_assets, hr_params, reg_params

try:
    hr_assets, reg_assets, hr_params, reg_params = load_all_assets()
except FileNotFoundError as e:
    st.error(f"Eroare la încărcare fișiere .pkl: {e}. Asigură-te că toate fișierele sunt în folderul aplicației.")

def explain_prediction(model, input_df, task_type="classification"):
    st.subheader("🔍 Explicația Predicției Locale (SHAP)")

    st.write("---")
    st.markdown("""
    **Cum se realizează interpretarea?**
    După ce modelul oferă rezultatul, utilizăm valorile **SHAP (SHapley Additive exPlanations)** pentru a descompune predicția. 
    Sistemul analizează impactul fiecărei variabile introduse față de media setului de date.

    ⚠️ *Generarea graficului poate dura câteva secunde deoarece depinde direct de complexitatea predicției tocmai efectuate.*
    """)

    with st.spinner('Se calculează influența factorilor... Vă rugăm așteptați.'):
        try:
            if task_type == "classification":
                background = hr_assets['X_test'].iloc[:50]
                predict_fn = lambda x: model.predict_proba(x)[:, 1] if hasattr(model,
                                                                               "predict_proba") else model.predict
            else:
                background = reg_assets['X_test'].iloc[:50]
                predict_fn = model.predict

            explainer = shap.Explainer(predict_fn, background)
            shap_result = explainer(input_df)

            st_shap(shap.force_plot(shap_result.base_values[0], shap_result.values[0], input_df), height=200)

            st.success("Analiza SHAP a fost generată cu succes!")
            st.info(
                "💡 **Interpretare:** Caracteristicile cu **roșu** au împins rezultatul în sus (probabilitate mai mare/ani mai mulți), iar cele cu **albastru** au tras rezultatul în jos.")

        except Exception as e:
            st.warning(f"Graficul SHAP local nu a putut fi generat: {e}")

st.sidebar.title("Navigare Proiect")
page = st.sidebar.radio("Selectează Sarcina ML:", ["Clasificare: HR Analytics", "Regresie: Life Expectancy"])

if page == "Clasificare: HR Analytics":
    st.title("👨‍💼 Predicția Fluctuației Resurselor Umane")

    with st.expander("📖 Detalii Obiectiv și Metodologie"):
        st.write("""
        **Obiectiv:** Identificarea candidaților care intenționează să părăsească compania.
        **Cum funcționează?** Modelul analizează profilul candidatului și returnează o probabilitate.
        Dacă probabilitatea depășește pragul de decizie, candidatul este marcat cu **1 (Va pleca)**.
        """)

    st.subheader("1. Analiză Exploratorie (EDA)")
    c1, c2 = st.columns(2)
    with c1:
        st.image("eda_target.png", caption="Distribuția Target")
    with c2:
        st.image("eda_pairplot.png", caption="Corelații Variabile")

    st.divider()
    st.subheader("2. Evaluare Model Selectat")
    model_name = st.selectbox("Selectează Modelul:", list(hr_assets['best_estimators'].keys()))
    model = hr_assets['best_estimators'][model_name]
    fname = model_name.replace(' ', '_')

    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        st.write("**Metrici Performanță:**")
        st.dataframe(hr_assets['final_ranking'][hr_assets['final_ranking']['Model'].str.contains(model_name)])
        st.write("**⚙️ Hiperparametri Optimi:**")
        with st.expander("Vezi detalii"): st.json(hr_params[model_name])

    with col_m2:
        st.write("**📊 Vizualizări Performanță Model:**")
        tab1, tab2, tab3 = st.tabs(["Curba Învățare", "Matrice Confuzie", "SHAP Global"])
        with tab1:
            if os.path.exists(f"lc_{fname}.png"):
                st.image(f"lc_{fname}.png")
            else:
                st.warning("Imagine lipsă.")
        with tab2:
            if os.path.exists(f"cm_{fname}.png"):
                st.image(f"cm_{fname}.png")
            else:
                st.warning("Imagine lipsă.")
        with tab3:
            if os.path.exists(f"shap_summary_{fname}.png"):
                st.image(f"shap_summary_{fname}.png")
            else:
                st.info("SHAP Global nu a fost găsit.")

    st.write("---")
    st.subheader("🔮 Simulator Predicție")
    input_data = {}
    cols = st.columns(3)
    for i, feature in enumerate(hr_assets['feature_names']):
        with cols[i % 3]:
            input_data[feature] = st.number_input(f"{feature}", value=float(hr_assets['X_test'][feature].mean()),
                                                  key=f"hr_{feature}")

    if st.button("Execută Predicție HR"):
        with st.status("Se procesează datele...", expanded=True) as status:
            st.write("Analizăm profilul introdus...")
            input_df = pd.DataFrame([input_data])
            pred = model.predict(input_df)[0]
            time.sleep(0.5)
            status.update(label="Predicție finalizată!", state="complete", expanded=False)

        label = "Va pleca (1)" if pred == 1 else "Va rămâne (0)"
        st.success(f"Rezultat Predicție: **{label}**")

        explain_prediction(model, input_df, task_type="classification")

elif page == "Regresie: Life Expectancy":
    st.title("🌍 Estimarea Speranței de Viață (OMS)")

    with st.expander("📖 Detalii Obiectiv și Metodologie"):
        st.write("""
        **Obiectiv:** Predicția speranței de viață în ani.
        **Cum funcționează?** Modelul primește indicatori de sănătate și economici (ex: PIB, mortalitate) 
        și calculează o valoare numerică continuă reprezentând anii de viață estimați.
        """)

    st.subheader("1. Analiză Exploratorie (EDA)")
    cr1, cr2 = st.columns(2)
    with cr1:
        st.image("eda_reg_dist.png", caption="Distribuție Target")
    with cr2:
        st.image("eda_reg_corr.png", caption="Matrice Corelație")

    st.divider()
    st.subheader("2. Evaluare Model Selectat")
    reg_model_name = st.selectbox("Selectează Modelul de Regresie:", list(reg_assets['best_estimators'].keys()))
    reg_model = reg_assets['best_estimators'][reg_model_name]
    fname_reg = reg_model_name.replace(' ', '_')

    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        st.write("**Metrici Evaluare:**")
        st.dataframe(reg_assets['tuned_df'][reg_assets['tuned_df']['Model'] == reg_model_name])
        st.write("**⚙️ Hiperparametri Optimi:**")
        with st.expander("Vezi detalii"): st.json(reg_params[reg_model_name])

    with col_r2:
        st.write("**📊 Vizualizări Performanță Model:**")
        tab_r1, tab_r2, tab_r3 = st.tabs(["Curba Învățare", "Eroare (Real vs Prezis)", "SHAP Global"])
        with tab_r1:
            if os.path.exists(f"lc_{fname_reg}.png"):
                st.image(f"lc_{fname_reg}.png")
            else:
                st.warning("Imagine lipsă.")
        with tab_r2:
            if os.path.exists(f"err_{fname_reg}.png"):
                st.image(f"err_{fname_reg}.png")
            else:
                st.warning("Imagine lipsă.")
        with tab_r3:
            if os.path.exists(f"shap_summary_{fname_reg}.png"):
                st.image(f"shap_summary_{fname_reg}.png")
            else:
                st.info("SHAP Global nu a fost găsit.")

    st.write("---")
    st.subheader("🔮 Simulator Predicție")
    reg_input_data = {}
    reg_cols = st.columns(3)
    for i, feature in enumerate(reg_assets['feature_names']):
        with reg_cols[i % 3]:
            reg_input_data[feature] = st.number_input(f"{feature}", value=float(reg_assets['X_test'][feature].mean()),
                                                      key=f"reg_{feature}")

    if st.button("Execută Predicție Regresie"):
        with st.status("Se calculează speranța de viață...", expanded=True) as status:
            reg_input_df = pd.DataFrame([reg_input_data])
            prediction = reg_model.predict(reg_input_df)[0]
            time.sleep(0.5)
            status.update(label="Calcul finalizat!", state="complete", expanded=False)

        st.info(f"Speranța de viață estimată: **{prediction:.2f} ani**")

        explain_prediction(reg_model, reg_input_df, task_type="regression")
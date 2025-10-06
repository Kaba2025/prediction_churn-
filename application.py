import os
# CONFIGURATION CRITIQUE - TOUJOURS AU DÉBUT
os.environ['STREAMLIT_GATHER_USAGE_STATS'] = 'false'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# =========================
# Chargement du modèle
# =========================
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("botswana_bank_customer_churn_light.csv")

pipe_best = load_model()
df = load_data()

# =========================
# Fonctions utilitaires
# =========================
def appliquer_regles(row):
    risque = "Risque faible"
    if (row["Balance"] < df["Balance"].quantile(0.33)) and (row["NumComplaints"] > df["NumComplaints"].quantile(0.66)):
        risque = "Risque élevé"
    return risque

def decision_finale(pred, regle):
    if pred == "Churn" and regle == "Risque élevé":
        return "🔴 Churn confirmé"
    elif pred == "Churn":
        return "🟠 Churn possible"
    else:
        return "🟢 Non Churn"

def recommandations(pred, risque):
    recos = []
    if pred == "Churn" or risque == "Risque élevé":
        recos.append("📞 Contacter rapidement le client pour comprendre ses besoins.")
        recos.append("💳 Proposer une offre promotionnelle ou un produit adapté.")
        recos.append("🤝 Améliorer le service client pour restaurer la confiance.")
    else:
        recos.append("✅ Continuer à fidéliser ce client avec des avantages.")
        recos.append("⭐ Demander un feedback pour améliorer l'expérience.")
    return recos

# =========================
# Mise en page Streamlit
# =========================
st.set_page_config(page_title="Application Churn", layout="wide", page_icon="📊")

st.markdown("<h1 style='text-align:center; color:#2E86C1;'>📊 Application de Prédiction du Churn</h1>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.header("⚙️ Paramètres client")
st.sidebar.info("🖊️ Saisissez les informations du client à analyser.")

# =========================
# Champs d'entrée
# =========================
age = st.sidebar.slider("Âge", 18, 90, 35)
dependents = st.sidebar.slider("Nombre de personnes à charge", 0, 10, 2)
income = st.sidebar.number_input("Revenu annuel", 100, 100000, 15000)
credit_score = st.sidebar.slider("Credit Score", 300, 900, 600)
history = st.sidebar.slider("Durée historique de crédit (années)", 0, 50, 5)
loans = st.sidebar.number_input("Montant des prêts en cours", 0, 100000, 2000)
balance = st.sidebar.number_input("Solde du compte", 0, 200000, 10000)
products = st.sidebar.slider("Nombre de produits bancaires", 1, 10, 2)
complaints = st.sidebar.slider("Nombre de réclamations", 0, 20, 1)

gender = st.sidebar.selectbox("Genre", ["Male", "Female"])
marital = st.sidebar.selectbox("État civil", ["Single", "Married", "Divorced"])
education = st.sidebar.selectbox("Niveau d'éducation", ["High School", "Bachelor", "Master", "PhD"])
segment = st.sidebar.selectbox("Segment client", ["Standard", "Premium", "VIP"])
channel = st.sidebar.selectbox("Canal préféré", ["Email", "SMS", "Phone", "In-person"])

client_df = pd.DataFrame([{
    "Age": age,
    "Number of Dependents": dependents,
    "Income": income,
    "Credit Score": credit_score,
    "Credit History Length": history,
    "Outstanding Loans": loans,
    "Balance": balance,
    "NumOfProducts": products,
    "NumComplaints": complaints,
    "Gender": gender,
    "Marital Status": marital,
    "Education Level": education,
    "Customer Segment": segment,
    "Preferred Communication Channel": channel
}])

# =========================
# Résultats
# =========================
st.subheader("📋 Profil du client")
st.dataframe(client_df, use_container_width=True)

if st.button("🔮 Prédire le Churn"):
    with st.spinner("⏳ Analyse en cours..."):
        time.sleep(1.5)

    proba = pipe_best.predict_proba(client_df)[0,1]
    pred = "Churn" if proba >= 0.5 else "Non Churn"
    risque = appliquer_regles(client_df.iloc[0])
    decision = decision_finale(pred, risque)

    # Affichage métriques jolies
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 Probabilité prédite", f"{proba*100:.2f}%")
    with col2:
        st.metric("✅ Décision finale", decision)

    # Barre de progression
    st.progress(int(proba*100))

    # Interprétation
    with st.expander("🔍 Interprétation par règles d'association"):
        st.write(f"Risque identifié : **{risque}**")
        if risque == "Risque élevé":
            st.warning("⚠️ Le client cumule un solde faible et beaucoup de réclamations.")
        else:
            st.success("✅ Aucun signe majeur de risque.")

    # Recommandations
    st.subheader("💡 Recommandations & Solutions")
    for reco in recommandations(pred, risque):
        st.write(f"- {reco}")

    # =========================
    # Tableau comparatif
    # =========================
    st.subheader("📊 Comparatif client vs population")

    # Variables quantitatives à comparer
    variables = ["Income", "Credit Score", "Balance", "NumOfProducts", "NumComplaints"]

    moyenne_pop = df[variables].mean().round(2)
    client_vals = client_df[variables].iloc[0]

    comparatif = pd.DataFrame({
        "Variable": variables,
        "Client": client_vals.values,
        "Moyenne population": moyenne_pop.values
    })

    st.dataframe(comparatif, use_container_width=True)
    st.bar_chart(comparatif.set_index("Variable"))

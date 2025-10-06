import os
os.environ['STREAMLIT_GATHER_USAGE_STATS'] = 'false'

import streamlit as st

st.title("🎉 TEST - Application Churn")
st.write("Si tu vois ce message, Streamlit Cloud marche !")

# Test basique sans charger les fichiers d'abord
age = st.slider("Âge test", 18, 90, 35)
st.write(f"Âge sélectionné : {age}")

st.success("✅ L'application fonctionne !")
st.balloons()

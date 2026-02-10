# Pour lancer cette application :
# se placer dans ce dossier `streamlit` puis lancer la commande suivante.
# streamlit run app.py

import streamlit as st

st.set_page_config(
    page_title="Rakuten - Catégorisation Multi-modale",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import de toutes les fonctions show_*
# Une fonction show_* par fichier de préférence, pour faciliter la collaboration
from parts.problematique import show_problematique
from parts.exploration import show_exploration
from parts.preprocessing import show_preprocessing
from parts.models import show_models
from parts.demo import show_demo
from parts.analyse_meilleur_modele import show_analyse_meilleur_modele
from parts.conclusion import show_conclusion

# Configuration de la navigation : partie → fonction montrant la partie
PAGES = {
    "🎯 Problématique": show_problematique,
    "📊 Analyse exploratoire": show_exploration,
    "⚙️ Preprocessing": show_preprocessing,
    "🤖 Modèles & résultats": show_models,
    "🚀 Démo interactive": show_demo,
    "📈 Analyse du meilleur modèle": show_analyse_meilleur_modele,
    "🎓 Conclusion & perspectives": show_conclusion
}

# Navigation
page = st.sidebar.selectbox("Navigation", list(PAGES.keys()))
PAGES[page]()

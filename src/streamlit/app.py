# Pour lancer cette application :
# se placer dans le dossier du repo git puis lancer la commande suivante.
# streamlit run src/streamlit/app.py

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
from parts.analyse_meilleur_modele import show_analyse_meilleur_modele
from parts.conclusion import show_conclusion
from parts.ML_FR_EN import show_ML_FR_EN
from parts.Deep_multimodal import show_Deep_multimodal

# Handle the possibility that loading demo fails because of missing dependencies (tensorflow, ...)
demo_available = True
try:
    from parts.demo import show_demo
except ImportError as e:
    demo_available = False
    def show_demo():
        st.title("🚀 Démo interactive")
        st.header("🚧 Erreur")

# Configuration de la navigation : partie → fonction montrant la partie
PAGES = {
    "🎯 Problématique": show_problematique,
    "📊 Analyse exploratoire": show_exploration,
    "⚙️ Preprocessing": show_preprocessing,
    "🤖 Modèles & résultats": show_models,
    "🚀 Démo interactive": show_demo,
    "📈 Analyse du meilleur modèle": show_analyse_meilleur_modele,
    "🎓 Conclusion & perspectives": show_conclusion,
    "Machine_Learning_en_français_et_anglais": show_ML_FR_EN,
    "Deep_multimodal": show_Deep_multimodal,
}

st.sidebar.title("Projet Rakuten")

# Navigation
page = st.sidebar.radio("Navigation", list(PAGES.keys()))
PAGES[page]()

# Pour lancer cette application :
# se placer dans le dossier du repo git, puis lancer les commandes suivantes.
# source venv/bin/activate
# streamlit run src/streamlit/app.py

import streamlit as st
import sys
from pathlib import Path

# Quick fix for import issues
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

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

# Handle the possibility that loading demo fails because of missing dependencies (tensorflow, ...)
demo_available = True
try:
    from parts.demo import show_demo
except ImportError as e:
    demo_available = False
    exc = e  # strangely, `e` is not accessible from `show_demo` but `exc` is
    def show_demo():
        st.error(f"🚧 Démo non disponible :\n\n{exc}")

# Configuration de la navigation : partie → fonction montrant la partie
PAGES = {
    "🎯 Problématique": show_problematique,
    "📊 Analyse exploratoire": show_exploration,
    "⚙️ Preprocessing": show_preprocessing,
    "🤖 Modèles & résultats": show_models,
    "🚀 Démo interactive": show_demo,
    "📈 Analyse du meilleur modèle": show_analyse_meilleur_modele,
    "🎓 Conclusion & perspectives": show_conclusion,
}

st.sidebar.title("Projet Rakuten")

# Navigation
page = st.sidebar.radio("Navigation", list(PAGES.keys()))
PAGES[page]()

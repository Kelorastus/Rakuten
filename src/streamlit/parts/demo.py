# pour un élément du dataset, on obtient sa catégorie prédite par le meilleur modèle pré-entraîné

"""
idées

générer un échantillon aléatoire du dataset
    ? quand l'utilisateur clique sur un bouton
le montrer
    ? st.dataframe
        ? st.column_config.ImageColumn pour affichage des images
    ? st.data_editor pour permettre à l'utilisateur de modifier l'échantillon
demander à l'utilisateur de choisir un indice dans l'échantillon
prédire le produit à cet indice (ou pour tout l'échantillon ?)
    ? quand l'utilisateur clique sur un bouton
? grad-cam
"""

import streamlit as st
# import torch
# from PIL import Image
# import numpy as np
# # vos imports pour le modèle et grad-cam


# @st.cache_resource
# def load_model():
#     """Charge le modèle une seule fois"""
#     model = torch.load("models/best_model.pth")
#     model.eval()
#     return model


def show_demo():
    st.title("🚀 Démo Interactive")
    st.info("🚧 Section en cours de développement")

#     model = load_model()

#     # Sélection d'un échantillon
#     sample_choice = st.selectbox(
#         "Choisir un produit du dataset",
#         ["Exemple 1", "Exemple 2", "Exemple 3", "Upload personnalisé"]
#     )

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("Entrées")
#         image = st.file_uploader("Image du produit", type=["jpg", "png"])
#         texte = st.text_area("Description", "Description du produit...")

#         if st.button("🔮 Prédire"):
#             # Votre code de prédiction
#             prediction = faire_prediction(model, image, texte)
#             st.session_state['prediction'] = prediction

#     with col2:
#         st.subheader("Résultats")
#         if 'prediction' in st.session_state:
#             pred = st.session_state['prediction']
#             st.success(f"Catégorie prédite : **{pred['categorie']}**")
#             st.metric("Confiance", f"{pred['confiance']:.2%}")

#             # Top 3 prédictions
#             st.write("Top 3 catégories :")
#             for i, (cat, prob) in enumerate(pred['top3']):
#                 st.progress(prob, text=f"{i+1}. {cat} ({prob:.1%})")


# def show_grad_cam():
#     st.title("📈 Analyse avec Grad-CAM")

#     st.markdown("""
#     Grad-CAM permet de visualiser les zones de l'image
#     qui ont influencé la décision du modèle.
#     """)

#     if 'prediction' in st.session_state:
#         # Grad-CAM simplifié
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image("image_originale.jpg", caption="Image originale")
#         with col2:
#             st.image("gradcam_overlay.jpg", caption="Grad-CAM")
#     else:
#         st.info("Faites d'abord une prédiction dans la démo")


# def show_conclusion():
#     st.title("🎓 Conclusion & Perspectives")

#     st.header("Résultats obtenus")
#     st.image("screenshots/resultats_finaux.png")

#     st.header("Lien avec la problématique métier")
#     st.markdown("""
#     - Amélioration de X% de la catégorisation
#     - Réduction du temps de traitement
#     - Impact sur l'expérience utilisateur
#     """)

#     st.header("Critiques & Perspectives")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("⚠️ Limites")
#         st.markdown("- Temps de calcul\n- Déséquilibre des classes")
#     with col2:
#         st.subheader("🚀 Améliorations")
#         st.markdown("- Augmentation de données\n- Ensemble de modèles")

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
from pathlib import Path
from src.preprocessing.core import load_reproducible_split
from src.preprocessing.pipelines.deep_learning import load_preprocessors
from src.preprocessing.pipelines.deep_learning_on_text_and_images import preprocess_features
from tensorflow import keras


@st.cache_resource
def preprocessing_DL3(row_index=0):
    #TODO: load only the picked row_index
    # parameters
    multimodal_artifacts_folder = Path(f'artifacts/on_text_and_images/deep_learning/v1')
    preprocessors_folder = multimodal_artifacts_folder
    rebalance_with_weights = True  # True is useful if some classes are ignored by the model. In that case, small_train_sample should be False.
    augment=False  # If True, augment data for training
    BATCH_SIZE = 32
    RANDOM_SEED = 42

    preprocessors = load_preprocessors(names=['target','tabular','hash','text_vectorizer'], artifacts_folder=preprocessors_folder)

    train_ds, new_preprocessors, class_weights, train_inputs_dict, y_train_ohe = preprocess_features(
        X_train, y_train, preprocessors, full_X_train=full_X_train, full_y_train=full_y_train, shuffle=True,
        BATCH_SIZE = BATCH_SIZE, rebalance_with_weights=rebalance_with_weights, augment=augment
    )
    preprocessors |= new_preprocessors
    if new_preprocessors:
        print(f"warning: preprocessors got fitted again, so they were probably not loaded properly. {new_preprocessors=}")

    test_ds, new_preprocessors, class_weights_test, test_inputs_dict, y_test_ohe = preprocess_features(X_test, y_test, preprocessors, shuffle=False, BATCH_SIZE = BATCH_SIZE, rebalance_with_weights=False, augment=False)

    if new_preprocessors:
        print(f"error: some preprocessors got fitted on the testing set, so they were probably not handled properly. {new_preprocessors=}")

    return test_ds


@st.cache_resource
def load_model_DL3():
    path = "artifacts/on_text_and_images/deep_learning/v1/best_model_arch-11_epoch_index-01_val_accuracy-0.8338_f1-0.8337.keras"
    model = keras.models.load_model(path)
    return model


@st.cache_data
def load_dataset():
    X_train, X_test, y_train, y_test = load_reproducible_split(folder = 'Dataset2')
    return X_train, X_test, y_train, y_test


def predict():
    pass


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


# Si exécuté directement (pour tester)
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show_demo()

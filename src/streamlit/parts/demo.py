# pour un élément du dataset, on obtient sa catégorie prédite par le meilleur modèle pré-entraîné

"""
idées

générer un échantillon aléatoire du dataset
    ? quand l'utilisateur clique sur un bouton
    ? st.data_editor pour permettre à l'utilisateur de modifier l'échantillon
demander à l'utilisateur de choisir un indice dans l'échantillon
prédire le produit à cet indice (ou pour tout l'échantillon ?)
    ? quand l'utilisateur clique sur un bouton
? grad-cam
"""

from tensorflow import keras
import streamlit as st
import base64
from pathlib import Path
from functools import partial
from src.preprocessing.image import get_image_path
from src.models.on_text_and_images.deep_learning import CATEGORY_MAPPING
from src.preprocessing.core import load_reproducible_split
from src.preprocessing.pipelines.deep_learning import load_preprocessors
from src.preprocessing.pipelines.deep_learning_on_text_and_images import preprocess_features


@st.cache_data
def load_Dataset2():
    X_train, X_test, y_train, y_test = load_reproducible_split(folder = 'Dataset2')
    return X_train, X_test, y_train, y_test


@st.cache_resource
def preprocessing_DL3(X_test, y_test):
    # parameters
    multimodal_artifacts_folder = Path(f'artifacts/on_text_and_images/deep_learning/v1')
    preprocessors_folder = multimodal_artifacts_folder
    BATCH_SIZE = 32

    preprocessors = load_preprocessors(names=['target','tabular','hash','text_vectorizer'], artifacts_folder=preprocessors_folder)

    test_ds, new_preprocessors, class_weights_test, test_inputs_dict, y_test_ohe = preprocess_features(X_test, y_test, preprocessors, shuffle=False, BATCH_SIZE = BATCH_SIZE, rebalance_with_weights=False, augment=False)

    if new_preprocessors:
        print(f"error: some preprocessors got fitted on the testing set, so they were probably not handled properly. {new_preprocessors=}")

    return test_ds, preprocessors


@st.cache_resource
def load_model_DL3():
    path = "artifacts/on_text_and_images/deep_learning/v1/best_model_arch-11_epoch_index-01_val_accuracy-0.8338_f1-0.8337.keras"
    model = keras.models.load_model(path)
    return model


def np_array_to_int(np_array):
    return int(np_array[0])


def predict_DL3(model, test_ds, preprocessors):
    y_pred = model.predict(test_ds)
    y_pred_class = np_array_to_int(preprocessors['target'].inverse_transform(y_pred.argmax(axis=1)))
    return y_pred_class


@st.cache_data
def get_class_description(class_code: int):
    description = CATEGORY_MAPPING.get(class_code, "inconnue")
    return description


def show_image_from_row(row):
    # e.g. row = X_test.iloc[0]
    image_path = get_image_path(row, folder = 'Dataset/images/image_train')
    if image_path.exists():
        st.image(image_path)
    else:
        st.text(f"Error: file not found: {image_path=}")
        st.text(f"cwd: {Path('.').resolve()}")
        st.text(f"folder: {image_path.parent.resolve()}")


def image_path_to_base64(path: str):
    with open(path, "rb") as p:
        file = p.read()
        return f"data:image/png;base64,{base64.b64encode(file).decode()}"


@st.cache_data
def get_df_with_images(initial_df):
    df = initial_df.copy()
    image_getter = partial(get_image_path, folder = 'Dataset/images/image_train')
    image_paths = df.apply(image_getter, axis=1)
    df.insert(0, 'image', image_paths)
    df["image"] = df['image'].apply(image_path_to_base64)
    return df


def show_demo(small_image_size = 200):
    st.title("🚀 Démo interactive")
    X_train, X_test, y_train, y_test = load_Dataset2()

    # Pick a sample from X_test (because images would use too many resources for the whole X_test)
    if 'sample' not in st.session_state:
        sample_size = 10
        sample = X_test.sample(sample_size)
        st.session_state['sample'] = get_df_with_images(sample)
        #TODO: allow refreshing sample

    # Product selection
    st.text("Veuillez cocher un produit à catégoriser par le modèle.\nPour consulter les détails des produits, faire défiler le tableau horizontalement/verticalement.")
    #TODO? slider for small_image_size
    event = st.dataframe(st.session_state['sample'],
                 column_config={'image': st.column_config.ImageColumn(width=small_image_size)},
                 row_height=small_image_size,
                 on_select="rerun",
                 selection_mode="single-row")

    # If user has selected a product
    if event.selection.rows:
        # Get user selection
        input_index = event.selection.rows[0]

        # Get proper index (convert index from `session_state['sample'].iloc` to `X_test.loc`)
        row_index = int(st.session_state['sample'].iloc[input_index].name)

        X_test_row = X_test.loc[[row_index]]  # Double-bracket: to keep it as a dataframe instead of a series
        y_test_row = y_test.loc[[row_index]]  # Double-bracket: to keep it as a series
        # st.write("Produit sélectionné :", st.session_state['sample'].iloc[input_index])
        # st.write(row_index,X_test_row,f"{y_test_row.iloc[0]=} {type(y_test_row)=}")

        # Prediction
        test_ds, preprocessors = preprocessing_DL3(X_test_row, y_test_row)
        model = load_model_DL3()
        y_pred_class = predict_DL3(model, test_ds, preprocessors)
        y_test_class = y_test_row.iloc[0]
        y_pred_description = get_class_description(y_pred_class)
        y_test_description = get_class_description(y_test_class)

        if y_pred_class == y_test_class:
            prediction_style = "green"
        else:
            prediction_style = "red"

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### :green[catégorie réelle]\n:green[{y_test_class} - {y_test_description}.]")
        with col2:
            st.markdown(f"### :{prediction_style}[catégorie prédite]\n:{prediction_style}[{y_pred_class} - {y_pred_description}.]")

    # col1, col2 = st.columns(2)
    # with col1:
    #         image = st.file_uploader("Image du produit", type=["jpg", "png"])
    #         texte = st.text_area("Description", "Description du produit...")

    #         if st.button("🔮 Prédire"):
    #             # Votre code de prédiction
    #             prediction = faire_prediction(model, image, texte)
    #             st.session_state['prediction'] = prediction
    # with col2:
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

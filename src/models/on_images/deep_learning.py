from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.models import Model
import re


def define_model(pHash_vocab_size, md5_vocab_size, n_cols_tabular=24, num_classes = 27):
    '''
    Args:
        pHash_vocab_size
        md5_vocab_size
        n_cols_tabular: Nombre de features numériques.
        num_classes: Nombre de classes de la variable cible.
    '''

    # Inputs

    image_input = keras.Input(shape=(500, 500, 3), name="image_input")
    tabular_input = keras.Input(shape=(n_cols_tabular,), name='tabular_input')

    pHash_input = keras.Input(shape=(1,), name='pHash_input', dtype='int64')  # Embedding a besoin du type int
    md5_input = keras.Input(shape=(1,), name='md5_input', dtype='int64')

    # Image branch

    # On utilise un modèle pré-entraîné.
    # C'est une base très puissante pour traiter les images.
    base_model = keras.applications.EfficientNetV2B0(
        include_top=False, # On ne garde que les couches d'extraction de features
        weights='imagenet', # Poids appris sur des millions d'images
        input_tensor=image_input
    )
    base_model.trainable = False # On "gèle" le modèle de base pour le début de l'entraînement

    # On ajoute nos propres couches par-dessus
    image_features = layers.GlobalAveragePooling2D(name='image_pooling')(base_model.output)

    image_features = layers.Dense(
        128, activation='relu', name='image_dense',
        kernel_regularizer=regularizers.l2(0.001), # Ajoute une pénalité L2
    )(image_features)

    # Tabular branch

    tabular_features = layers.Dense(
        64, activation='relu', name='tabular_dense_1',
        kernel_regularizer=regularizers.l2(0.001), # Ajoute une pénalité L2
    )(tabular_input)

    tabular_features = layers.Dense(
        32, activation='relu', name='tabular_dense_2',
        kernel_regularizer=regularizers.l2(0.001), # Ajoute une pénalité L2
    )(tabular_features)

    # Hash branches

    # Chaque hash passe par sa propre couche d'Embedding.

    pHash_features = layers.Embedding(input_dim=pHash_vocab_size, output_dim=16, name='pHash_embedding')(pHash_input)
    pHash_features = layers.Flatten(name='pHash_flatten')(pHash_features)  # retire une dimension superflue de taille 1

    md5_features = layers.Embedding(input_dim=md5_vocab_size, output_dim=16, name='md5_embedding')(md5_input)
    md5_features = layers.Flatten(name='md5_flatten')(md5_features)

    # Fusing branches

    # On fusionne toutes les features apprises en un seul grand vecteur
    all_features = layers.concatenate([
        image_features,
        tabular_features,
        pHash_features,
        md5_features
    ])
    # x = layers.Dropout(0.2)(all_features)

    # Classification

    # Quelques couches denses pour apprendre les interactions entre les différentes modalités
    x = layers.Dense(
        256, activation='relu', name='final_dense_1',
        kernel_regularizer=regularizers.l2(0.001), # Ajoute une pénalité L2
    )(all_features)

    x = layers.Dropout(0.5)(x)

    output = layers.Dense(
        num_classes, activation='softmax', name='output',
        kernel_regularizer=regularizers.l2(0.001), # Ajoute une pénalité L2
    )(x)

    # Model

    model = keras.Model(
        inputs=[image_input, tabular_input, pHash_input, md5_input],
        outputs=output
    )

    return model, base_model


def grad_cam(image, model, layer_name):
    # Récupérer la couche convolutive
    layer = model.get_layer(layer_name)

    # Créer un modèle qui génère les sorties de la couche convolutive et les prédictions
    grad_model = Model(inputs=model.input, outputs=[layer.output, model.output])

    # Ajout d'une dimension de batch
    image = tf.expand_dims(image, axis=0)

    # Calcul des gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        predicted_class = tf.argmax(predictions[0])  # Classe prédite
        loss = predictions[:, predicted_class]  # Perte pour la classe prédite

    # Gradients des scores par rapport aux sorties de la couche convolutive
    grads = tape.gradient(loss, conv_outputs)

    # Moyenne pondérée des gradients pour chaque canal
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Pondération des activations par les gradients calculés
    conv_outputs = conv_outputs[0]  # Supprimer la dimension batch
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalisation de la carte de chaleur
    heatmap = tf.maximum(heatmap, 0)  # Se concentrer uniquement sur les valeurs positives
    heatmap /= tf.math.reduce_max(heatmap)  # Normaliser entre 0 et 1
    heatmap = heatmap.numpy()  # Convertir en tableau numpy pour la visualisation

   # Redimensionner la carte de chaleur pour correspondre à la taille de l'image d'origine
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (image.shape[1], image.shape[2])).numpy()
    heatmap_resized = np.squeeze(heatmap_resized, axis=-1) # supprimer la dimension de taille 1 à la fin du tableau heatmap_resized

    # Colorier la carte de chaleur avec une palette (par exemple, "jet")
    heatmap_colored = plt.cm.jet(heatmap_resized)[..., :3] # Récupérer les canaux R, G, B

    superimposed_image = heatmap_colored * 0.7 + image[0].numpy() / 255.0

    return np.clip(superimposed_image, 0, 1), predicted_class


def show_grad_cam_cnn(images, model, conv_layers):
    number_of_images = images.shape[0]

    plt.figure(figsize=(16, 4 * len(conv_layers)))

    # row 0 of subplots (Original Images) ---
    for i in range(number_of_images):
        plt.subplot(len(conv_layers) + 1, number_of_images, i + 1)

        # Get image and convert to displayable format (assuming 0-255 range based on your previous code)
        img_display = images[i].numpy().astype("uint8")

        plt.imshow(img_display)
        plt.title("Original")
        plt.axis("off")

    # rows of grad-cam subplots
    for j, layer_name in enumerate(conv_layers):

        for i in range(number_of_images):

            subplot_index = i + 1 + (j + 1) * number_of_images
            plt.subplot(len(conv_layers) + 1, number_of_images, subplot_index)

            # Obtenir l'image avec la carte de chaleur superposée
            grad_cam_image, predicted_class = grad_cam(images[i], model, layer_name)

            plt.imshow(grad_cam_image)
            plt.title(f'{layer_name}\nPred: {predicted_class}')
            plt.axis("off")

    plt.show()


def get_efficientnet_selection(all_layers):
    """Select most relevant layer names for grad-cam."""
    selected = {}

    # 1. Always include the final convolution
    if 'top_conv' in all_layers:
        selected['top'] = 'top_conv'

    # 2. Iterate to find the last "project_conv" of each Major Block
    for layer_name in all_layers:
        # We look for patterns like "block2b_..." or "block6h_..."
        # We only care about the number (2, 6, etc.)
        match = re.search(r'block(\d+)[a-z]_project_conv', layer_name)

        if match:
            block_num = match.group(1) # e.g., "2" or "6"

            # Because the list is sorted, the loop will encounter
            # block6a, then 6b, ... then 6h.
            # We simply keep overwriting 'block6', so the last one wins.
            selected[f'block{block_num}'] = layer_name

    # 3. Sort them to maintain network order (Block 1 -> Block 6 -> Top)
    sorted_layers = sorted(selected.items(), key=lambda x: x[0])
    return [name for key, name in sorted_layers]

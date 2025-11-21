from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.models import Model


def define_model(text_vectorizer, pHash_vocab_size, md5_vocab_size, n_cols_tabular=24, num_classes = 27):  #TODO: was copied from DL_on_img
    '''
    Args:
        text_vectorizer: Une couche TextVectorization déjà adaptée.
        pHash_vocab_size
        md5_vocab_size
        n_cols_tabular: Nombre de features numériques.
        num_classes: Nombre de classes de la variable cible.
    '''

    # Inputs

    text_input = keras.Input(shape=(1,), dtype=tf.string, name='text_input')

    image_input = keras.Input(shape=(500, 500, 3), name="image_input")
    tabular_input = keras.Input(shape=(n_cols_tabular,), name='tabular_input')

    pHash_input = keras.Input(shape=(1,), name='pHash_input', dtype='int64')  # Embedding a besoin du type int
    md5_input = keras.Input(shape=(1,), name='md5_input', dtype='int64')

    # Text branch

    vectorized_text = text_vectorizer(text_input)
    text_embedding = layers.Embedding(
        input_dim=len(text_vectorizer.get_vocabulary()),
        output_dim=128,
        name='text_embedding'
    )(vectorized_text)
    x = layers.Conv1D(128, 5, activation='relu')(text_embedding)
    x = layers.GlobalMaxPooling1D()(x) # Prend l'information la plus importante de la séquence
    text_features = layers.Dense(64, activation='relu')(x)

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
        md5_features,
        text_features
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
        inputs=[image_input, tabular_input, pHash_input, md5_input, text_input],
        outputs=output
    )

    return model, base_model


def get_custom_hyperparams(model, base_model=None):
    """
    Inspecte un modèle Keras et extrait les hyperparamètres
    des couches Dense et Embedding personnalisées (entraînables).

    Args:
        base_model: Useful for transfer learning if you want to omit the layers of the pretrained model.
    """
    hyperparams = {
        'dense_layers_sizes': {},
        'embedding_dims': {}
    }

    for layer in model.layers:
        # On ne s'intéresse qu'aux couches que nous entraînons
        if base_model is None or layer not in base_model.layers:
            # Si c'est une couche Dense
            if isinstance(layer, Dense):
                # On ignore la couche de sortie finale (softmax)
                if layer.activation.__name__ != 'softmax':
                    hyperparams['dense_layers_sizes'][layer.name] = layer.units

            # Si c'est une couche Embedding
            elif isinstance(layer, Embedding):
                hyperparams['embedding_dims'][layer.name] = layer.output_dim

    return hyperparams


def flatten_params_for_logging(params_dict):
    """
    Transforme les dictionnaires et listes en chaînes de caractères
    pour un stockage facile dans un DataFrame/Parquet.
    """
    flat_params = params_dict.copy()

    # Aplatir les tailles des couches denses
    if 'dense_layers_sizes' in flat_params:
        # Trie par nom de couche pour la cohérence
        sizes = dict(sorted(flat_params['dense_layers_sizes'].items()))
        flat_params['dense_layers_sizes'] = '_'.join(map(str, sizes.values())) # ex: '128_32_64_256'

    # Aplatir les dimensions des embeddings
    if 'embedding_dims' in flat_params:
        dims = dict(sorted(flat_params['embedding_dims'].items()))
        flat_params['embedding_dims'] = '_'.join(map(str, dims.values())) # ex: '16_16'

    return flat_params


def log_experiment(tracker_dict, loaded_model=False, log_file_path='artifacts/on_images/deep_learning/v1/experiments.parquet', columns=['modality','version','arch_version', 'rebalance_with_weights', 'X_train.shape[0]', 'BATCH_SIZE', 'minutes_per_epoch', 'total_epochs', 'learning_rate', 'val_accuracy', 'weighted_avg_f1_score', 'min_f1_score', 'std_f1_score', 'dense_layers_sizes', 'embedding_dims', 'timestamp', 'comment', 'best_model_path']):
    """
    Met à jour la ligne correspondant à la version de l'expérience
    dans le fichier de log Parquet.
    """
    log_file = Path(log_file_path)

    # Aplatir les hyperparamètres
    flat_tracker = flatten_params_for_logging(tracker_dict)

    current_arch_version = flat_tracker['arch_version']

    if log_file.exists():
        logs_df = pd.read_parquet(log_file)
        new_log_df = pd.DataFrame([flat_tracker])
        logs_df = pd.concat([logs_df, new_log_df], ignore_index=True)
    else:
        # Créer le DataFrame si le fichier n'existe pas
        logs_df = pd.DataFrame([flat_tracker])

    logs_df = logs_df[columns]

    # Sauvegarder le fichier mis à jour
    logs_df.to_parquet(log_file)
    print(f"Log pour l'expérience arch_version {current_arch_version} ajouté dans {log_file_path} .")


def grad_cam(input_sample_dict, model, layer_name):
    """
    input_sample_dict : Dictionnaire contenant UNE seule entrée pour chaque modalité (pas de dimension batch)
    Ex: {'image_input': (500,500,3), 'text_input': (Shape...), ...}
    """

    # 1. Récupérer la couche cible
    layer = model.get_layer(layer_name)

    # 2. Modèle Grad-CAM (Inputs globaux -> [Sortie couche conv, Sortie prédiction])
    grad_model = Model(inputs=model.input, outputs=[layer.output, model.output])

    # 3. Préparer le batch de 1 pour le modèle
    # On ajoute une dimension (axis=0) à TOUTES les entrées du dictionnaire
    input_batch = {k: tf.expand_dims(v, axis=0) for k, v in input_sample_dict.items()}

    # 4. Calcul des gradients
    with tf.GradientTape() as tape:
        # On passe le dictionnaire complet au modèle
        conv_outputs, predictions = grad_model(input_batch)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    # 5. Gradients & Heatmap (Le reste est identique au standard)
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    if tf.math.reduce_max(heatmap) != 0: # Sécurité division par zero
        heatmap /= tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # 6. Récupération de l'image d'origine pour la superposition
    # Elle est dans le dictionnaire
    original_image = input_sample_dict['image_input'].numpy() # (500, 500, 3)

    # Resize heatmap
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (original_image.shape[0], original_image.shape[1])).numpy()
    heatmap_resized = np.squeeze(heatmap_resized, axis=-1)

    # Colorisation
    heatmap_colored = plt.cm.jet(heatmap_resized)[..., :3]

    # Superposition (Image doit être entre 0 et 255)
    superimposed_image = heatmap_colored * 0.6 + original_image / 255.0

    return np.clip(superimposed_image, 0, 1), predicted_class


def show_grad_cam_cnn(inputs_batch_dict, model, conv_layers, true_labels, hash_encoder, save_plot=False):
    # On déduit le nombre d'images via une des clés (ex: 'image_input')
    number_of_images = inputs_batch_dict['image_input'].shape[0]
    hashes = []

    plt.figure(figsize=(16, 4 * (len(conv_layers) + 1)))

    # row 0 of subplots (Original Images + Metadata) ---
    for i in range(number_of_images):
        plt.subplot(len(conv_layers) + 1, number_of_images, i + 1)

        # Display Image
        img_display = inputs_batch_dict['image_input'][i].numpy().astype("uint8")
        plt.imshow(img_display)

        # Decode MD5 (Integer -> String)
        encoded_val = int(inputs_batch_dict['md5_input'][i]) # Get the integer

        if encoded_val == -1: # Handle unknown values (-1)
            md5_str = None
        else:
            # Access the list of categories for the 2nd column (index 1 is MD5)
            md5_str = hash_encoder.categories_[1][encoded_val]
        hashes.append(md5_str)

        plt.title(f"Original: {true_labels[i]}", fontsize=9)
        plt.axis("off")

    # rows of grad-cam subplots
    for j, layer_name in enumerate(conv_layers):
        for i in range(number_of_images):

            # On crée un dictionnaire pour le i-ème échantillon uniquement
            single_sample = {key: value[i] for key, value in inputs_batch_dict.items()}
            # ----------------------------------

            subplot_index = i + 1 + (j + 1) * number_of_images
            plt.subplot(len(conv_layers) + 1, number_of_images, subplot_index)

            try:
                grad_cam_image, predicted_class = grad_cam(single_sample, model, layer_name)

                plt.imshow(grad_cam_image)
                plt.title(f'{layer_name}\nPred: {predicted_class}')
            except Exception as e:
                print(f"Erreur sur {layer_name}: {e}")
                # Parfois les couches "project" ont des dimensions bizarres, utile pour debug

            plt.axis("off")

    plt.tight_layout()
    if save_plot:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(f"gradcam_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

    return hashes

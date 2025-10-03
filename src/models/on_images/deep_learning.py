import pandas as pd
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Dense, Embedding


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

    # Classification

    # Quelques couches denses pour apprendre les interactions entre les différentes modalités
    x = layers.Dense(
        256, activation='relu', name='final_dense_1',
        kernel_regularizer=regularizers.l2(0.001), # Ajoute une pénalité L2
    )(all_features)

    x = layers.Dropout(0.7)(x)  # pour éviter l'overfitting

    output = layers.Dense(
        num_classes, activation='softmax', name='output',
        kernel_regularizer=regularizers.l2(0.001), # Ajoute une pénalité L2
    )(x)

    # Model

    model = keras.Model(
        inputs=[image_input, tabular_input, pHash_input, md5_input],
        outputs=output
    )

    return model


def get_custom_hyperparams(model):
    """
    Inspecte un modèle Keras et extrait les hyperparamètres
    des couches Dense et Embedding personnalisées (entraînables).
    """
    hyperparams = {
        'dense_layers_sizes': {},
        'embedding_dims': {}
    }

    for layer in model.layers:
        # On ne s'intéresse qu'aux couches que nous entraînons
        if layer.trainable:
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


def log_experiment(tracker_dict, log_file_path='artifacts/on_images/deep_learning/v1/experiments.parquet', columns=['subversion', 'rebalance_with_weights', 'X_train.shape[0]', 'BATCH_SIZE', 'minutes_per_epoch', 'max_epochs', 'total_epochs', 'learning_rate', 'val_accuracy', 'weighted_avg_f1_score', 'min_f1_score', 'std_f1_score', 'dense_layers_sizes', 'embedding_dims', 'comment', 'best_model_path']):
    """
    Met à jour la ligne correspondant à la version de l'expérience
    dans le fichier de log Parquet.
    """
    log_file = Path(log_file_path)

    # Aplatir les hyperparamètres
    flat_tracker = flatten_params_for_logging(tracker_dict)

    current_subversion = flat_tracker['subversion']

    if log_file.exists():
        logs_df = pd.read_parquet(log_file)
        # Vérifier si notre version existe déjà
        if current_subversion in logs_df['subversion'].values:
            # Mettre à jour la ligne existante
            logs_df.loc[logs_df['subversion'] == current_subversion, flat_tracker.keys()] = flat_tracker.values()
        else:
            # Sinon, ajouter la nouvelle ligne (cas d'une nouvelle expérience)
            new_log_df = pd.DataFrame([flat_tracker])
            logs_df = pd.concat([logs_df, new_log_df], ignore_index=True)
    else:
        # Créer le DataFrame si le fichier n'existe pas
        logs_df = pd.DataFrame([flat_tracker])

    logs_df = logs_df[columns]

    # Sauvegarder le fichier mis à jour
    logs_df.to_parquet(log_file)
    print(f"Log pour l'expérience subversion {current_subversion} mis à jour dans {log_file_path} .")

from tensorflow import keras
from tensorflow.keras import layers


def define_model(embedding_dim = 16, n_cols_tabular=24, num_classes = 27):
    '''
    Args:
        embedding_dim: La taille souhaitée du vecteur pour chaque hash. C'est un hyperparamètre.
        n_cols_tabular: Nombre de features numériques.
        num_classes: Nombre de classes de la variable cible.
    '''

    # Inputs

    image_input = keras.Input(shape=(500, 500, 3), name="image_input")
    tabular_input = keras.Input(shape=(n_cols_tabular,), name='tabular_input')

    pHash_vocab_size = len(preprocessors['hash'].categories_[0])
    md5_vocab_size = len(preprocessors['hash'].categories_[1])
    print(f"{pHash_vocab_size=}, {md5_vocab_size=}")

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
    dense_layers_sizes=[128]
    image_features = layers.Dense(dense_layers_sizes[-1], activation='relu', name='image_dense')(image_features)

    # Tabular branch

    dense_layers_sizes.append(64)
    tabular_features = layers.Dense(dense_layers_sizes[-1], activation='relu', name='tabular_dense_1')(tabular_input)
    dense_layers_sizes.append(32)
    tabular_features = layers.Dense(dense_layers_sizes[-1], activation='relu', name='tabular_dense_2')(tabular_features)

    # Hash branches

    # Chaque hash passe par sa propre couche d'Embedding.

    pHash_features = layers.Embedding(input_dim=pHash_vocab_size, output_dim=embedding_dim, name='pHash_embedding')(pHash_input)
    pHash_features = layers.Flatten(name='pHash_flatten')(pHash_features)  # retire une dimension superflue de taille 1

    md5_features = layers.Embedding(input_dim=md5_vocab_size, output_dim=embedding_dim, name='md5_embedding')(md5_input)
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
    dense_layers_sizes.append(256)
    x = layers.Dense(dense_layers_sizes[-1], activation='relu', name='final_dense_1')(all_features)
    #TODO: ajouter une couche dense ?

    x = layers.Dropout(0.5)(x)  # pour éviter l'overfitting
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)

    # Model

    model = keras.Model(
        inputs=[image_input, tabular_input, pHash_input, md5_input],
        outputs=output
    )

    return model

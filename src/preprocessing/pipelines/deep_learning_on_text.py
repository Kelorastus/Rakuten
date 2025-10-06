import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers

from sklearn.utils.class_weight import compute_class_weight


def concatenate_text_columns(X):
    # Concaténer les deux colonnes
    # Le jeton [SEP] est optionnel mais peut aider le deep learning
    X['full_text'] = X['designation'] + ' [SEP] ' + X['description'].fillna('')


def preprocess_features(X, y, preprocessors, full_X_train=None, full_y_train=None, shuffle=True, BATCH_SIZE = 32, rebalance_with_weights=False):
    """
    Preprocess a training or test dataset for deep learning.

    For a training dataset, preprocessors are optional, this function fits them if missing. For a test dataset, preprocessors must be given as arguments.

    Args:
        X:
        y:
        preprocessors (dict[str]):
        full_X_train: Must be provided if and only if X is (a sample of or the full) training set.
        full_y_train: Must be provided if and only if rebalance_with_weights is True and X is (a sample of or the full) training set.
        shuffle: Must be True for training, False for validation.
        BATCH_SIZE:
        rebalance_with_weights: If True, full_y_train must be provided and X must be (a sample of or the full) training set. Useful if some classes are ignored by the model.

    Returns:
        ds: Tensorflow dataset
        dict[str]: Preprocessors that were fitted by this function, if any
        dict[int]: Class weights for class imbalance.
        dict[str]: Preprocessed data that was given to the tensorflow dataset
        y: Preprocessed target.
        text_vectorizer: fitted first layer of the model.
    """

    new_preprocessors = {}

    if "target" not in preprocessors:
        new_preprocessors["target"] = LabelEncoder()
        new_preprocessors["target"].fit(y)
        preprocessors["target"] = new_preprocessors["target"]

    NUM_CLASSES=len(preprocessors["target"].classes_)

    y = preprocessors["target"].transform(y)

    # Utile sur l'ensemble du y_train (PAS sur l'échantillon)
    # On a besoin des vraies proportions.
    if full_y_train is None:
        if rebalance_with_weights:
            raise ValueError("rebalance_with_weights is True so full_y_train should be given.")
        full_y_train = y
    classes = np.unique(full_y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=full_y_train)
    # On transforme ça en un dictionnaire que Keras comprend
    class_weight_dict = dict(zip(classes, class_weights))

    y = to_categorical(y, num_classes=NUM_CLASSES)  # one-hot encoding

    if "text_vectorizer" not in preprocessors:
        if full_X_train is None:
            raise ValueError("text_vectorizer or full_X_train must be provided.")
        concatenate_text_columns(full_X_train)

        # --- Couche de preprocessing qui sera intégrée au modèle ---
        # Elle gère la tokenisation (texte -> entiers), la mise en minuscules, etc.
        new_preprocessors["text_vectorizer"] = layers.TextVectorization(
            max_tokens=20000, # Taille du vocabulaire
            output_sequence_length=310 # Longueur max des phrases (tronque/padde) (95ème percentile des nombres de mots)
        )
        # sur jeu entier de données texte
        new_preprocessors["text_vectorizer"].adapt(full_X_train['full_text'].values)

    concatenate_text_columns(X)

    # On prépare un dictionnaire de toutes nos entrées
    inputs_dict = {
        'text_input': X['full_text'].values,
    }

    ds = tf.data.Dataset.from_tensor_slices((inputs_dict, y))

    if shuffle:
        ds = ds.shuffle(1000)

    AUTOTUNE = tf.data.AUTOTUNE

    if rebalance_with_weights:
        # Créer une table de correspondance statique pour TensorFlow
        keys = list(class_weight_dict.keys())
        values = list(class_weight_dict.values())
        class_weight_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.int64, value_dtype=tf.float32),
            default_value=1.0 # Poids de 1 pour toute classe non trouvée
        )
        # Créer une fonction qui ajoute le poids au dataset
        def add_sample_weights(inputs, label):
            # 'label' est en one-hot, on doit retrouver l'index
            label_index = tf.argmax(label, axis=-1)
            # Chercher le poids correspondant à l'index de la classe
            sample_weight = class_weight_table.lookup(label_index)
            return (inputs, label, sample_weight)
        ds = ds.map(add_sample_weights)

    ds = (
        ds
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=AUTOTUNE)  # prépare le prochain lot pendant que le GPU travaille sur le lot actuel
    )
    return ds, new_preprocessors, class_weight_dict, inputs_dict, y

from pathlib import Path
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight

from src.preprocessing.image import get_image_path


def load_image_and_format(inputs, label):
    # 'inputs' est le dictionnaire produit par preprocess_features
    filepath = inputs['image_input']  # On récupère le chemin de l'image

    # On charge l'image
    image_raw = tf.io.read_file(filepath)
    image = tf.io.decode_jpeg(image_raw, channels=3)
    image = tf.image.resize(image, (500, 500))

    # On met à jour le dictionnaire : on remplace le chemin par le tenseur de l'image
    inputs['image_input'] = image

    return inputs, label


def load_preprocessors(names=['target','tabular','hash'],artifacts_folder='artifacts/on_images/deep_learning/v1'):
    folder = Path(artifacts_folder) / "preprocessors"
    preprocessors = {}
    for name in names:
        path = folder / f'{name}.joblib'
        preprocessors[name]=joblib.load(path)
        print(path)
    return preprocessors


def save_preprocessors(preprocessors,artifacts_folder='artifacts/on_images/deep_learning/v1'):
    folder = Path(artifacts_folder) / "preprocessors"
    folder.mkdir(parents=True, exist_ok=True)

    for name, preprocessor in preprocessors.items():
        path = folder / f'{name}.joblib'
        joblib.dump(preprocessor, path)
        print(path)


def preprocess_features(X, y, preprocessors, shuffle=True, BATCH_SIZE = 32):
    """
    Preprocess a training or test dataset for deep learning.

    For a training dataset, preprocessors are optional, this function fits them if missing. For a test dataset, preprocessors must be given as arguments.

    Args:
        X
        y
        preprocessors (dict[str])
        shuffle: Must be True for training, False for validation.
        BATCH_SIZE

    Returns:
        ds: Tensorflow dataset
        dict[str]: Preprocessors that were fitted by this function, if any
        dict[int]: Class weights to handle class imbalance. Useful if some classes are ignored by the model. For this dictionary to be relevant, arguments X and y should be the full training dataset instead of a small sample.
        dict[str]: Preprocessed data that was given to the tensorflow dataset
        y: Preprocessed target.
    """

    new_preprocessors = {}

    X['image_path']=X.apply(get_image_path,axis=1,as_string=True)

    X = X.drop(columns=['designation', 'description', 'gray_image_pHash', 'productid', 'imageid'])

    if "target" not in preprocessors:
        new_preprocessors["target"] = LabelEncoder()
        new_preprocessors["target"].fit(y)
        preprocessors["target"] = new_preprocessors["target"]

    NUM_CLASSES=len(preprocessors["target"].classes_)

    y = preprocessors["target"].transform(y)

    # Utile sur l'ensemble du y_train (PAS sur l'échantillon)
    # On a besoin des vraies proportions.
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    # On transforme ça en un dictionnaire que Keras comprend
    class_weight_dict = dict(zip(classes, class_weights))

    y = to_categorical(y, num_classes=NUM_CLASSES)  # one-hot encoding

    numeric_features=['mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b', 'median_r', 'median_g', 'median_b', 'mean_gray', 'std_gray', 'median_gray', 'essential_pixel_count', 'x_min', 'y_min', 'x_max', 'y_max', 'len_designation', 'len_description', 'essential_width', 'essential_height', 'essential_aspect_ratio', 'essential_area', 'rectangleness']

    if "tabular" not in preprocessors:
        new_preprocessors["tabular"] = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
            ],
            remainder='drop' # Garde les autres colonnes si besoin
        )
        new_preprocessors["tabular"].fit(X)
        preprocessors["tabular"] = new_preprocessors["tabular"]

    X_tabular_processed = preprocessors["tabular"].transform(X)

    hash_features=['image_pHash', 'image_hash_md5']

    if "hash" not in preprocessors:
        # OrdinalEncoder gère les données 2D (DataFrames) contrairement à LabelEncoder. handle_unknown est une bonne pratique pour l'inférence.
        new_preprocessors["hash"] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        new_preprocessors["hash"].fit(X[hash_features])
        preprocessors["hash"] = new_preprocessors["hash"]

    X_hashes_processed = preprocessors["hash"].transform(X[hash_features])

    # On prépare un dictionnaire de toutes nos entrées
    inputs_dict = {
        'tabular_input': X_tabular_processed,
        'pHash_input': X_hashes_processed[:, 0],
        'md5_input': X_hashes_processed[:, 1],
        'image_input': X['image_path'].values
    }

    ds = tf.data.Dataset.from_tensor_slices((inputs_dict, y))

    if shuffle:
        ds = ds.shuffle(1000)

    AUTOTUNE = tf.data.AUTOTUNE
    ds = (
        ds
        .map(load_image_and_format, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=AUTOTUNE)  # prépare le prochain lot pendant que le GPU travaille sur le lot actuel
    )
    return ds, new_preprocessors, class_weight_dict, inputs_dict, y

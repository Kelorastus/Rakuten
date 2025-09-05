import os
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed


def load_extended_df(path='../../Dataset2/df.parquet'):
    '''
    Charge le dataframe étendu.

    Exemple d'utilisation :
    df=load_extended_df(path='../../Dataset2/df.parquet')

    (Le fichier parquet a été créé à la place des CSV originaux car le calculer à partir des CSV prend longtemps.)

    Excepté la variable cible, les éléments du dataframe étendu s'obtiennent indépendamment de la varible cible et les lignes sont étendues indépendamment les unes des autres. Ainsi, on peut partir de ce dataframe étendu pour faire le train_test_split sans craindre de data leakage.

    Explication de certaines colonnes du dataframe :

    prdtypecode : variable cible (catégorie produit)

    essential_pixel_count : densité (nombre de pixels non blancs)
    x_min, y_min, x_max, y_max : extrémités de la partie non blanche de l'image

    mean_[rgb] : teinte globale de l'image
    mean_gray : luminosité de l'image
    std_[rgb] : contraste, variété des couleurs
    std_gray : variété de la luminosité
    '''
    df = pd.read_parquet(path)

    # Longueurs des textes.
    df['len_designation']=df.designation.apply(len)
    df['len_description']=df.description.fillna('').apply(len)

    # Dimensions de la partie non blanche.
    df['essential_width'] = df.x_max - df.x_min + 1
    df['essential_height'] = df.y_max - df.y_min + 1

    # À quel point la partie non blanche est en format paysage plutôt que portrait.
    df['essential_aspect_ratio'] = df.essential_width / df.essential_height

    # Aire du rectangle délimitant la partie non blanche.
    df['essential_area'] = df.essential_width * df.essential_height

    # Mesure à quel point la partie non blanche remplit son rectangle délimitant.
    df['rectangleness'] = df.essential_pixel_count / df.essential_area

    return df


def parallel_feature_creation(df, f, verbose=1):
    '''
    Applique f en parallèle (pour la rapidité) à chaque ligne de df.
    Retourne un dataframe `features_df` avec le résultat.
    Envisager ensuite `df.join(features_df)`.
    '''

    print("Démarrage de l'extraction de features en parallèle...")

    # n_jobs=-1 utilise tous les coeurs disponibles
    # backend="loky" est plus robuste pour les processus complexes
    results = Parallel(n_jobs=-1, backend="loky", verbose=verbose)(
        delayed(f)(row) for row in df.itertuples(index=False)
    )
    print("Extraction terminée.")

    features_df = pd.DataFrame(results, index=df.index)

    features_df.info()

    print("Envisager `df = df.join(features_df)`.")

    return features_df


def split_and_save_dataframe(df, test_size=0.2, output_dir = '../../Dataset2', target_column = 'prdtypecode', SPLIT_SEED = 42):
    # --- Création du split ---
    print("Création du split train/test...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=SPLIT_SEED,
        stratify=df[target_column]
    )

    # --- Sauvegarde des index ---
    os.makedirs(output_dir, exist_ok=True)
    train_indices_path = os.path.join(output_dir, 'train_indices.parquet')
    test_indices_path = os.path.join(output_dir, 'test_indices.parquet')

    print(f"Sauvegarde des index de train dans {train_indices_path}...")
    train_df.index.to_frame(name='index').to_parquet(train_indices_path)

    print(f"Sauvegarde des index de test dans {test_indices_path}...")
    test_df.index.to_frame(name='index').to_parquet(test_indices_path)

    return train_df, test_df


def load_reproducible_split(data_path = '../../Dataset2/df.parquet', train_idx_path = '../../Dataset2/train_indices.parquet', test_idx_path = '../../Dataset2/test_indices.parquet', target_column = 'prdtypecode'):
    """
    Charge le dataset complet et le divise en ensembles d'entraînement et de test
    en utilisant des fichiers d'index pré-sauvegardés.

    Args:
        data_path (str): Chemin vers le fichier de données.
        train_idx_path (str): Chemin vers le fichier contenant les index de train.
        test_idx_path (str): Chemin vers le fichier contenant les index de test.
        target_column (str): Nom de la colonne cible.

    Returns:
        tuple: Un tuple contenant (X_train, X_test, y_train, y_test).
    """
    # Charger les données et les index
    full_df = pd.read_parquet(data_path)
    train_indices = pd.read_parquet(train_idx_path)['index']
    test_indices = pd.read_parquet(test_idx_path)['index']

    # Recréer les dataframes de train et de test en utilisant .loc
    train_df = full_df.loc[train_indices]
    test_df = full_df.loc[test_indices]

    # Séparer les features (X) de la cible (y)
    X_train = train_df.drop(columns=target_column)
    X_test = test_df.drop(columns=target_column)

    y_train = train_df[target_column]
    y_test = test_df[target_column]

    return X_train, X_test, y_train, y_test

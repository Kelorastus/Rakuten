import pandas as pd


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


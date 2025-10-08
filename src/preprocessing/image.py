import numpy as np
import cv2
import imagehash
import hashlib
from PIL import Image
from pathlib import Path


def get_image_path(row, folder = 'Dataset/images/image_train', as_string=False):
    filename = f"image_{row.imageid}_product_{row.productid}.jpg"
    path = Path(folder) / filename
    if as_string:
        path = str(path)
    return path


def load_image(row, color=True, folder = 'Dataset/images/image_train'):
    '''OpenCV charge en BGR, donc l'ordre retourné est Bleu, Vert, Rouge.'''
    path = get_image_path(row, folder)
    if color:
        color_arg = cv2.IMREAD_COLOR
    else:
        color_arg = cv2.IMREAD_GRAYSCALE
    img_color = cv2.imread(str(path), color_arg)
    if img_color is None:
        raise ValueError
    return img_color


def get_image_features_with_hash(row, folder='Dataset/images/image_train', threshold=250) -> dict:
    """
    Calcule un ensemble complet de features (géométrie, couleur, hash)
    à partir d'une image, en se basant sur un masque de contenu unique.

    Returns:
        dict: Un dictionnaire contenant toutes les features calculées.
    """
    # 1. Charger l'image en couleur.
    img_color = load_image(row, color=True, folder=folder)

    # Si l'image n'a pas pu être chargée, retourner un dictionnaire vide/nul.
    if img_color is None:
        # Créer une liste de toutes les clés attendues pour la cohérence du DataFrame
        keys = [
            'image_hash', 'gray_image_hash', 'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b',
            'median_r', 'median_g', 'median_b', 'mean_gray', 'std_gray', 'median_gray',
            'essential_pixel_count', 'x_min', 'y_min', 'x_max', 'y_max'
        ]
        return {key: None for key in keys} # None est plus approprié que 0 pour le hash

    # 2. Calculer le hash perceptuel de l'image.
    # Convertir l'image de BGR (OpenCV) à RGB (Pillow).
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    # Créer un objet Image de Pillow et calculer le pHash.
    image_hash = str(imagehash.phash(Image.fromarray(img_rgb)))

    # 3. Créer le masque de contenu.
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    gray_image_hash = str(imagehash.phash(Image.fromarray(img_gray)))
    content_mask = img_gray < threshold

    # Si l'image est vide de contenu, retourner les features possibles (le hash) et des zéros.
    if not np.any(content_mask):
        # (le code pour retourner les zéros peut être factorisé, mais laissons-le pour la clarté)
        return {
            'image_hash': image_hash, 'gray_image_hash': gray_image_hash, 'mean_r': 0, 'mean_g': 0, 'mean_b': 0,
            'std_r': 0, 'std_g': 0, 'std_b': 0, 'median_r': 0, 'median_g': 0, 'median_b': 0,
            'mean_gray': 0, 'std_gray': 0, 'median_gray': 0, 'essential_pixel_count': 0,
            'x_min': 0, 'y_min': 0, 'x_max': 0, 'y_max': 0
        }

    # 4. Calculer toutes les autres features (géométriques et de couleur)
    essential_pixel_count = np.sum(content_mask)
    coords = np.argwhere(content_mask)
    y_coords, x_coords = coords[:, 0], coords[:, 1]
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

    content_pixels_color = img_color[content_mask]
    content_pixels_gray = img_gray[content_mask]

    means_color = np.mean(content_pixels_color, axis=0)
    stds_color = np.std(content_pixels_color, axis=0)
    medians_color = np.median(content_pixels_color, axis=0)

    mean_gray, std_gray, median_gray = np.mean(content_pixels_gray), np.std(content_pixels_gray), np.median(content_pixels_gray)

    # 5. Retourner le dictionnaire final avec le hash inclus.
    return {
        'image_hash': image_hash, 'gray_image_hash': gray_image_hash,
        'mean_r': means_color[2], 'mean_g': means_color[1], 'mean_b': means_color[0],
        'std_r': stds_color[2], 'std_g': stds_color[1], 'std_b': stds_color[0],
        'median_r': medians_color[2], 'median_g': medians_color[1], 'median_b': medians_color[0],
        'mean_gray': mean_gray, 'std_gray': std_gray, 'median_gray': median_gray,
        'essential_pixel_count': essential_pixel_count, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max
    }


def get_image_md5_hash(row, folder='Dataset/images/image_train') -> str:
    # Calculer le hash MD5 de l'image.
    image_path = get_image_path(row, folder)
    with open(image_path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

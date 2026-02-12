# Contexte

Ce projet s’inscrit dans une formation en alternance en machine learning engineering commencée en novembre 2024. Le projet a commencé environ 9 mois après le début de cette formation de 2 ans. Il nous permet de concrétiser nos acquis sur un défi intéressant de data science, machine learning et deep learning.

Le projet est de type apprentissage supervisé multimodal multi-classes. À partir d’une base de données de produits du site d’e-commerce Rakuten décrivant les produits par des textes et images, l’objectif est de prédire la catégorie de chaque produit. Il faudra veiller à la qualité des prédictions sur chaque catégorie pour éviter que les catégories les moins représentées soient pénalisées.

La catégorisation des produits est une problématique importante en e-commerce car :
- elle peut aider les clients à trouver plus rapidement des produits pertinents
- elle peut aider un système de recommandation à proposer des produits pertinents au clients
- une catégorisation erronée peut induire un client en erreur sur la nature du produit, augmentant le risque de retour, d’avis négatif sur le produit, un risque pour la réputation du site d’e-commerce, un risque de perte de clients et diminution de part du marché
- un retour par un client dû à une catégorie erronée peut ajouter inutilement des charges logistiques et la charge de travail du service après-vente
- les points précédents sont pertinents pour améliorer les bénéfices.

# Réalisations

Nous avons mis en place plusieurs pipelines de preprocessing.

Nous avons entraîné divers modèles de machine learning classique et de deep learning, sur différentes modalités :
- texte seul
- images et features numériques dérivés des images et du texte
- texte, images, et features numériques dérivés des images et du texte.

Concernant les modalités incluant des images, du transfer learning a été employé.

Pour l'interprétabilité du deep learning par rapport aux images, nous avons utilisé Grad-CAM.

Nous avons obtenu des résultats supérieurs au benchmark de texte et au benchmark d'images.

# Installation

Voir section "Initialization on your machine" du fichier `CONTRIBUTING.md`.

# Arborescence

```
Rakuten
├── artifacts : les artefacts des modèles de deep learning DL1, DL2, DL3 (modèles pré-entraînés, préprocesseurs au format joblib, logs de tensorboard, logs d'expériences au format parquet)
│   ├── on_images
│   │   └── deep_learning
│   │       └── v1
│   │           ├── 2025-10-02 experiments-v1.parquet : ancienne version du log d'expériences du modèle de deep learning DL1
│   │           ├── best_model_sv-9_epoch_index-01_val_accuracy-0.6365_f1-0.6253.keras : meilleur modèle de deep learning sur images pour les architectures DL1
│   │           └── *.keras : des autres modèles des architectures DL1
│   ├── on_text
│   │   ├──deep_learning
│   │   │   └── v1
│   │   │       ├── best_model_arch-10_epoch_index-12_val_accuracy-0.8214_f1-0.8209.keras : meilleur modèle de deep learning sur texte pour les architectures DL2
│   │   │       └── *.keras : des autres modèles des architectures DL2
│   │   └── tfidf_baselines
│   │       └── v1 : régénérer en lançant (`ml_tfidf_preprocessing.ipynb`).
│   │           ├── X_train_vectors.npz : matrice TF‑IDF d’entraînement (trop volumineuse pour le repo 400mb)
│   │           ├── X_valid_vectors.npz : matrice TF‑IDF de validation (trop volumineuse pour le repo 105mb)
│   │           ├── y_train.npy : labels d’entraînement
│   │           ├── y_valid.npy : labels de validation
│   │           ├── label_names.json : liste des classes
│   │           └── metadata.json : paramètres de génération des features
│   └── on_text_and_images
│       └── deep_learning
│           └── v1
│               ├── best_model_arch-11_epoch_index-01_val_accuracy-0.8338_f1-0.8337.keras : meilleur modèle de deep learning sur texte et images pour les architectures DL3
│               ├── *.keras : des autres modèles des architectures DL3
│               ├── experiments.parquet : dernière version du log d'expériences de DL1, DL2, DL3
│               ├── preprocessors : préprocesseurs des modèles de deep learning DL1, DL2, DL3
│               │   ├── hash.joblib : préprocesseurs des variables de hashes d'images
│               │   ├── tabular.joblib : préprocesseurs des variables numériques venant d'images et des longueurs de textes
│               │   ├── target.joblib: préprocesseur de la variable cible
│               │   └── text_vectorizer.joblib : préprocesseur d'une variable textuelle combinant les variables textuelles initiales
│               ├── tensorboard_logs : derniers logs de tensorboard
│               └── tensorboard_logs_v1 : logs de tensorboard pour une ancienne configuration
├── CONTRIBUTING.md : instructions pour installer et/ou contribuer
├── Dataset
│   ├── images.zip : archive originale des images
│   ├── images : contenu de l'archive images.zip
│   │   ├── image_test
│   │   │   └── *.jpg
│   │   └── image_train
│   │       └── *.jpg
│   ├── X_test.csv : jeu de test original
│   ├── X_train.csv : jeu d'entraînement original
│   ├── Y_train.csv : variable cible du jeu d'entraînement original
│   ├── X_en.csv
│   ├── X_fr.csv
│   ├── y_en.csv
│   └── y_fr.csv
├── Dataset2
│   ├── df.parquet : jeu de données contenant le jeu d'entraînement original `X_train.csv` et sa variable cible `Y_train.csv` , étendu par feature engineering sur les images (via des calculs qui durent longtemps), créé depuis le notebook `notebooks/data_exploration/exploration_and_feature_engineering.ipynb`, et à charger par la fonction `load_extended_df` ou `load_reproducible_split` du fichier `src/preprocessing/core.py`
│   ├── test_indices.parquet : indices du jeu de test d'un `train_test_split` fait sur `df.parquet` ; ce `train_test_split` est reproductible via la fonction `load_reproducible_split` du fichier `src/preprocessing/core.py`
│   └── train_indices.parquet : indices du jeu d'entraînement d'un `train_test_split` fait sur `df.parquet` ; ce `train_test_split` est reproductible via la fonction `load_reproducible_split` du fichier `src/preprocessing/core.py`
├── notebooks : notebooks Jupyter
│   ├── data_exploration
│   │   ├── exploration_and_feature_engineering.ipynb : feature engineering pour créer `Dataset2/df.parquet` et exploration de ses variables (y compris les variables du jeu de données original), mais principalement les variables liées aux images
│   │   ├── images : quelques images créées par le notebook `exploration_and_feature_engineering.ipynb`
│   │   │   ├── proportion de na dans description.png
│   │   │   ├── wordcloud_description.png
│   │   │   └── wordcloud_designation.png
│   │   ├── Jessy.ipynb
│   │   ├── exploration_listing_text_stats.ipynb :
│   │   ├── meilleur_modele_optimise_en.pkl
│   │   ├── meilleur_modele_optimise_fr.pkl
│   │   ├── nltk_data
│   │   ├── vectorizer_optimise.pkl
│   │   └── X_tfidf.npz
│   ├── deep_learning_on_images
│   │   └── DL1.ipynb : modèle DL1 de deep learning sur les images (entraînement, évaluation, Grad-CAM)
│   ├── deep_learning_on_text
│   │   └── DL2.ipynb : modèle DL2 de deep learning sur le texte (entraînement, évaluation)
│   ├── deep_learning_on_text_and_images
│   │   └── DL3.ipynb : modèle DL3 de deep learning sur texte et images (entraînement, évaluation, Grad-CAM)
│   └── machine_learning_on_text
│       ├── ml_tfidf_preprocessing.ipynb : prétraitement du texte + vectorisation TF‑IDF (mots + caractères)
│       ├── ml_tfidf_models_comparison.ipynb : entraîne 4 modèles classiques sur les mêmes features TF‑IDF et   compare les scores
│       └── ML.md : note de synthèse sur le preprocessing et la comparaison des modèles Machine Learning
├── README.md : contexte, réalisations, arborescence
├── requirements.txt : dépendances essentielles (voir `CONTRIBUTING.md`)
├── requirements.lock : dépendances essentielles et leurs dépendances (voir `CONTRIBUTING.md`)
├── requirements-gpu.txt : dépendances essentielles pour tensorflow si GPU Nvidia (voir `CONTRIBUTING.md`)
├── requirements-gpu.lock : dépendances essentielles pour tensorflow si GPU Nvidia, et leurs dépendances (voir `CONTRIBUTING.md`)
├── requirements-cpu.lock : dépendances essentielles pour tensorflow sans GPU Nvidia (voir `CONTRIBUTING.md`)
├── requirements-cpu.txt : dépendances essentielles pour tensorflow [sans GPU Nvidia], et leurs dépendances (voir `CONTRIBUTING.md`)
└── src : modules et scripts python
    ├── fix_import_and_cwd_in_notebooks.py : code pour réparer les imports dans les notebooks
    ├── fix_tensorboard.py : script pour aider à la transition lors d'un changement de configuration de tensorboard
    ├── models
    │   ├── on_images
    │   │   └── deep_learning.py : architecture du modèle DL1 de deep learning sur les images, et fonction pour Grad-CAM
    │   ├── on_text
    │   │   └── deep_learning.py : architecture du modèle DL2 de deep learning sur le texte
    │   └── on_text_and_images
    │       └── deep_learning.py : architecture du modèle DL3 de deep learning sur texte et images ; fonctions pour log d'expériences et pour Grad-CAM
    │── preprocessing
    │   ├── core.py : fonctions pour créer ou charger `Dataset2/df.parquet`
    │   ├── image.py : fonctions liées aux images (chargement, feature engineering)
    │   └── pipelines
    │       ├── deep_learning.py : chargement et sauvegarde des préprocesseurs des modèles de deep learning DL1, DL2, DL3
    │       ├── deep_learning_on_images.py : preprocessing du modèle DL1 de deep learning sur les images
    │       ├── deep_learning_on_text.py : preprocessing du modèle DL2 de deep learning sur le texte
    │       └── deep_learning_on_text_and_images.py : preprocessing du modèle DL3 de deep learning sur texte et images
    │── streamlit-archive
    │   ├── app.py : ancienne structure de la présentation streamlit du projet
    │   ├── generate_page.py : script pour générer un nouveau brouillon de fichier dans le dossier `parts`
    │   └── parts : pages de la présentation streamlit
    │       └── *.py
    └── streamlit.py : présentation streamlit du projet
```

# Documents "analyse, feature engineering, preprocess et modelisation..."

Les documents "analyse, feature engineering, preprocess et modelisation ML+DL.ipynb" et "analyse.... avec visualisation ML" sont les mêmes. Le premier document donne le retour notebook du travail de transfert-learning multimodal avec pytorch tandis que le second donne le retour notebook de la visualisation de données, et du travail de machine learning uniquement sur du texte avec seulement des lignes en français et en anglais (séparement l'un de l'autre).

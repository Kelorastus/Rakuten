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

# Machine Learning sur le texte (TF-IDF)

**Objectif**  
Mettre en place un pipeline de prétraitement du texte avec TF-IDF, puis comparer plusieurs modèles de classification linéaires sur les mêmes features. L’évaluation se fait surtout avec l’accuracy et le F1 pondéré.

**Notebooks**
- `notebooks/machine_learning_on_text/ml_tfidf_preprocessing.ipynb` : prétraitement et vectorisation TF‑IDF.
- `notebooks/machine_learning_on_text/ml_tfidf_models_comparison.ipynb` : entraînement des 4 modèles et comparaison des scores.

**Workflow**
1. Lancer `ml_tfidf_preprocessing.ipynb` pour générer les matrices TF‑IDF.
2. Lancer `ml_tfidf_models_comparison.ipynb` pour entraîner et comparer les modèles.

**Paramètres globaux**
- `RANDOM_STATE = 42`
- `TEST_SIZE = 0.2`
- `CLASS_WEIGHT = 'balanced'` (quand le modèle le supporte)

**Pourquoi ces paramètres ?**
- `RANDOM_STATE` : fixer l’aléatoire pour obtenir des résultats reproductibles.
- `TEST_SIZE` : garder 20% des données pour évaluer le modèle sur un jeu non vu.
- `CLASS_WEIGHT` : compenser le déséquilibre des classes en donnant plus de poids aux classes rares.

**Prétraitement (ml_tfidf_preprocessing.ipynb)**
Le notebook réalise les étapes suivantes.
- Chargement des données `Dataset/X_train.csv` et `Dataset/Y_train.csv` avec l’index commun.
- Fusion features/labels et création d’un texte unique `text_test` à partir de `designation` et `description`.
- Nettoyage du texte : suppression du HTML, passage en minuscules, normalisation des espaces, suppression des accents.
- Tokenisation et lemmatisation avec spaCy, puis filtrage des stopwords (français + anglais) avec NLTK.
- Vectorisation TF‑IDF mots (1‑2 grammes) avec lemmatiseur.
- Vectorisation TF‑IDF caractères (3‑5 grammes) pour capturer les sous‑mots et fautes.
- Split train/validation stratifié avec `TEST_SIZE = 0.2`.
- Sauvegarde des matrices et labels dans `artifacts/on_text/tfidf_baselines/v1/`.

**Comparaison des modèles (ml_tfidf_models_comparison.ipynb)**
Ce notebook charge les matrices TF‑IDF puis entraîne, séparément, 4 modèles.
- LinearSVC
- SGDClassifier (log loss)
- LogisticRegression (solver saga)
- MultinomialNB

Pour chaque modèle, on calcule :
- `accuracy`
- `f1_weighted`
- `classification_report` complet

**Résultats (notre exécution)**
On se concentre sur l’accuracy et le F1 pondéré.
- LinearSVC : Accuracy ≈ 0,853, F1 pondéré ≈ 0,852
- LogisticRegression : Accuracy ≈ 0,834, F1 pondéré ≈ 0,835
- SGDClassifier : Accuracy ≈ 0,789, F1 pondéré ≈ 0,792
- MultinomialNB : Accuracy ≈ 0,668, F1 pondéré ≈ 0,642

**Conclusion**
Le meilleur compromis performance/qualité est **LinearSVC**, qui obtient les meilleurs scores en accuracy et F1 pondéré. LogisticRegression est proche en score mais a été beaucoup plus lente sur notre run. SGDClassifier est correct mais moins performant, et MultinomialNB est nettement en dessous.

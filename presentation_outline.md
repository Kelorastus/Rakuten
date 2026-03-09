# Presentation - Partie Jimmy: Machine learning sur texte

Duree visee: ~5 minutes pour ma partie. Chaque slide contient le contenu a afficher et mon script oral.

## Slide 1 - Partie Jimmy / Machine learning sur texte
### Contenu a afficher
- Slide d'ouverture avec le titre: "Partie Jimmy".
- Sous-titre: "Machine learning sur texte".
- Logo Rakuten en haut a droite.
### Script a dire
- "Dans cette partie, je me concentre sur la classification des produits a partir du texte uniquement."
- "Je vais d'abord presenter le preprocessing, puis comparer plusieurs modeles de machine learning."

## Slide 2 - Preprocessing
### Contenu a afficher
- Slide de transition de section: "Preprocessing".
- Meme template visuel Rakuten.
### Script a dire
- "Je commence par le preprocessing, qui est une etape cle pour stabiliser les performances."
- "Je le presente en deux temps: nettoyage/normalisation, puis vectorisation TF-IDF."

## Slide 3 - Nettoyage & Normalisation
### Contenu a afficher
- Bloc `Clean_text`:
  - Suppression HTML (BeautifulSoup)
  - Unescape: minuscules
  - Unicode: normalisation espaces
- Bloc `Filtrage tokens`:
  - `token.is_alpha`
  - Longueur > 2
  - Exclusion stopwords pour garder les mots porteur
- Bloc `Gestion des NaN`:
  - Remplacement par chaines vides
  - Concat titre et description: `text_test`
- Bloc `Stopwords`:
  - FR+EN (NLTK)
  - Lemmatisation spaCy: `fr_core_news_sm`
- Capture "Tokens conserves" + comparaison "Brut (titre + desc)" vs "Apres clean_text".
### Script a dire
- "Sur cette slide, je standardise le texte: suppression du HTML, normalisation unicode et passage en minuscules pour homogeniser les entrees."
- "Je gere aussi les NaN en les remplacant par des chaines vides, puis je concatene titre et description dans `text_test`."
- "Enfin, je filtre le bruit avec stopwords + lemmatisation, et le tableau montre les tokens effectivement conserves."

## Slide 4 - Vectorization TF-IDF
### Contenu a afficher
- Bloc `FeatureUnion`:
  - TF-IDF mots
  - TF-IDF caracteres
- Bloc `Matrices creuses generees`:
  - `X_train_vectors` (67932 x 317321)
  - `X_valid_vectors` (16984 x 317321)
  - `y_train/y_valid`
- Bloc `Artefacts`: "Pour eviter de recalculer".
- Capture console des shapes train/validation.
### Script a dire
- "Ici, je combine deux representations via `FeatureUnion`: n-grammes de mots et n-grammes de caracteres."
- "On obtient des matrices creuses de grande dimension, visibles au centre avec les shapes train et validation."
- "Ces features sont sauvegardees en artefacts pour eviter de relancer toute la vectorisation a chaque modele."

## Slide 5 - Modelisation Machine Learning
### Contenu a afficher
- Slide de transition de section: "Modelisation Machine Learning".
- Meme template visuel Rakuten.
### Script a dire
- "Une fois les features TF-IDF pretes, je compare quatre modeles classiques."
- "L'objectif est de trouver le meilleur compromis entre performance, cout de calcul et usage metier."

## Slide 6 - LinearCSV
### Contenu a afficher
- Bloc `Algorithme`:
  - SVM lineaire sur vecteurs TF-IDF
- Bloc `Hyperparametres cles`:
  - `C = 1.0`
  - `class_weight = "balanced"`
  - `random_state = 42`
- Bloc `Scores validation`:
  - Accuracy = 0,853
  - F1 macro = 0,839
  - F1 weighted = 0,852
- Bloc `Comportement par classe`:
  - Code 2583 monte a 0,98 de F1
  - Codes 10 et 1281 autour de 0,6 (classes rares)
  - Meilleure baseline texte-only, robuste grace aux poids de classe
- Captures: classification report + histogramme des classes.
### Script a dire
- "LinearCSV est notre meilleure baseline texte-only, avec 0,852 en F1 weighted."
- "Le `class_weight='balanced'` aide a tenir les classes rares, meme si certaines restent autour de 0,6."
- "La classe 2583 performe tres haut, et globalement ce modele offre le meilleur compromis precision/robustesse."

## Slide 7 - Logistic Regression
### Contenu a afficher
- Bloc `Algorithme`:
  - Regression logistique multinomiale
- Bloc `Hyperparametres cles`:
  - `C = 1.0`
  - `class_weight = "balanced"`
  - `random_state = 42`
- Bloc `Scores validation`:
  - Accuracy = 0,834
  - F1 macro = 0,821
  - F1 weighted = 0,835 (legerement sous LinearSVC)
- Bloc `Limites`:
  - `max_iter`
  - Convergence plus lente
  - Temps d'entrainement plus long (128 minutes contre 45 secondes avec linearSVC)
- Message metier:
  - Alternative quand on a besoin de scores probabilistes exploitables, au prix d'un cout compute superieur.
- Capture classification report.
### Script a dire
- "La Logistic Regression est juste derriere en performance, avec 0,835 en F1 weighted."
- "Son vrai atout, c'est la sortie probabiliste utile pour prioriser ou scorer des demandes metier."
- "En contrepartie, elle est beaucoup plus lente a entrainer, avec une convergence plus couteuse."

## Slide 8 - SGDClassifier
### Contenu a afficher
- Bloc `Algorithme`:
  - Gradient stochastique
- Bloc `Hyperparametres cles`:
  - `alpha = 1e-4`
  - `class_weight = "balanced"`
  - `max_iter = 1000`
- Bloc `Scores validation`:
  - Accuracy = 0,789
  - F1 macro = 0,780
  - F1 weighted = 0,792
- Bloc `Avantages`:
  - Entrainement tres rapide
  - Possibilite de re-entrainer incrementalement sur de nouvelles donnees
- Bloc `Limites`:
  - Sensibilite accrue aux classes rares
  - Perte d'environ 6 points de F1 weighted vs LinearSVC
- Capture classification report.
### Script a dire
- "SGDClassifier est le plus leger et le plus rapide a re-entrainer."
- "C'est un bon choix quand l'infra est contrainte ou quand on veut des mises a jour frequentes."
- "Mais on perd nettement en qualite, surtout sur les classes rares."

## Slide 9 - MultinomialNB
### Contenu a afficher
- Bloc `Algorithme`:
  - Naive Bayes multinomial
- Bloc `Mise en oeuvre`:
  - Aucune ponderation de classes possible
  - Forte influence des categories majoritaires
- Bloc `Observations`:
  - F1 = 0,75 sur code 2583
  - F1 = 0,25 sur code 10
  - Mauvais rappel sur classes rares
- Bloc `Positionnement`:
  - Baseline "sanity check" ultra rapide
  - Surtout utile pour verifier le pipeline TF-IDF, plutot qu'un candidat production
- Capture classification report.
### Script a dire
- "MultinomialNB sert surtout de baseline de controle, pas de modele cible."
- "Il est tres rapide, mais degrade fortement sur les classes minoritaires faute de ponderation."
- "Si ce modele depasse les autres, c'est generalement un signal que le pipeline a un probleme."

## Slide 10 - Tableau comparatif
### Contenu a afficher
- Message cle 1:
  - `LinearSVC` reste la reference (meilleure combinaison precision/rappel)
- Message cle 2:
  - `LogisticRegression` = option probabiliste quand on veut scorer/ordonner les demandes
- Message cle 3:
  - `SGDClassifier` = choix "rapide/infra legere" pour re-entrainements frequents
- Message cle 4:
  - `MultinomialNB` = point de controle pour detecter un pipeline TF-IDF casse
- Message transversal:
  - Tous exploitent le meme encodage TF-IDF -> facile de brancher une future modalite image sans tout refaire
- Tableau recapitulatif des scores et points cles (capture en bas a droite).
### Script a dire
- "Le classement final est clair: LinearSVC est la reference en texte-only."
- "LogisticRegression est le plan B quand le besoin metier priorise des probabilites exploitables."
- "SGD est l'option rapide, et MultinomialNB reste un garde-fou pour verifier la sante du pipeline."
- "Comme tout repose sur le meme socle TF-IDF, on peut ajouter une modalite image ensuite sans casser l'existant."

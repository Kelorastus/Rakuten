# Le code suivant dans un notebook permet d'autoriser les imports de fichiers python de ce repo.

import sys
import os

# Ce code cherche le dossier racine en remontant dans l'arborescence
# jusqu'à ce qu'il trouve le dossier 'src'.
# Cela le rend indépendant de l'endroit où vous lancez le notebook.
try:
    # On part du dossier du notebook
    notebook_dir = os.path.dirname(__file__)
except NameError:
    # __file__ n'existe pas en mode interactif, on utilise le répertoire de travail
    notebook_dir = os.getcwd()

# On remonte jusqu'à trouver un dossier contenant 'src'
project_root = notebook_dir
while not os.path.isdir(os.path.join(project_root, 'src')):
    parent_dir = os.path.dirname(project_root)
    if parent_dir == project_root: # On a atteint la racine du système
        raise FileNotFoundError("Impossible de trouver le dossier 'src'. Vérifiez la structure du projet.")
    project_root = parent_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Ajout de '{project_root}' au sys.path .")

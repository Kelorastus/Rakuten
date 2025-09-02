# Rakuten

## Initialization on your machine

1. Recommended Python version: 3.10.
2. Create a virtual environment: from the folder of this repository, run the command `python3 -m venv venv` in a terminal.
3. Activate it: `source venv/bin/activate`.
4. Install dependencies: `pip install -r requirements.txt`.

## To add new dependencies

1. Activate the virtual environment (if not yet done in the current terminal): from the folder of this repository, run `source venv/bin/activate`.
2. For example, to add pandas to the virtual environment, run `pip install pandas`.
3. Update the requirements file: `pip freeze > requirements.txt`.
4. Commit `requirements.txt` to main and push.
5. Tell your collaborators: "Please pull main, then run (from the repository folder) the commands `source venv/bin/activate` then `pip install -r requirements.txt`. Then restart VS Code or the Jupyter kernels."

## Contributing guidelines

Éviter de travailler à plusieurs en même temps sur :
- le même notebook
- la même branche.

Si on veut travailler à plusieurs sur la même tâche d'un notebook, il vaut mieux créer chacun un notebook avec un nom différent, puis envisager de les réunir après coup en un seul notebook.

Chacun travaille dans sa branche. Faire un merge entre sa branche et main régulièrement (environ une fois par semaine) puis supprimer et recréer sa branche à partir de main. (Pour éviter d'avoir à faire un merge monstrueux si on a des branches trop différentes.)

### Si difficulté à push main

Si quelqu'un a fait un ou des commits sur main sans avoir fetch la dernière version de main, ça peut poser problème.
Alors la procédure suivante peut servir :
- créer une branche qui a les commits que la personne veut push
- faire des commits sur cette branche si il y a encore des changements importants non committed (pour bien les sauvegarder)
- push cette branche
- revenir sur main
- `git fetch origin`
- `git reset --hard origin/main` (c'est une commande irréversible qui réinitialise la branche main du PC de la personne pour qu'elle soit pareil que la branche main de Github)
- puis merge la branche personnelle vers main
- push main
- supprimer la branche personnelle du PC de la personne et du remote origin (Github) (comme recommandé après chaque merge).

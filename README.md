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

Chacun travaille plutôt dans sa branche que sur main.

### Si tu veux travailler sur la branche main (par exemple pour merge ta branche personnelle) :
- `pull main` juste avant
- faire le commit rapidement
- `push` main rapidement.
(La rapidité permet d'éviter les conflits vu qu'on se partage la branche main.)

### Merge

Faire un merge entre sa branche et main régulièrement (environ une fois par semaine).
Puis :
- supprimer sa branche sur son PC et sur le remote origin (Github)
- recréer sa branche à partir de main.

(Pour éviter d'avoir à faire un merge monstrueux si on a des branches trop différentes.)

### Si tu as déjà fait un commit sur main pas encore push, et que quelqu'un a push sur main depuis ton dernier pull main :

- crée une branche qui a les commits que tu veux push
- commit sur cette branche si tu as encore des changements non committed (pour bien les sauvegarder)
- push cette branche
- reviens sur main
- `git fetch origin`
- `git reset --hard origin/main` (c'est une commande irréversible qui réinitialise ta branche main pour qu'elle soit pareille que celle de Github)
- puis merge ta branche vers main
- push main
- supprimer ta branche de ton PC et du remote origin (Github) (comme recommandé après chaque merge)

C'est pour éviter d'avoir à faire cette manipulation qu'on préfère utiliser une branche par personne. Sinon il faut bien penser à pull main avant de commit sur main, puis à push rapidement.

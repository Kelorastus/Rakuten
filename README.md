# Rakuten

## Initialization on your machine

1. Recommended Python version: 3.10.
2. Clone the repository.
3. Create a virtual environment: in a terminal in the folder of this repository, run the command `python3 -m venv venv`.
4. Activate it: `source venv/bin/activate`.
5. If you don't have an NVIDIA GPU: `pip install -r requirements-cpu.lock`
6. If you have an NVIDIA GPU:
    - Make sure your NVIDIA drivers are installed.
    - If you are on Windows: install WSL2 and the NVIDIA CUDA Driver for WSL. Then, open your WSL terminal (e.g., Ubuntu) and proceed as if you are on Linux, activating the environment of the repository as in step 4.
    - `pip install -r requirements-gpu.lock`

## To clean up your venv if anything went wrong

Delete the `venv` folder of your local repository.
Then repeat the above from step 3 "Create a virtual environment".

## To add new dependencies

1. Activate the virtual environment (if not yet done in the current terminal): from the folder of this repository, run `source venv/bin/activate`.
2. For example, to add the dependency `wordcloud` version 1.9.4: add the line `wordcloud==1.9.4` to the requirements file `requirements.txt`. (If the dependency depends on having / lacking an NVIDIA GPU, the line should instead be added to `requirements-gpu.txt` or `requirements-cpu.txt`, respectively.)
3. Install: do step 5 or 6 above in the section "Initialization on your machine".
4. Update your lock file:
    - if you have an NVIDIA GPU: `pip freeze > requirements-gpu.lock`.
    - if you don't have an NVIDIA GPU: `pip freeze > requirements-cpu.lock`.
4. Commit the modified `requirements...` file(s) to main and push.
5. Tell your collaborators: "Please pull main, then perform steps 4 to 6 of the file `README.md`, then restart VS Code or the Jupyter kernels of any open notebook."

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

- créer une branche qui a les commits que tu veux push
- commit sur cette branche si tu as encore des changements non committed (pour bien les sauvegarder)
- push cette branche
- revenir sur main
- `git fetch origin`
- `git reset --hard origin/main` (c'est une commande irréversible qui réinitialise ta branche main pour qu'elle soit pareille que celle de Github)
- merge ta branche vers main
- push main
- supprimer ta branche de ton PC et du remote origin (Github) (comme recommandé après chaque merge)

C'est pour éviter d'avoir à faire cette manipulation qu'on préfère utiliser une branche par personne. Sinon il faut bien penser à pull main avant de commit sur main, puis à push rapidement.

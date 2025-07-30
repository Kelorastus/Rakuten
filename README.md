# Rakuten

## Initialization on your machine

1. Recommended Python version: 3.10.
2. Create a virtual environment: from the folder of this repository, run the command `python3 -m venv venv` in a terminal.
3. Activate it: `source venv/bin/activate`.
4. Install dependencies: `pip install -r requirements.txt`.

## To install new dependencies

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

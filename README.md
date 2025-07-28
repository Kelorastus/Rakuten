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
4. Commit `requirements.txt` and push.

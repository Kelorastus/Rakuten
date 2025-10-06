from pathlib import Path
import joblib


def load_preprocessors(names=['target','tabular','hash'],artifacts_folder='artifacts/on_images/deep_learning/v1'):
    folder = Path(artifacts_folder) / "preprocessors"
    preprocessors = {}
    for name in names:
        path = folder / f'{name}.joblib'
        preprocessors[name]=joblib.load(path)
        print(path)
    return preprocessors


def save_preprocessors(preprocessors,artifacts_folder='artifacts/on_images/deep_learning/v1'):
    folder = Path(artifacts_folder) / "preprocessors"
    folder.mkdir(parents=True, exist_ok=True)

    for name, preprocessor in preprocessors.items():
        path = folder / f'{name}.joblib'
        joblib.dump(preprocessor, path)
        print(path)

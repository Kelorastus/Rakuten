"""
Script pour ranger après un usage erroné de TensorBoard.
"""

import os
import shutil
from pathlib import Path

# --- À configurer ---
# Le dossier où se trouvent actuellement vos dossiers 'train' et 'validation'
source_root = Path("artifacts/on_images/deep_learning/v1/tensor_board_wrong")
# Le dossier où vous voulez mettre les logs organisés
target_root = Path("artifacts/on_images/deep_learning/v1/tensorboard_logs")
# --------------------

runs = {} # Dictionnaire pour regrouper les fichiers par PID

print("Scan des fichiers de log...")
# Parcourir les dossiers train et validation
for log_type in ["train", "validation"]:
    source_dir = source_root / log_type
    if not source_dir.exists():
        continue

    for filename in os.listdir(source_dir):
        try:
            # Extraire le PID du nom de fichier (ex: 7025)
            # events.out.tfevents.1759395635.pop-os.7025.0.v2
            pid = filename.split('.')[-3]

            # Initialiser le dictionnaire si c'est la première fois qu'on voit ce PID
            if pid not in runs:
                runs[pid] = {'train': [], 'validation': []}

            # Ajouter le chemin complet du fichier à la bonne liste
            runs[pid][log_type].append(source_dir / filename)
        except IndexError:
            print(f"  - Ignoré (nom de fichier non standard): {filename}")

print(f"Trouvé {len(runs)} runs distincts.")

# Créer les nouveaux dossiers et déplacer les fichiers
for pid, files_dict in runs.items():
    # Créer un dossier de destination pour ce run, ex: .../run_7025_train_val
    run_dest_folder = target_root / f"run_{pid}"

    print(f"Organisation du run {pid} dans {run_dest_folder}...")

    for log_type, file_paths in files_dict.items():
        if not file_paths:
            continue

        # Créer le sous-dossier train/validation
        type_dest_folder = run_dest_folder / log_type
        type_dest_folder.mkdir(parents=True, exist_ok=True)

        # Déplacer chaque fichier
        for file_path in file_paths:
            shutil.move(str(file_path), str(type_dest_folder))

print("\nRéorganisation terminée !")
print(f"Lancez maintenant: tensorboard --logdir {target_root}")

# Optionnel : supprimer les anciens dossiers 'train' et 'validation' s'ils sont vides
if not os.listdir(source_root / "train"):
    os.rmdir(source_root / "train")
if not os.listdir(source_root / "validation"):
    os.rmdir(source_root / "validation")

import os
from pathlib import Path

# Chemin des données
data_path = "https://raw.githubusercontent.com/efagbola/option-prices-project/refs/heads/main/code/Data/"

# Déterminer le chemin du dossier output
def get_output_path():
    # Chemin du dossier parent du dossier actuel
    parent_dir = Path(__file__).parent.parent
    output_dir = parent_dir / 'output'
    
    # Créer le dossier s'il n'existe pas
    output_dir.mkdir(exist_ok=True)
    
    # Retourner le chemin absolu sous forme de chaîne avec un séparateur à la fin
    return str(output_dir) + os.sep

# Définir les chemins globaux
DATA_PATH = data_path
OUTPUT_PATH = get_output_path()

import pandas as pd
import re
import os

INPUT_FILE = ".armedangels/all_armedangels_products.xlsx"
OUTPUT_FILE = ".armedangels/armedangels_dataset.xlsx"

def clean_text_content(text):
    if not isinstance(text, str):
        return ""
    
    # Passage en minuscules
    text = text.lower()
    
    # Suppressions spécifiques (contenu entre crochets)
    text = re.sub(r'\[.*?\]', ' ', text)

    # Suppression des CHIFFRES (0-9)
    text = re.sub(r'\d+', '', text)

    # Suppression des CARACTÈRES SPÉCIAUX
    # Cible tout ce qui n'est pas (^) un mot (\w) ou un espace (\s).
    # On remplace par un espace ' ' pour éviter de coller deux mots
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Suppression spécifique du underscore
    text = text.replace('_', ' ')

    # Nettoyage final des espaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def run_cleaning():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File {INPUT_FILE} not found.")
        return

    print("--- Loading Raw Data ---")
    df = pd.read_excel(INPUT_FILE)
    print(f"Original shape: {df.shape}")

    # Standardisation des colonnes
    df.columns = df.columns.str.lower().str.strip()

    # Suppression des doublons exacts sur le nom
    if 'name' in df.columns:
        df = df.drop_duplicates(subset=['name'], keep='first')
        print(f"After deduplication: {df.shape}")

    # Application du nettoyage profond sur la description
    print("Cleaning text content (removing numbers & special chars)...")
    if 'description' in df.columns:
        df['description_clean'] = df['description'].apply(clean_text_content)
        
        # Suppression des lignes vides après nettoyage
        df = df[df['description_clean'].str.len() > 3]
    else:
        print("Warning: 'description' column not found!")

    #Sauvegarde
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"--- DONE. Cleaned Dataset saved to {OUTPUT_FILE} ---")

if __name__ == "__main__":
    run_cleaning()
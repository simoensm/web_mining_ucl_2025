import pandas as pd
import re
import os

INPUT_FILE = ".patagonia/patagonia_products.xlsx"
OUTPUT_FILE = ".patagonia/patagonia_dataset.xlsx"

def clean_text_content(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Passage en minuscules
    text = text.lower()
    
    # 2. Suppressions spécifiques (contenu entre crochets)
    text = re.sub(r'\[.*?\]', ' ', text)

    # 3. Suppression des CHIFFRES (0-9)
    text = re.sub(r'\d+', '', text)

    # 4. Suppression des CARACTÈRES SPÉCIAUX
    # Le pattern [^\w\s] cible tout ce qui n'est pas (^) un mot (\w) ou un espace (\s).
    # On remplace par un espace ' ' pour éviter de coller deux mots (ex: "waterproof/breathable" -> "waterproof breathable")
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Suppression spécifique du underscore (_) qui est parfois considéré comme un caractère mot
    text = text.replace('_', ' ')

    # 5. Nettoyage final des espaces (doubles espaces -> simple espace)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def run_cleaning():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File {INPUT_FILE} not found.")
        return

    print("--- Loading Raw Data ---")
    df = pd.read_excel(INPUT_FILE)
    print(f"Original shape: {df.shape}")

    # 1. Standardisation des colonnes
    df.columns = df.columns.str.lower().str.strip()

    # 2. Suppression des doublons exacts sur le nom
    if 'name' in df.columns:
        df = df.drop_duplicates(subset=['name'], keep='first')
        print(f"After deduplication: {df.shape}")

    # 3. Application du nettoyage profond sur la description
    print("Cleaning text content (removing numbers & special chars)...")
    if 'description' in df.columns:
        df['description_clean'] = df['description'].apply(clean_text_content)
        
        # 4. Suppression des lignes vides après nettoyage (moins de 3 caractères restants)
        df = df[df['description_clean'].str.len() > 3]
    else:
        print("Warning: 'description' column not found!")

    # Sauvegarde
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"--- DONE. Cleaned Dataset saved to {OUTPUT_FILE} ---")

if __name__ == "__main__":
    run_cleaning()
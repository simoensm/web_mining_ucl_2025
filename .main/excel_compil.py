#Code pouvant aggréger plusieurs fichiers Excel ou CSV en un seul fichier Excel.
import pandas as pd # Pour la manipulation des données
import glob # Pour la recherche de fichiers
import os

FILE_TYPE = 'xlsx'  #Type de fichier à traiter ('xlsx' ou 'csv')
OUTPUT_FILENAME = 'all_products.xlsx' #Nom du fichier de sortie

def merge_files():
    search_pattern = f"*.{FILE_TYPE}" 
    files = glob.glob(search_pattern) #Trouver tous les fichiers du type spécifié dans le répertoire courant
    
    if OUTPUT_FILENAME in files:
        files.remove(OUTPUT_FILENAME) #Éviter de ré-inclure le fichier de sortie s'il existe déjà
        
    all_data_frames = [] #Liste pour stocker les tables de chaque fichier
    
    print(f"Found {len(files)} files to merge...")
     
    for filename in files: #Lire chaque fichier et l'ajouter à la liste
        try: 
            if FILE_TYPE == 'xlsx':
                df = pd.read_excel(filename) 
            else:
                df = pd.read_csv(filename)
            
            df['Source_File'] = filename
            
            all_data_frames.append(df) #Ajouter la table à la liste
            print(f" -> Successfully read: {filename} ({len(df)} rows)")
            
        except Exception as e:
            print(f" ! Error reading {filename}: {e}") #Pour gérer les erreurs de lecture de fichiers

    if all_data_frames:
        print("Merging data...")
        merged_df = pd.concat(all_data_frames, ignore_index=True) #Fusionner toutes les tables en une seule, ignorant les index d'origine
   
        merged_df.to_excel(OUTPUT_FILENAME, index=False) #Enregistrer la table fusionnée dans un nouveau fichier Excel, sans les index
        print(f"\nSUCCESS: All files merged into '{OUTPUT_FILENAME}'")
        print(f"Total products: {len(merged_df)}")
    else:
        print("No files found to merge.")

if __name__ == "__main__":
    merge_files()
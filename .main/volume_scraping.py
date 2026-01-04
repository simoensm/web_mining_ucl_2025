import pandas as pd
import matplotlib.pyplot as plt
import os

# --- PARTIE 1 : COLLECTE INITIALE (URLS) ---

fichiers_txt = {
    'Patagonia': ['.patagonia/patagonia_men_links.txt', '.patagonia/patagonia_women_links.txt'],
    'Ecoalf': ['.ecoalf/ecoalf_men_links.txt', '.ecoalf/ecoalf_women_links.txt'],
    'Armedangels': ['.armedangels/armedangels_men_links.txt', '.armedangels/armedangels_women_links.txt']
}

def compter_lignes(nom_fichier):
    try:
        with open(nom_fichier, 'r') as f:
            return len(f.readlines())
    except:
        return 0

stats_txt = []
for marque, liste in fichiers_txt.items():
    h, f = compter_lignes(liste[0]), compter_lignes(liste[1])
    stats_txt.append({'Marques': marque, 'Hommes': h, 'Femmes': f, 'Total': h + f})

df_initial = pd.DataFrame(stats_txt)
df_initial.to_csv('stats_collecte_initiale.csv', index=False)

# Graphique 1 : Collecte Initiale
df_initial.plot(x='Marques', y=['Hommes', 'Femmes'], kind='bar', figsize=(8,5), color=["#227DAA", "#DD521C"])
plt.title("Collecte Initiale")
plt.ylabel("Nombre de Produits")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.savefig("graphique_initial.png")
plt.close()

# --- PARTIE 2 : COLLECTE APRÈS NETTOYAGE (EXCEL) ---

datasets_excel = {
    'Patagonia': '.patagonia/patagonia_dataset.xlsx',
    'Ecoalf': '.ecoalf/ecoalf_dataset.xlsx',
    'Armedangels': '.armedangels/armedangels_dataset.xlsx'
}

stats_cleaned = []

for marque, fichier in datasets_excel.items():
    if os.path.exists(fichier):
        try:
            temp_df = pd.read_excel(fichier)
            # Comptage basé sur la colonne source_file
            h = temp_df[temp_df['source_file'].str.contains('_men', case=False, na=False)].shape[0]
            f = temp_df[temp_df['source_file'].str.contains('women', case=False, na=False)].shape[0]
            stats_cleaned.append({'Marques': marque, 'Hommes': h, 'Femmes': f, 'Total': h + f})
        except:
            stats_cleaned.append({'Marques': marque, 'Hommes': 0, 'Femmes': 0, 'Total': 0})

df_cleaned = pd.DataFrame(stats_cleaned)
df_cleaned.to_csv('stats_collecte_nettoyee.csv', index=False)

# Graphique 2 : Collecte Après Nettoyage
df_cleaned.plot(x='Marques', y=['Hommes', 'Femmes'], kind='bar', figsize=(8,5), color=["#227DAA", "#DD521C"])
plt.title("Collecte après nettoyage")
plt.ylabel("Nombre de produits")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.savefig("graphique_nettoye.png")

# --- AFFICHAGE SYNTHÈSE ---
print("TABLEAU 1 : COLLECTE INITIALE")
print(df_initial)
print("\nNombre total de produits après la collecte :", df_initial['Total'].sum())

print("\nTABLEAU 2 : COLLECTE APRÈS NETTOYAGE")
print(df_cleaned)
print("\nNombre total de produits après nettoyage :", df_cleaned['Total'].sum())
import pandas as pd
import matplotlib.pyplot as plt

# 1. Liste  des fichiers
fichiers = {
    'Patagonia': ['Patagonia/mens_product_links.txt', 'Patagonia/womens_product_links.txt'],
    'Ecoalf': ['Ecoalf/ecoalf_men_links.txt', 'Ecoalf/ecoalf_women_links.txt'],
    'Armedangels': ['Armedangels/armedangels_men_links.txt', 'Armedangels/armedangels_women_links.txt']
}

# 2. Fonction de comptage 
def compter_lignes(nom_fichier):
    try:
        with open(nom_fichier, 'r') as f:
            return len(f.readlines())
    except:
        return 0 # Si le fichier n'existe pas encore

# 3. Création du tableau de stats
stats = []

for marque, liste_fichiers in fichiers.items():
    hommes = compter_lignes(liste_fichiers[0])
    femmes = compter_lignes(liste_fichiers[1])
    
    # On ajoute une ligne au tableau pour chaque marque
    stats.append({
        'Marques': marque,
        'Hommes': hommes,
        'Femmes': femmes,
        'Total': hommes + femmes
    })

# On transforme ça en DataFrame Pandas
df = pd.DataFrame(stats)

# Affichage des résultats dans la console
print("Résumé de la collecte :")
print(df)
print("\nNombre total de produits collectés :", df['Total'].sum())

# 4. Export fichier csv
df.to_csv('stats_corpus.csv', index=False)
print("Fichier stats_corpus.csv créé avec succès.")

# 5. Graphique 
df.plot(x='Marques', y=['Hommes', 'Femmes'], kind='bar', figsize=(10,6), color=["#227DAA", "#DD521C"])

plt.title("Nombre de produits par marque et genre")
plt.ylabel("Nombre de produits")
plt.xticks(rotation=0) # Pour garder les noms des marques horizontaux
plt.grid(axis='y', linestyle='--')

# Sauvegarde de l'image 
plt.savefig("graphique_stats.png")


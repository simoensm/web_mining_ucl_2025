import pandas as pd

df1 = pd.read_excel('.patagonia/token_frequencies_patagonia_unigram.xlsx') # Produits
df2 = pd.read_excel('.patagonia/report_frequencies_patagonia_unigram.xlsx') # Rapport

total_mots_doc1 = df1['frequency'].sum() #somme des fréquences du doc 1
total_mots_doc2 = df2['frequency'].sum() #somme des fréquences du doc 2
#On fait ceci pour éviter le biais de la taille des documents
df1['freq_norm_1'] = df1['frequency'] / total_mots_doc1 #fréquence normalisée du doc 1
df2['freq_norm_2'] = df2['frequency'] / total_mots_doc2 #fréquence normalisée du doc 2

comparaison = pd.merge(
    df1[['token', 'freq_norm_1']], 
    df2[['token', 'freq_norm_2']], 
    on='token', 
    how='outer'
) #fusion des deux tableaux sur la colonne 'token'

comparaison = comparaison.fillna(0) #remplace les valeurs manquantes par 0

comparaison['part_commune'] = comparaison[['freq_norm_1', 'freq_norm_2']].min(axis=1) #partie commune des fréquences normalisées

score_similarite = comparaison['part_commune'].sum() #somme des parties communes pour obtenir le score de similarité global

print(f"Score de similarité global : {score_similarite:.2%}") #affiche le score de similarité en pourcentage

print("\nTop 10 des mots qui rendent les documents similaires :") #affiche les 10 mots avec le plus grand score de similarité

print(comparaison.sort_values(by='part_commune', ascending=False).head(10)) #affiche les 10 premières lignes du tableau trié de façon décroissante
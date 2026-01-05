import pandas as pd

df1 = pd.read_excel('.patagonia/token_frequencies_patagonia_unigram.xlsx') # Produits
df2 = pd.read_excel('.patagonia/frequencies_patagonia_unigram.xlsx') # Rapport

total_mots_doc1 = df1['frequency'].sum()
total_mots_doc2 = df2['frequency'].sum()

df1['freq_norm_1'] = df1['frequency'] / total_mots_doc1
df2['freq_norm_2'] = df2['frequency'] / total_mots_doc2

comparaison = pd.merge(
    df1[['token', 'freq_norm_1']], 
    df2[['token', 'freq_norm_2']], 
    on='token', 
    how='outer'
)

comparaison = comparaison.fillna(0)

comparaison['part_commune'] = comparaison[['freq_norm_1', 'freq_norm_2']].min(axis=1)

score_similarite = comparaison['part_commune'].sum()

print(f"Score de similarité global : {score_similarite:.2%}")

print("\nTop 10 des mots qui rendent les documents similaires :")

print(comparaison.sort_values(by='part_commune', ascending=False).head(10))
import numpy as np
import pandas as pd

nom_fichier = '.patagonia/similarity_matrix_unigram.xlsx'
df = pd.read_excel(nom_fichier, index_col=0, engine='openpyxl')

base_matrix = df.values

print(base_matrix)
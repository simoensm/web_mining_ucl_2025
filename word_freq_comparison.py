import pandas as pd
from collections import Counter
from itertools import combinations

def get_word_frequencies(file_path):
    """Lit un fichier Excel et retourne les fréquences relatives des mots."""
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Erreur de lecture {file_path}: {e}")
        return {}, 0

    # Convertir tout le contenu en chaînes de caractères et aplatir
    all_text_values = df.astype(str).values.flatten()
    
    # Créer une liste de mots (minuscules)
    all_words = " ".join(all_text_values).lower().split()
    
    total_count = len(all_words)
    if total_count == 0: return {}, 0

    counts = Counter(all_words)
    # Calcul des fréquences relatives (ex: 'le' = 0.05)
    relative_freqs = {word: count / total_count for word, count in counts.items()}
    
    return relative_freqs, total_count

def calculate_overlap(freqs_a, freqs_b):
    """Calcule le score de similarité (Histogram Intersection) entre deux dicts de fréquences."""
    common_vocab = set(freqs_a.keys()) & set(freqs_b.keys())
    return sum(min(freqs_a[word], freqs_b[word]) for word in common_vocab)

def compare_multiple_documents(file_list):
    print(f"=== ANALYSE DE {len(file_list)} FICHIERS ===\n")
    
    # 1. Chargement des données (pour ne pas relire les fichiers plusieurs fois)
    docs_data = {}
    for file_path in file_list:
        print(f"Traitement de : {file_path}...")
        freqs, total = get_word_frequencies(file_path)
        if freqs:
            docs_data[file_path] = {'freqs': freqs, 'total': total}
        else:
            print(f"⚠️ Attention: Pas de données pour {file_path}")

    print("-" * 50)
    
    # 2. Comparaisons par Paires (Pairwise Comparison)
    # Cela génère toutes les combinaisons possibles : (A,B), (A,C), (B,C)
    for file_1, file_2 in combinations(docs_data.keys(), 2):
        freqs_1 = docs_data[file_1]['freqs']
        freqs_2 = docs_data[file_2]['freqs']
        
        score = calculate_overlap(freqs_1, freqs_2)
        
        print(f"COMPARAISON: {file_1.split('/')[-1]} <--> {file_2.split('/')[-1]}")
        print(f"   -> Frequency Overlap: {score * 100:.2f}%")
        print("-" * 30)

    # 3. (Optionnel) Chevauchement global aux 3 fichiers
    if len(docs_data) == 3:
        files = list(docs_data.keys())
        f1, f2, f3 = [docs_data[f]['freqs'] for f in files]
        
        # Vocabulaire commun aux trois
        common_all = set(f1.keys()) & set(f2.keys()) & set(f3.keys())
        
        # On prend le minimum des trois fréquences pour chaque mot
        global_score = sum(min(f1[w], f2[w], f3[w]) for w in common_all)
        
        print(f"=== CHEVAUCHEMENT GLOBAL (Commun aux 3) ===")
        print(f"   -> Global Overlap: {global_score * 100:.2f}%")

# --- Exécution ---

files_to_compare = [
    'word_frequencies_products_patagonia.xlsx',
    'word_frequencies_products_ecoalf.xlsx',
    'word_frequencies_products_northface.xlsx' # Exemple de 3ème fichier
]

compare_multiple_documents(files_to_compare)
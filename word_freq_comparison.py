import pandas as pd
from collections import Counter

def get_word_frequencies(file_path):
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}, 0

    all_text_values = df.astype(str).values.flatten()
    
    all_words = " ".join(all_text_values).lower().split()
    
    total_count = len(all_words)
    if total_count == 0: return {}, 0

    counts = Counter(all_words)
    relative_freqs = {word: count / total_count for word, count in counts.items()}
    
    return relative_freqs, total_count

def compare_documents(file_1, file_2):
    print(f"--- Comparing {file_1} vs {file_2} ---")
    
    freqs_1, total_1 = get_word_frequencies(file_1)
    freqs_2, total_2 = get_word_frequencies(file_2)
    
    if not freqs_1 or not freqs_2:
        return

    common_vocab = set(freqs_1.keys()) & set(freqs_2.keys())
    
    similarity_score = sum(min(freqs_1[word], freqs_2[word]) for word in common_vocab)
    
    print(f"Total Words in File 1: {total_1}")
    print(f"Total Words in File 2: {total_2}")
    print(f"Shared Vocabulary Count: {len(common_vocab)}")
    print(f"-"*30)
    print(f"FREQUENCY OVERLAP: {similarity_score * 100:.2f}%")

file_1 = 'Patagonia/word_frequencies_products_patagonia.xlsx'
file_2 = 'Ecoalf/word_frequencies_products_ecoalf.xlsx'

compare_documents(file_1, file_2)
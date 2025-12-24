import pandas as pd
import nltk
import string
import math
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 1. Téléchargement des ressources
nltk.download('punkt')
nltk.download('stopwords')

# 2. Chargement Excel
file_path = r"C:\Users\mathi\Downloads\loom_bdd.xlsx"

try:
    df = pd.read_excel(file_path)
except Exception as e:
    print(f"Erreur : {e}")
    exit()

if 'full_content' not in df.columns:
    print("Erreur : Colonne 'full_content' manquante.")
    exit()

df['full_content'] = df['full_content'].fillna("")

# 3. Configuration des Stopwords (Mots vides)
stop_words = set(stopwords.words('french'))

# --- MISE A JOUR DES STOPWORDS ---
# Liste complémentaire de mots "bruit" fréquents en français
new_stopwords = [
    "à", "au", "aux", "avec", "ce", "ces", "cette", "cet", "c", "c’est", "ça",
    "de", "des", "du", "d", "dans", "en", "et", "est", "être", "été", "sont",
    "elle", "elles", "il", "ils", "je", "tu", "nous", "vous", "on",
    "ne", "pas", "plus", "moins", "mais", "ou", "où", "donc", "alors", "aussi",
    "car", "ni", "que", "qui", "quoi", "dont", "quand", "comment", "comme",
    "pour", "par", "sur", "sous", "entre", "chez", "vers", "sans", "contre",
    "avant", "après", "pendant", "depuis", "jusqu’à", "afin", "ainsi",
    "très", "trop", "peu", "encore", "déjà", "toujours", "jamais", "souvent", "parfois",
    "ici", "là", "y",
    "leur", "leurs", "lui", "me", "te", "se",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "notre", "nos", "votre", "vos",
    "un", "une", "le", "la", "les", "l", "d", "n", "s", "qu", "j", "m", "t",
    "avoir", "fait", "faire", "dire", "donner", "mettre", "prendre", "pouvoir",
    "peut", "peuvent", "devoir", "doit", "doivent", "falloir", "faut",
    "aller", "venir", "voir", "savoir", "sembler", "rester", "devenir",
    "utiliser", "choisir", "porter", "fabriquer", "vendre"
]
stop_words.update(new_stopwords)

def preprocess_text(text):
    """
    Tokenization, Suppression des chiffres et Nettoyage
    """
    if not isinstance(text, str):
        text = str(text)

    tokens = word_tokenize(text.lower(), language='french')
    
    cleaned_tokens = []
    for token in tokens:
        # 1. Vérifie si c'est un stopword
        if token in stop_words:
            continue
            
        # 2. Vérifie si le mot contient UNIQUEMENT des lettres
        # .isalpha() renvoie False pour "2024", "10kg", "1er", "!" 
        # Cela supprime donc les chiffres et la ponctuation
        if token.isalpha():
            
            # (Optionnel) On peut aussi filtrer les mots trop courts (1 ou 2 lettres)
            if len(token) > 2:
                cleaned_tokens.append(token)
            
    return cleaned_tokens

# Application
print("Traitement en cours (suppression des chiffres et stopwords étendus)...")
df['tokens'] = df['full_content'].apply(preprocess_text)

# 4. Calcul TF-IDF
vocabulaire = sorted(list(set(token for sublist in df['tokens'] for token in sublist)))
N_documents = len(df)
idf_scores = {}

# Calcul IDF
for term in vocabulaire:
    d_i = sum(1 for tokens in df['tokens'] if term in tokens)
    idf_scores[term] = math.log(N_documents / (d_i + 1e-5)) if d_i > 0 else 0

# Calcul TF-IDF
tf_idf_matrix = []
for tokens in df['tokens']:
    doc_counts = Counter(tokens)
    max_occurrence = max(doc_counts.values()) if doc_counts else 1
    row = {}
    for term, count in doc_counts.items():
        row[term] = (count / max_occurrence) * idf_scores[term]
    tf_idf_matrix.append(row)

df_tfidf = pd.DataFrame(tf_idf_matrix).fillna(0)

# 5. Résultats
print("\n--- Top 10 mots les plus spécifiques (IDF) ---")
print(pd.Series(idf_scores).sort_values(ascending=False).head(10))

print("\n--- Top 10 mots les plus fréquents/communs (IDF faible) ---")
print(pd.Series(idf_scores).sort_values(ascending=True).head(10))

print(f"\n--- Aperçu Matrice TF-IDF ({df_tfidf.shape[1]} mots trouvés) ---")
print(df_tfidf.head())
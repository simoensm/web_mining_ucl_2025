import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# --- INITIALISATION DES RESSOURCES ---
try:
    nltk.data.find('tokenizers/punkt') 
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except LookupError:
    nltk.download('punkt')

class TextMiner:
    def __init__(self, file_path, column_name='description_clean', ngram_type='unigram', normalization='lemmatization'): 
        self.file_path = file_path
        self.column_name = column_name
        self.ngram_type = ngram_type
        self.normalization = normalization
        
        # Choix de l'outil de normalisation
        if self.normalization == 'stemming':
            self.stemmer = PorterStemmer()
        else:
            self.lemmatizer = WordNetLemmatizer()
            
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update({
            'intro', 'details', 'specs', 'features', 'materials','material', 'care', 'instructions', 'weight', 
            'country', 'origin', 'made', 'factory', 'certified','machine', 'wash', 'warm', 'cold', 'bleach', 'dry',
            'tumble', 'iron','oz', 'g', 'lbs', 'premium', 'product', 'regular', 'fit', 'size', 'color', 'cool', 
            'intentionally', 'saying', "finish", "visit", "guide"
        })
        
        self.tfidf_matrix = None
        self.df = None
        self.feature_names = None

    def preprocess(self, text):
        tokens = word_tokenize(str(text))
        cleaned = []
        for t in tokens:
            if t.lower() not in self.stop_words and len(t) > 2:
                if self.normalization == 'stemming': 
                    word_final = self.stemmer.stem(t)
                else:
                    word_final = self.lemmatizer.lemmatize(t)
                cleaned.append(word_final)
        return " ".join(cleaned)

    def load_and_prepare(self):
        # Lecture du fichier Excel
        self.df = pd.read_excel(self.file_path).dropna(subset=[self.column_name]).reset_index(drop=True)
        self.df['processed_text'] = self.df[self.column_name].apply(self.preprocess)
        
        # Configuration des N-grammes
        n_range = (1, 1)
        if self.ngram_type == 'bigram': n_range = (2, 2)
        elif self.ngram_type == 'trigram': n_range = (3, 3)

        cv = CountVectorizer(ngram_range=n_range, min_df=3, max_df=0.80)
        raw_counts = cv.fit_transform(self.df['processed_text']).toarray()
        self.feature_names = cv.get_feature_names_out()

        # Calcul TF-IDF 
        max_counts_per_doc = raw_counts.max(axis=1)
        max_counts_per_doc[max_counts_per_doc == 0] = 1 
        tf = raw_counts / max_counts_per_doc[:, np.newaxis]
        
        doc_freq = (raw_counts > 0).sum(axis=0)
        doc_freq[doc_freq == 0] = 1
        idf = np.log(len(self.df) / doc_freq)

        self.tfidf_matrix = normalize(tf * idf, norm='l2', axis=1)
        print(f">> Prétraitement terminé ({len(self.df)} lignes).")


if __name__ == "__main__":
    # Chemin vers le fichier 
    path = ".ecoalf/ecoalf_dataset.xlsx" 
    
    # 1. Menu interactif pour la normalisation 
    print("Choisissez la méthode de normalisation :")
    print("1. Lemmatisation")
    print("2. Racinisation")
    choix_norm = input(">> ")
    methode = 'lemmatization' if choix_norm == '1' else 'stemming'
    
    # 2. Menu interactif pour les N-grammes
    print("\nChoisissez le type de N-gramme :")
    print("1. Unigramme ")
    print("2. Bigramme ")
    choix_ng = input(">> ")
    type_ng = 'unigram' if choix_ng == '1' else 'bigram'

    # Initialisation
    miner = TextMiner(path, ngram_type=type_ng, normalization=methode)
    miner.load_and_prepare()

    print(f"\n--- ANALYSE : {methode.upper()} + {type_ng.upper()} ---")
    print(f"{'K':<5} | {'Silhouette Score':<18} | {'Inertia':<10}")
    print("-" * 40)

    for k in range(2, 15):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = km.fit_predict(miner.tfidf_matrix)
        
        # Calcul de la qualité du clustering
        score = silhouette_score(miner.tfidf_matrix, labels)
        inertia = km.inertia_
        
        print(f"{k:<5} | {score:<18.4f} | {inertia:<10.2f}")

    print("\n>> Test terminé !")
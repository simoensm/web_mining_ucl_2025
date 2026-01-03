import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print(">> Téléchargement des ressources NLTK...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class TextMinerTXT:
    def __init__(self, file_path, ngram_type='unigram', normalization='lemmatization'):
        self.file_path = file_path
        self.ngram_type = ngram_type
        self.normalization = normalization
        
        if self.normalization == 'stemming':
            self.stemmer = PorterStemmer()
        elif self.normalization == 'lemmatization':
            self.lemmatizer = WordNetLemmatizer()
            
        self.stop_words = set(stopwords.words('english'))
        
        noise_words = {
            'intro', 'details', 'specs', 'features', 'materials', 'care', 'instructions',
            'weight', 'country', 'origin', 'made', 'factory', 'certified',
            'machine', 'wash', 'warm', 'cold', 'bleach', 'dry', 'tumble', 'iron',
            'oz', 'g', 'lbs', 'premium', 'product', 'regular', 'fit', 'year', 'patagonia', 'report', 'work', 'worker', 'unusualproductgrants'
        }
        self.stop_words.update(noise_words)
        
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None
        self.documents = []

    def clean_boilerplate(self, text):
        """Nettoyage des structures récurrentes et caractères spéciaux."""
        text = str(text).lower()
        text = re.sub(r'\[.*?\]', ' ', text) # Retire texte entre crochets
        text = re.sub(r'care instructions.*', '', text, flags=re.DOTALL)
        text = re.sub(r'[^\w\s]', '', text) # Retire ponctuation
        text = re.sub(r'\d+', '', text)     # Retire chiffres
        return text

    def preprocess(self, text):
        """Tokenization et Normalisation (Lemmatization/Stemming)."""
        # Étape 1 : Nettoyage de base
        text = self.clean_boilerplate(text)
        
        # Étape 2 : Tokenization
        tokens = word_tokenize(text)
        
        cleaned = []
        for t in tokens:
            if t not in self.stop_words and len(t) > 2:
                word_final = t
                if self.normalization == 'stemming':
                    word_final = self.stemmer.stem(t)
                elif self.normalization == 'lemmatization':
                    word_final = self.lemmatizer.lemmatize(t)
                
                cleaned.append(word_final)
                
        return " ".join(cleaned)
    
    def get_word_frequencies(self):
        """
        Calculates the raw frequency (count) of each term across all text segments.
        """
        from sklearn.feature_extraction.text import CountVectorizer
        import numpy as np
        
        print(f"\n--- CALCULATING RAW FREQUENCIES ({self.ngram_type}) ---")
        
        if not self.documents:
            print("Error: No processed documents found. Run load_and_process() first.")
            return None

        # 1. Define N-gram range to match your settings
        if self.ngram_type == 'trigram': n_range = (3, 3)
        elif self.ngram_type == 'bigram': n_range = (2, 2)
        else: n_range = (1, 1)

        # 2. Count using CountVectorizer
        # We use self.documents because they are already cleaned and preprocessed by load_and_process
        cv = CountVectorizer(ngram_range=n_range, min_df=1) 
        X = cv.fit_transform(self.documents)
        
        # 3. Sum up counts
        total_counts = np.asarray(X.sum(axis=0)).flatten()
        vocab = cv.get_feature_names_out()
        
        # 4. Create sorted DataFrame
        freq_df = pd.DataFrame({'Term': vocab, 'Frequency': total_counts})
        freq_df = freq_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
        
        return freq_df

    def load_and_process(self):
        print(f"\n--- CHARGEMENT & TRAITEMENT ({self.ngram_type.upper()}) ---")
        
        try:
            # MODIFICATION: Read the whole file as one string first
            with open(self.file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
        except FileNotFoundError:
            print(f"Erreur : Le fichier '{self.file_path}' est introuvable.")
            return False
            
        # MODIFICATION: Split into words and create "chunks" to simulate documents
        # If we don't do this, TF-IDF fails because it sees only 1 document.
        all_words = full_text.split()
        
        # We create chunks of 100 words (adjustable) to act as "paragraphs"
        chunk_size = 100 
        raw_docs = [' '.join(all_words[i:i + chunk_size]) for i in range(0, len(all_words), chunk_size)]
        
        print(f"Texte découpé en {len(raw_docs)} segments (chunks) pour l'analyse.")
        
        print("Pré-traitement du texte (Nettoyage + Normalisation)...")
        self.documents = [self.preprocess(doc) for doc in raw_docs]
        
        # Filter empty docs
        self.documents = [doc for doc in self.documents if len(doc) > 0]

        if len(self.documents) < 2:
            print("Attention: Pas assez de segments pour min_df=2. Ajustement à min_df=1.")
            min_df_val = 1
        else:
            min_df_val = 2

        print(f"Vectorisation (TF-IDF)...")
        
        # Configuration N-Grams
        if self.ngram_type == 'trigram': n_range = (3, 3)
        elif self.ngram_type == 'bigram': n_range = (2, 2)
        else: n_range = (1, 1)
        
        # MODIFICATION: Dynamic min_df based on chunk count
        self.vectorizer = TfidfVectorizer(ngram_range=n_range, min_df=min_df_val, max_df=0.90)
        
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        except ValueError as e:
            print(f"Erreur de vectorisation : {e}")
            return False

        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"Analyse terminée : {self.tfidf_matrix.shape[1]} termes uniques identifiés.")
        return True

    def generate_statistics_and_cloud(self):
        if self.tfidf_matrix is None:
            return

        print(f"\n--- RÉSULTATS & VISUALISATION ---")
        
        # Conversion de la matrice creuse en DataFrame pour manipuler les scores
        tfidf_df = pd.DataFrame(self.tfidf_matrix.toarray(), columns=self.feature_names)
        
        # Calcul de la somme des scores TF-IDF pour chaque mot
        word_weights = tfidf_df.sum(axis=0).sort_values(ascending=False)
        
        print("\n>> Top 20 des termes les plus importants (Score TF-IDF cumulé) :")
        print(word_weights)#.head(20))

        # Génération du Nuage de Mots
        weights_dict = word_weights.to_dict()
        
        if not weights_dict:
            print("Pas assez de données pour générer un nuage de mots.")
            return

        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            max_words=100,
            collocations=False
        ).generate_from_frequencies(weights_dict)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Analyse du fichier : {self.file_path} ({self.ngram_type})")
        plt.show()

# --- EXEMPLE D'UTILISATION ---

# 1. Configuration
# Assurez-vous d'avoir un fichier 'report_ecoalf_2022.txt' dans le même dossier
miner = TextMinerTXT(
    file_path='report_ecoalf_2022.txt',  
    ngram_type='unigram',       # 'unigram' (mots seuls) ou 'bigram' (paires de mots)
    normalization='lemmatization' # 'lemmatization' ou 'stemming'
)

# 2. Exécution
if miner.load_and_process():
    miner.generate_statistics_and_cloud()
    
    # --- NEW: EXPORT FREQUENCY LIST ---
    df_frequencies = miner.get_word_frequencies()
    
    if df_frequencies is not None:
        print("\n>> Top 20 Most Frequent Terms:")
        print(df_frequencies.head(20))
        
        # Save to Excel
        output_filename = "word_frequencies_report.xlsx"
        df_frequencies.to_excel(output_filename, index=False)
        print(f">> Saved full frequency list to '{output_filename}'")

    #word_weights
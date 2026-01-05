import pandas as pd # Pour la manipulation des données et l'export Excel
import re # Pour le Regex
import nltk # Pour le traitement du langage naturel
import numpy as np # Pour les calculs numériques
import os # Pour la gestion des fichiers
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer 

# S'assure que les ressources NLTK nécessaires sont téléchargées
try:
    nltk.data.find('tokenizers/punkt') 
    nltk.data.find('corpora/wordnet')
except LookupError:
    print(">> NLTK...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class TextMinerFrequency:
    def __init__(self, file_path, ngram_type='unigram', normalization='lemmatization'): 
        self.file_path = file_path
        self.ngram_type = ngram_type
        self.normalization = normalization
        
        # Choix entre Lemmantisation et Racinisation
        if self.normalization == 'stemming':
            self.stemmer = PorterStemmer()
        elif self.normalization == 'lemmatization':
            self.lemmatizer = WordNetLemmatizer()
            
        self.stop_words = set(stopwords.words('english'))

        # Mots à ignorer (bruit)
        noise_words = {
            'intro', 'details', 'specs', 'features', 'materials', 'care', 'instructions',
            'weight', 'country', 'origin', 'made', 'factory', 'certified',
            'machine', 'wash', 'warm', 'cold', 'bleach', 'dry', 'tumble', 'iron',
            'oz', 'g', 'lbs', 'premium', 'product', 'regular', 'fit', 'size', 'color',
            'cool', 'intentionally', 'saying', 'finish', 'visit', 'guide',
            'year', 'report', 'work', 'worker', 'page', 'chapter'
        }
        self.stop_words.update(noise_words)
        
        self.full_cleaned_text = None # Variable pour stocker tout le texte nettoyé

    def preprocess(self, text):
        """Nettoyage du texte : minuscules, suppression ponctuation/chiffres, stop-words, lemmatisation"""
        print("   > Cleaning text (Regex, Stopwords, Normalization)...")
        text = str(text).lower()
        text = re.sub(r'\[.*?\]', ' ', text)  # Retire texte entre crochets
        text = re.sub(r'[^\w\s]', '', text)   # Enlève la ponctuation
        text = re.sub(r'\d+', '', text)       # Enlève les chiffres
        tokens = word_tokenize(text)
        
        cleaned = []
        for t in tokens:
            if t not in self.stop_words and len(t) > 2:
                if self.normalization == 'stemming': 
                    word_final = self.stemmer.stem(t)
                elif self.normalization == 'lemmatization':
                    word_final = self.lemmatizer.lemmatize(t)
                else:
                    word_final = t
                cleaned.append(word_final)

        return " ".join(cleaned)

    def load_and_process(self):
        print(f"\n1. Loading & Pre-Process ({self.normalization.upper()})") 
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
        except FileNotFoundError:
            print(f"Error: Unable to find {self.file_path}")
            return False
        
        # On traite tout le texte d'un coup
        self.full_cleaned_text = self.preprocess(full_text)
        
        if not self.full_cleaned_text.strip():
            print("   > Error: The text is empty after cleaning.")
            return False
            
        print(f"   > Text loaded and cleaned successfully.")
        return True

    def export_token_frequencies(self, output_file='token_frequencies.xlsx'):
        print(f"\n2. Calculation & Export Token Frequencies")
        
        if self.full_cleaned_text is None:
            print("   > Error: Data not loaded.")
            return None
        
        # Définir la plage de n-grammes
        if self.ngram_type == 'trigram': n_range = (3, 3)
        elif self.ngram_type == 'bigram': n_range = (2, 2)
        else: n_range = (1, 1)
        
        print(f"   > Counting {self.ngram_type}s...")
        
        # On passe le texte dans une liste [texte] car CountVectorizer attend une liste de documents
        cv = CountVectorizer(ngram_range=n_range, min_df=1)
        X = cv.fit_transform([self.full_cleaned_text])
        
        # Calcul des fréquences
        total_counts = np.asarray(X.sum(axis=0)).flatten()
        tokens = cv.get_feature_names_out()
        
        # Création du DataFrame final (Uniquement Token et Frequency)
        freq_df = pd.DataFrame({
            'token': tokens,
            'frequency': total_counts
        })
        
        # Tri par fréquence décroissante
        freq_df = freq_df.sort_values(by='frequency', ascending=False).reset_index(drop=True)
        
        try:
            freq_df.to_excel(output_file, index=False)
            print(f"   > DONE! File saved: {output_file}")
            print(f"   > Total unique tokens: {len(freq_df)}")
            print(f"   > Top token: '{freq_df.iloc[0]['token']}' ({freq_df.iloc[0]['frequency']})")
        except Exception as e:
            print(f"   > Error exporting file: {e}")
        
        return freq_df

if __name__ == "__main__":
    
    print("\n--- TEXT MINER : SIMPLE FREQUENCY ---")
    
    # 1. Configuration
    print("\nCONFIGURATION")
    print("1. Lemmatization (Recommended)")
    print("2. Stemming")
    norm_choice = input("Choice (1/2) : ").strip()
    norm_mode = 'lemmatization' if norm_choice == '1' else 'stemming'

    print("\n1. Unigram")
    print("2. Bigram")
    print("3. Trigram")
    ngram_choice = input("Choice (1/2/3) : ").strip()
    ngram_map = {'1': 'unigram', '2': 'bigram', '3': 'trigram'}
    ngram_mode = ngram_map.get(ngram_choice, 'unigram')

    # 2. Exécution
    file_to_analyze = '.patagonia/patagonia-progress-report-2025.txt' # À MODIFIER

    miner = TextMinerFrequency(
        file_to_analyze, 
        ngram_type=ngram_mode, 
        normalization=norm_mode
    )

    if miner.load_and_process():
        output_filename = f"report_frequencies_patagonia_{ngram_mode}.xlsx"
        miner.export_token_frequencies(output_file=output_filename)
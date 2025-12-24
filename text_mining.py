import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

class TextMiner:
    def __init__(self, file_path, column_name='description', id_column=None, ngram_type='unigram', use_aggressive_stemming=True):
        self.file_path = file_path
        self.column_name = column_name
        self.id_column = id_column 
        self.ngram_type = ngram_type
        
        if use_aggressive_stemming:
            self.stemmer = LancasterStemmer()
            print(">> Initialized with Lancaster Stemmer (Aggressive)")
        else:
            self.stemmer = PorterStemmer()
            print(">> Initialized with Porter Stemmer (Standard)")
            
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.df = None

    def preprocess(self, text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        
        cleaned = []
        for t in tokens:
            if t not in self.stop_words and len(t) > 2:
                cleaned.append(self.stemmer.stem(t))
                
        return " ".join(cleaned)

    def run_analysis(self):
        print(f"\n--- LOADING & PROCESSING ({self.ngram_type}) ---")
        try:
            self.df = pd.read_excel(self.file_path)
        except FileNotFoundError:
            print("Error: File not found.")
            return
            
        self.df = self.df.dropna(subset=[self.column_name]).reset_index(drop=True)
        print(f"Loaded {len(self.df)} documents.")
        
        print("Preprocessing text...")
        processed_corpus = self.df[self.column_name].apply(self.preprocess).tolist()
        
        print("Vectorizing (TF-IDF)...")
        if self.ngram_type == 'bigram':
            n_range = (2, 2)
        else:
            n_range = (1, 1)
            
        self.vectorizer = TfidfVectorizer(ngram_range=n_range, min_df=2)
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_corpus)
        
        print("Calculating Cosine Similarity...")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print("Analysis Complete.")
        return self

    def export_to_gephi(self, name_column, threshold=0.2, nodes_file='gephi_nodes.csv', edges_file='gephi_edges.csv'):
        """
        name_column: The exact name of the Excel column containing Product Names (e.g., 'Product Name')
        """
        print(f"\n--- EXPORTING TO GEPHI (Threshold: {threshold}) ---")
        
        ids = [str(i) for i in range(len(self.df))]
        
        try:
            labels = self.df[name_column].astype(str).tolist()
        except KeyError:
            print(f"Error: Column '{name_column}' not found in Excel. Using IDs as labels instead.")
            labels = ids


        nodes_df = pd.DataFrame({
            'Id': ids,
            'Label': labels 
        })
        nodes_df.to_csv(nodes_file, index=False)
        print(f">> Saved Nodes to {nodes_file}")

 
        upper_matrix = np.triu(self.cosine_sim, k=1)
        
        rows, cols = np.where(upper_matrix > threshold)
        weights = upper_matrix[rows, cols]
        
        source_ids = [ids[r] for r in rows]
        target_ids = [ids[c] for c in cols]
        
        edges_df = pd.DataFrame({
            'Source': source_ids,
            'Target': target_ids,
            'Weight': weights,
            'Type': 'Undirected'
        })
        
        edges_df.to_csv(edges_file, index=False)
        print(f">> Saved Edges to {edges_file}")


miner = TextMiner('patagonia_women_final.xlsx', column_name='Description', id_column=None)
miner.run_analysis()

miner.export_to_gephi(name_column = 'Name', threshold=0.2)
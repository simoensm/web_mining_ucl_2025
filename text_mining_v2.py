import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Gestion des téléchargements NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print(">> Téléchargement des ressources NLTK...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class TextMiner:
    def __init__(self, file_path, column_name='description', id_column=None, ngram_type='unigram', normalization='lemmatization'):
        self.file_path = file_path
        self.column_name = column_name
        self.id_column = id_column 
        self.ngram_type = ngram_type
        self.normalization = normalization
        
        if self.normalization == 'stemming':
            self.stemmer = PorterStemmer()
        elif self.normalization == 'lemmatization':
            self.lemmatizer = WordNetLemmatizer()
            
        # 1. STOPWORDS AMÉLIORÉS
        self.stop_words = set(stopwords.words('english'))
        # Ajout de mots "bruit" qui polluent les clusters sans donner d'info sur le produit
        noise_words = {
            'intro', 'details', 'specs', 'features', 'materials', 'care', 'instructions',
            'weight', 'country', 'origin', 'made', 'factory', 'certified',
            'machine', 'wash', 'warm', 'cold', 'bleach', 'dry', 'tumble', 'iron',
            'oz', 'g', 'lbs', 'premium', 'product', 'regular', 'fit', 
        }
        self.stop_words.update(noise_words)
        
        self.vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.df = None
        self.feature_names = None
        self.kmeans_model = None

    def clean_boilerplate(self, text):
        """
        Supprime les phrases récurrentes (Lavage, Intro, etc.) qui faussent le clustering.
        """
        text = str(text).lower()
        # Supprime tout ce qui est entre crochets [intro]
        text = re.sub(r'\[.*?\]', ' ', text)
        # Supprime les instructions de lavage (souvent à la fin)
        text = re.sub(r'care instructions.*', '', text, flags=re.DOTALL)
        # Supprime info poids et origine
        text = re.sub(r'country of origin.*', '', text)
        text = re.sub(r'weight\s*\d+.*', '', text)
        return text
    
    def get_word_frequencies(self):
        """
        Generates a simple dataframe with Words and their Global Count.
        """
        from sklearn.feature_extraction.text import CountVectorizer
        
        print(f"\n--- CALCULATING RAW FREQUENCIES ({self.ngram_type}) ---")
        
        # 1. Define N-gram range based on your class settings
        if self.ngram_type == 'trigram': n_range = (3, 3)
        elif self.ngram_type == 'bigram': n_range = (2, 2)
        else: n_range = (1, 1)

        # 2. Re-process the text to ensure we count the cleaned versions
        # (We apply the existing self.preprocess function)
        corpus = self.df[self.column_name].apply(self.preprocess)
        
        # 3. Count using CountVectorizer (Pure counting, no TF-IDF weighting)
        # We keep min_df/max_df generic here to capture most words
        cv = CountVectorizer(ngram_range=n_range, min_df=2) 
        X = cv.fit_transform(corpus)
        
        # 4. Sum up the counts per word
        total_counts = np.asarray(X.sum(axis=0)).flatten()
        vocab = cv.get_feature_names_out()
        
        # 5. Create DataFrame and Sort
        freq_df = pd.DataFrame({'Term': vocab, 'Frequency': total_counts})
        freq_df = freq_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
        
        return freq_df

    def preprocess(self, text):
        # Étape 1 : Nettoyage structurel
        text = self.clean_boilerplate(text)
        
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
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

    def run_analysis(self):
        print(f"\n--- LOADING & PROCESSING ({self.ngram_type.upper()}) ---")
        try:
            self.df = pd.read_excel(self.file_path)
        except FileNotFoundError:
            print("Error: File not found.")
            return None
            
        self.df = self.df.dropna(subset=[self.column_name]).reset_index(drop=True)
        print(f"Loaded {len(self.df)} documents.")
        
        print("Preprocessing text (Cleaning + Normalization)...")
        processed_corpus = self.df[self.column_name].apply(self.preprocess).tolist()
        
        print(f"Vectorizing (TF-IDF)...")
        if self.ngram_type == 'trigram': n_range = (3, 3)
        elif self.ngram_type == 'bigram': n_range = (2, 2)
        else: n_range = (1, 1)
        
        # 2. OPTIMISATION TF-IDF POUR LE CLUSTERING
        # max_df=0.85 : Ignore les mots qui apparaissent dans + de 85% des docs (trop communs)
        # min_df=5 : Ignore les mots qui apparaissent dans moins de 5 docs (bruit/typo)
        self.vectorizer = TfidfVectorizer(ngram_range=n_range, min_df=5, max_df=0.85)
        
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(processed_corpus)
        except ValueError:
            print("Erreur : Vocabulaire vide. Essayez de baisser min_df ou changer ngram_type.")
            return None

        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"Matrice créée : {self.tfidf_matrix.shape[1]} termes uniques.")
        
        print("Calculating Cosine Similarity...")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        return self

    # --- NOUVEAU : Méthode du Coude pour trouver le bon nombre de clusters ---
    def find_optimal_clusters(self, max_k=15):
        print(f"\n--- RECHERCHE DU NOMBRE OPTIMAL DE CLUSTERS (Elbow Method) ---")
        inertias = []
        K_range = range(1, max_k + 1)
        
        print("Calcul en cours...", end='')
        for k in K_range:
            print(f".", end='')
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            km.fit(self.tfidf_matrix)
            inertias.append(km.inertia_)
        print(" Terminé.")

        plt.figure(figsize=(8, 4))
        plt.plot(K_range, inertias, 'bx-')
        plt.xlabel('Nombre de Clusters (k)')
        plt.ylabel('Inertie (Distance intra-cluster)')
        plt.title('Méthode du Coude : Cherchez le point de cassure')
        plt.grid(True)
        plt.show()

    def perform_clustering(self, n_clusters=3):
        print(f"\n--- CLUSTERING (K-Means: {n_clusters} groupes) ---")
        self.kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=20, random_state=42)
        self.kmeans_model.fit(self.tfidf_matrix)
        
        self.df['Cluster_Label'] = self.kmeans_model.labels_
        print(">> Clustering terminé.")
        
        print("\n>> Top termes par cluster :")
        order_centroids = self.kmeans_model.cluster_centers_.argsort()[:, ::-1]
        
        for i in range(n_clusters):
            print(f"Cluster {i} : ", end='')
            top_terms = [self.feature_names[ind] for ind in order_centroids[i, :12]] 
            print(", ".join(top_terms))
            
        return self.df

    def visualize_clusters_pca(self):
        if 'Cluster_Label' not in self.df.columns:
            return

        print("\n--- VISUALISATION DES CLUSTERS (PCA) ---")
        pca = PCA(n_components=2, random_state=42)
        reduced_features = pca.fit_transform(self.tfidf_matrix.toarray())
        reduced_centers = pca.transform(self.kmeans_model.cluster_centers_)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                              c=self.df['Cluster_Label'], cmap='tab10', s=60, alpha=0.7)
        
        plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], 
                    marker='x', s=200, c='red', linewidths=3, label='Centres')

        plt.title(f"Carte des Produits (PCA - {self.ngram_type})")
        plt.colorbar(scatter, label='Cluster ID')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()

    def generate_statistics_and_cloud(self):
        print(f"\n--- STATISTIQUES & VISUALISATION ({self.ngram_type}) ---")
        tfidf_df = pd.DataFrame(self.tfidf_matrix.toarray(), columns=self.feature_names)
        
        word_weights = tfidf_df.sum(axis=0).sort_values(ascending=False)
        print("\n>> Top 20 des expressions les plus importantes :")
        print(word_weights.head(20))

        weights_dict = word_weights.to_dict()
        wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate_from_frequencies(weights_dict)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Nuage de mots ({self.ngram_type})")
        plt.show()

    def export_to_gephi(self, name_column, threshold=0.2, nodes_file='gephi_nodes.csv', edges_file='gephi_edges.csv'):
        print(f"\n--- EXPORTING TO GEPHI (Threshold: {threshold}) ---")
        
        ids = [str(i) for i in range(len(self.df))]
        
        try:
            labels = self.df[name_column].astype(str).tolist()
        except KeyError:
            print(f"Error: Column '{name_column}' not found. Using IDs.")
            labels = ids

        nodes_df = pd.DataFrame({'Id': ids, 'Label': labels})
        nodes_df.to_csv(nodes_file, index=False)
        print(f">> Saved Nodes to {nodes_file}")

        upper_matrix = np.triu(self.cosine_sim, k=1)
        rows, cols = np.where(upper_matrix > threshold)
        weights = upper_matrix[rows, cols]
        
        edges_df = pd.DataFrame({
            'Source': [ids[r] for r in rows],
            'Target': [ids[c] for c in cols],
            'Weight': weights,
            'Type': 'Undirected'
        })
        
        edges_df.to_csv(edges_file, index=False)
        print(f">> Saved Edges to {edges_file}")

# --- EXECUTION OPTIMISÉE ---

# --- EXECUTION COMPLETE ---

# 1. Initialisation
miner = TextMiner(
    'cleaned_all_products.xlsx', 
    column_name='description', 
    ngram_type='unigram',         
    normalization='lemmatization'
)

# 2. Lancement de l'analyse
if miner.run_analysis():

    df_frequencies = miner.get_word_frequencies()

    print("\n>> Top 20 Most Frequent Terms:")
    print(df_frequencies.head(20))

    df_frequencies.to_excel("word_frequencies_products.xlsx", index=False)
    print(">> Saved full frequency list to 'word_frequencies_products.xlsx'")
    
    # A. Recherche du nombre optimal de clusters (Méthode du coude)
    # (Vous pouvez commenter cette ligne une fois que vous avez trouvé votre chiffre)
    miner.find_optimal_clusters(max_k=15) 
    
    # B. Clustering (Mettez ici le chiffre trouvé grâce au graphique, ex: 6)
    miner.perform_clustering(n_clusters=10)
    
    # C. Visualisation des clusters (Graphique PCA)
    miner.visualize_clusters_pca()
    
    # D. EXPORT VERS GEPHI (C'est la ligne qu'il manquait)
    # Note : J'ai mis un seuil de 0.15. Si le fichier est vide, baissez à 0.10
    miner.export_to_gephi(name_column='name', threshold=0.2)
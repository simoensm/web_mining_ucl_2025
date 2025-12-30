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
from sklearn.cluster import KMeans             # <--- NOUVEAU
from sklearn.decomposition import PCA          # <--- NOUVEAU

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
        
        # Initialisation du normalisateur
        if self.normalization == 'stemming':
            self.stemmer = PorterStemmer()
            print(">> Mode: STEMMING (Racinisation)")
        elif self.normalization == 'lemmatization':
            self.lemmatizer = WordNetLemmatizer()
            print(">> Mode: LEMMATIZATION (Lemmatisation)")
        else:
            print(">> Mode: AUCUN (Mots complets)")
            
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.df = None
        self.feature_names = None
        self.kmeans_model = None # Pour stocker le modèle de clustering

    def preprocess(self, text):
        text = str(text).lower()
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
        
        print("Preprocessing text...")
        processed_corpus = self.df[self.column_name].apply(self.preprocess).tolist()
        
        print(f"Vectorizing (TF-IDF)...")
        if self.ngram_type == 'trigram':
            n_range = (3, 3)
        elif self.ngram_type == 'bigram':
            n_range = (2, 2)
        else:
            n_range = (1, 1)
            
        self.vectorizer = TfidfVectorizer(ngram_range=n_range, min_df=2)
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_corpus)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print("Calculating Cosine Similarity...")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print("Analysis Complete.")
        return self

    # --- NOUVELLE FONCTION DE CLUSTERING ---
    def perform_clustering(self, n_clusters=3):
        """
        Applique K-Means pour regrouper les textes.
        n_clusters : Le nombre de groupes souhaités.
        """
        print(f"\n--- CLUSTERING (K-Means: {n_clusters} groupes) ---")
        
        # 1. Exécution de K-Means
        self.kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=10, random_state=42)
        self.kmeans_model.fit(self.tfidf_matrix)
        
        # 2. Ajout des labels au DataFrame
        self.df['Cluster_Label'] = self.kmeans_model.labels_
        print(">> Clustering terminé. Labels ajoutés au DataFrame.")
        
        # 3. Affichage des mots-clés par cluster
        print("\n>> Top termes par cluster :")
        order_centroids = self.kmeans_model.cluster_centers_.argsort()[:, ::-1]
        
        for i in range(n_clusters):
            print(f"Cluster {i} : ", end='')
            top_terms = [self.feature_names[ind] for ind in order_centroids[i, :10]] # Top 10 mots
            print(", ".join(top_terms))
            
        return self.df

    # --- NOUVELLE FONCTION DE VISUALISATION CLUSTERING ---
    def visualize_clusters_pca(self):
        """
        Utilise l'ACP (PCA) pour réduire la matrice à 2 dimensions et afficher les clusters.
        """
        if 'Cluster_Label' not in self.df.columns:
            print("Erreur : Veuillez lancer perform_clustering() avant de visualiser.")
            return

        print("\n--- VISUALISATION DES CLUSTERS (PCA) ---")
        
        # Réduction de dimensionnalité (ACP)
        # On utilise toarray() car PCA ne gère pas toujours bien les matrices sparse directement
        pca = PCA(n_components=2, random_state=42)
        reduced_features = pca.fit_transform(self.tfidf_matrix.toarray())
        
        reduced_cluster_centers = pca.transform(self.kmeans_model.cluster_centers_)
        
        plt.figure(figsize=(10, 8))
        
        # Création d'une palette de couleurs
        colors = plt.cm.get_cmap('tab10', len(self.df['Cluster_Label'].unique()))
        
        # Scatter Plot
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                              c=self.df['Cluster_Label'], 
                              cmap='viridis', 
                              s=50, alpha=0.6)
        
        # Affichage des centres de clusters (croix rouges)
        plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], 
                    marker='x', s=200, c='red', label='Centres')

        plt.title(f"Visualisation des Clusters de Produits (PCA - {self.ngram_type})")
        plt.xlabel("Composante Principale 1")
        plt.ylabel("Composante Principale 2")
        plt.colorbar(scatter, label='Numéro du Cluster')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
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

# --- EXECUTION ---

miner = TextMiner(
    'cleaned_all_products.xlsx', 
    column_name='description', 
    ngram_type='bigram',          
    normalization='lemmatization'
)

if miner.run_analysis():
    # 1. Générer statistiques globales
    miner.generate_statistics_and_cloud()
    
    # 2. Lancer le CLUSTERING (Ex: 5 groupes)
    # Ajustez 'n_clusters' selon la diversité de vos produits
    miner.perform_clustering(n_clusters=10)
    
    # 3. Visualiser les clusters
    miner.visualize_clusters_pca()
    
    # (Optionnel) Sauvegarder le fichier Excel avec la nouvelle colonne 'Cluster_Label'
    # miner.df.to_excel("resultats_avec_clusters.xlsx", index=False)
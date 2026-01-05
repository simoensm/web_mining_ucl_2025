
import pandas as pd # Pour la manipulation des données
import re # Pour le Regex
import nltk # Pour le traitement du langage naturel
import numpy as np # Pour les calculs numériques
import matplotlib.pyplot as plt # Pour les graphiques
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer # sklearn contient des outils de Machine Learning
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity # Pour le calcul de la similarité cosinus
from sklearn.preprocessing import normalize
import os # Pour la gestion des fichiers

# S'assure que les ressources NLTK nécessaires sont téléchargées
try:
    nltk.data.find('tokenizers/punkt') 
    nltk.data.find('corpora/wordnet')
except LookupError:
    print(">> NLTK...")
    nltk.download('punkt') # Tokenisation : découpage en mots
    nltk.download('stopwords') # Liste des mots vides
    nltk.download('punkt_tab') # Dépendance pour punkt
    nltk.download('wordnet') # Lemmantisation
    nltk.download('omw-1.4') # Plusieurs langues pour WordNet

class TextMinerTXT:
    def __init__(self, file_path, chunk_size=100, ngram_type='unigram', normalization='lemmatization'): 
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.ngram_type = ngram_type
        self.normalization = normalization
        
        # Choix entre Lemmantisation et Racinisation
        if self.normalization == 'stemming':
            self.stemmer = PorterStemmer()
        elif self.normalization == 'lemmatization':
            self.lemmatizer = WordNetLemmatizer()
            
        self.stop_words = set(stopwords.words('english'))

        # Mots à ignorer car non pertinents
        noise_words = {
            'intro', 'details', 'specs', 'features', 'materials', 'care', 'instructions',
            'weight', 'country', 'origin', 'made', 'factory', 'certified',
            'machine', 'wash', 'warm', 'cold', 'bleach', 'dry', 'tumble', 'iron',
            'oz', 'g', 'lbs', 'premium', 'product', 'regular', 'fit', 'size', 'color',
            'cool', 'intentionally', 'saying', 'finish', 'visit', 'guide',
            'year', 'report', 'work', 'worker', 'page', 'chapter'
        }
        self.stop_words.update(noise_words)
        
        # Variables pour stocker les résultats temporaires
        self.vectorizer = None
        self.tfidf_matrix = None
        self.tf_matrix = None     
        self.idf_vector = None    
        self.cosine_sim = None
        self.df = None  # DataFrame avec les segments
        self.feature_names = None
        self.kmeans_model = None

    def preprocess(self, text):
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
        print(f"\n1. Loading & Pre-Process({self.normalization.upper()})") 
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
        except FileNotFoundError:
            print(f"Error: Unable to find {self.file_path}")
            return False
        
        # Découpage en segments (chunks) pour simuler des "documents"
        all_words = full_text.split()
        raw_segments = [' '.join(all_words[i:i + self.chunk_size]) 
                        for i in range(0, len(all_words), self.chunk_size)]
        
        print(f"   > Text split into {len(raw_segments)} segments (chunk_size={self.chunk_size} words)")
        print("   > Applying pre-processing...")
        
        # Création du DataFrame (similaire à text_mining_main)
        processed_segments = [self.preprocess(seg) for seg in raw_segments]
        
        # Filtrer les segments vides
        valid_data = [(i, seg) for i, seg in enumerate(processed_segments) if len(seg.strip()) > 0]
        
        self.df = pd.DataFrame({
            'segment_id': [d[0] for d in valid_data],
            'name': [f"Segment_{d[0]}" for d in valid_data],  # Pour compatibilité avec export Gephi
            'processed_text': [d[1] for d in valid_data]
        })
        
        print(f"   > {len(self.df)} segments loaded (after filtering empty ones).")
        return True

    def show_word_frequencies(self):
        print(f"\n2. Frequency Analysis ({self.ngram_type.upper()})")
        
        if self.ngram_type == 'trigram': n_range = (3, 3)
        elif self.ngram_type == 'bigram': n_range = (2, 2)
        else: n_range = (1, 1)

        min_df_val = 2 if len(self.df) > 2 else 1
        cv = CountVectorizer(ngram_range=n_range, min_df=min_df_val)
        X = cv.fit_transform(self.df['processed_text']) 
        
        sum_words = X.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        
        print("\n>> Top 20 Most Frequent Words:")
        for word, freq in words_freq[:20]:
            print(f"   - {word}: {freq}")

        # Génération du nuage de mots
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(words_freq))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Nuage de Mots ({self.ngram_type} - {self.normalization})")
        plt.show()

    def vectorize_tfidf_manual(self, n_docs=5):
        print(f"\n3. TF-IDF Vectorization")
        
        if self.ngram_type == 'trigram': n_range = (3, 3)
        elif self.ngram_type == 'bigram': n_range = (2, 2)
        else: n_range = (1, 1)

        # Ajustement dynamique de min_df et max_df selon le nombre de segments
        n_segments = len(self.df)
        min_df_val = max(1, min(3, n_segments // 10))
        max_df_val = 0.80
        
        self.vectorizer = CountVectorizer(ngram_range=n_range, min_df=min_df_val, max_df=max_df_val)

        raw_matrix_sparse = self.vectorizer.fit_transform(self.df['processed_text']) 
        raw_counts = raw_matrix_sparse.toarray() 
        self.feature_names = self.vectorizer.get_feature_names_out() 
        
        print(f"   > matrix calculated: {raw_counts.shape} (Segments x Tokens)")

        # TF (Fréquence Relative)
        max_counts_per_doc = raw_counts.max(axis=1)
        max_counts_per_doc[max_counts_per_doc == 0] = 1 
        self.tf_matrix = raw_counts / max_counts_per_doc[:, np.newaxis]

        # IDF (Specificité)
        N_docs = raw_counts.shape[0]
        doc_freq = (raw_counts > 0).sum(axis=0)
        doc_freq[doc_freq == 0] = 1
        self.idf_vector = np.log(N_docs / doc_freq)

        # TF-IDF Final
        self.tfidf_matrix = self.tf_matrix * self.idf_vector
        print(f"   > Final TF-IDF matrix ready.")

        # Normalisation L2
        self.tfidf_matrix = normalize(self.tfidf_matrix, norm='l2', axis=1)
        print(f"   > Final TF-IDF matrix normalized and ready.")

        # Exportation de la matrice TF-IDF
        try:
            filename = f"tfidf_matrix_txt_{self.ngram_type}.xlsx"
            print(f"   > Exporting full matrix to {filename}")
            df_tfidf = pd.DataFrame(self.tfidf_matrix, columns=self.feature_names)
            df_tfidf.to_excel(filename, index=False)
            print(f"   > Export successful: {filename}")
        except Exception as e:
            print(f"   > Error exporting TF-IDF matrix: {e}")

        # Prévisualisation dans la console
        subset_tf = self.tf_matrix[:n_docs, :]
        active_indices = np.where(subset_tf.sum(axis=0) > 0)[0]
        if len(active_indices) > 8:  
            active_indices = active_indices[:8]
        elif len(active_indices) == 0:
            active_indices = np.arange(min(8, len(self.feature_names)))

        feature_subset = self.feature_names[active_indices]

        print("\n   > TF (Relative Frequency):")
        print(pd.DataFrame(self.tf_matrix[:n_docs, active_indices], columns=feature_subset).round(3))

        print("\n   > IDF (Logarithm):")
        print(pd.DataFrame([self.idf_vector[active_indices]], columns=feature_subset).round(3)) 

        print("\n   > TF-IDF Final:")
        print(pd.DataFrame(self.tfidf_matrix[:n_docs, active_indices], columns=feature_subset).round(3))

    def show_elbow_method(self, max_k=10):
        print(f"\n4. Determining Number of Clusters (Elbow Method)")
        
        # Limiter max_k au nombre de segments
        max_k = min(max_k, len(self.df) - 1)
        if max_k < 2:
            print("   > Not enough segments for clustering analysis.")
            return
            
        inertias = []
        K_range = range(1, max_k + 1)
        
        print("   > Calculating inertias...", end='')
        for k in K_range:
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            km.fit(self.tfidf_matrix)
            inertias.append(km.inertia_)
            print(".", end='')
        print(" Done.")

        plt.figure(figsize=(8, 4))
        plt.plot(K_range, inertias, 'bx-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.grid(True)
        plt.show()

    def perform_clustering(self, n_clusters):
        """Effectue le clustering K-Means"""
        print(f"\n5. Clustering (K={n_clusters})")
        
        # Vérification du nombre de clusters
        n_clusters = min(n_clusters, len(self.df) - 1)
        if n_clusters < 2:
            print("   > Not enough segments for clustering.")
            return
            
        self.kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
        self.kmeans_model.fit(self.tfidf_matrix)
        self.df['cluster'] = self.kmeans_model.labels_
        
        # Calcul du Silhouette Score
        if n_clusters > 1:
            sil_score = silhouette_score(self.tfidf_matrix, self.kmeans_model.labels_)
            print(f"\n   > Silhouette Score: {sil_score:.4f}")
            print(f"   > (Score between -1 and 1: closer to 1 = better clustering)")
        
        order_centroids = self.kmeans_model.cluster_centers_.argsort()[:, ::-1]
        
        for i in range(n_clusters):
            print(f"\n   CLUSTER {i} :")
            top_terms = [self.feature_names[ind] for ind in order_centroids[i, :10]]
            print(f"   Keywords : {', '.join(top_terms)}")
            print(f"   Size : {len(self.df[self.df['cluster'] == i])} segments")

    def save_results(self, output_file):
        """Sauvegarde les résultats dans un fichier Excel"""
        print(f"\n6. SAVING")
        try:
            self.df.to_excel(output_file, index=False)
            print(f"   > File saved successfully: {output_file}")
            print(f"   > Added column: 'cluster'")
        except Exception as e:
            print(f"   > Error during saving: {e}")

    def visualize_pca(self):
        """Visualisation PCA en 2D"""
        print("\n7. PCA Visualization")
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(self.tfidf_matrix)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=self.df['cluster'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title('PCA Projection (based on manual TF-IDF)')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()

    def compute_cosine_similarity(self):
        """Calcule la matrice de similarité cosinus entre tous les segments"""
        print("\n8. Computing Cosine Similarity Matrix")
        if self.tfidf_matrix is None:
            print("   > Error: TF-IDF matrix not computed. Run vectorize_tfidf_manual() first.")
            return None
        
        self.cosine_sim = cosine_similarity(self.tfidf_matrix)
        print(f"   > Cosine similarity matrix computed: {self.cosine_sim.shape}")
        return self.cosine_sim

    def export_to_gephi(self, output_dir='.', name_column='name', similarity_threshold=0.20):
        """
        Exporte les données vers Gephi avec deux fichiers CSV:
        - nodes.csv : contient les noms des segments (Id, Label, Cluster)
        - edges.csv : contient les arêtes avec le poids (similarité cosinus)
        """
        print(f"\n9. Export to Gephi (Cosine Similarity)")
        
        if self.tfidf_matrix is None:
            print("   > Error: TF-IDF matrix not computed. Run vectorize_tfidf_manual() first.")
            return False
        
        if self.df is None:
            print("   > Error: Data not loaded. Run load_and_process() first.")
            return False
        
        if self.cosine_sim is None:
            self.compute_cosine_similarity()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"   > Created output directory: {output_dir}")
        
        # CRÉATION DU FICHIER NODES
        print("   > Generating Nodes file...")
        
        if name_column not in self.df.columns:
            print(f"   > Warning: Column '{name_column}' not found. Using index as label.")
            labels = [f"Segment_{i}" for i in range(len(self.df))]
        else:
            labels = self.df[name_column].tolist()
        
        nodes_data = {
            'Id': range(len(self.df)),
            'Label': labels
        }
        
        if 'cluster' in self.df.columns:
            nodes_data['Cluster'] = self.df['cluster'].tolist()
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_file = os.path.join(output_dir, f"gephi_nodes_txt_{self.ngram_type}.csv")
        nodes_df.to_csv(nodes_file, index=False, encoding='utf-8')
        print(f"   > Nodes file saved: {nodes_file} ({len(nodes_df)} nodes)")
        
        # CRÉATION DU FICHIER EDGES
        print(f"   > Generating Edges file (threshold: {similarity_threshold})...")
        
        upper_tri = np.triu(self.cosine_sim, k=1)
        rows, cols = np.where(upper_tri > similarity_threshold)
        weights = upper_tri[rows, cols]
        
        print(f"   > Found {len(weights)} connections above threshold")
        
        edges_df = pd.DataFrame({
            'Source': rows,
            'Target': cols,
            'Weight': np.round(weights, 4),
            'Type': 'Undirected'
        })
        
        edges_file = os.path.join(output_dir, f"gephi_edges_txt_{self.ngram_type}.csv")
        edges_df.to_csv(edges_file, index=False, encoding='utf-8')
        print(f"   > Edges file saved: {edges_file} ({len(edges_df)} edges)")
        
        # STATISTIQUES DE SIMILARITÉ
        print("\n   > Similarity Statistics:")
        if len(weights) > 0:
            print(f"      - Min similarity: {weights.min():.4f}")
            print(f"      - Max similarity: {weights.max():.4f}")
            print(f"      - Mean similarity: {weights.mean():.4f}")
            print(f"      - Median similarity: {np.median(weights):.4f}")
        else:
            print("      - No connections found above threshold.")
            print("      - Try lowering the similarity_threshold parameter.")
        
        print("\n   > Gephi Export Complete!")
        print(f"      Import '{nodes_file}' as Nodes Table")
        print(f"      Import '{edges_file}' as Edges Table")
        
        return True

    def export_similarity_matrix(self, output_file='similarity_matrix_txt.xlsx'):
        """Exporte la matrice complète de similarité cosinus vers Excel"""
        print(f"\n10. Export Similarity Matrix")
        
        if self.cosine_sim is None:
            self.compute_cosine_similarity()
        
        if 'name' in self.df.columns:
            labels = self.df['name'].tolist()
        else:
            labels = [f"Segment_{i}" for i in range(len(self.df))]
        
        sim_df = pd.DataFrame(self.cosine_sim, index=labels, columns=labels)
        
        try:
            sim_df.to_excel(output_file)
            print(f"   > Similarity matrix exported to: {output_file}")
        except Exception as e:
            print(f"   > Error exporting similarity matrix: {e}")
        
        return sim_df

    def export_token_frequencies(self, output_file='token_frequencies_txt.xlsx'):
        print(f"\n11. Export Token Frequencies")
        
        if self.df is None:
            print("   > Error: Data not loaded. Run load_and_process() first.")
            return None
        
        # Définir la plage de n-grammes
        if self.ngram_type == 'trigram': n_range = (3, 3)
        elif self.ngram_type == 'bigram': n_range = (2, 2)
        else: n_range = (1, 1)
        
        # CountVectorizer pour compter toutes les occurrences (min_df=1 pour inclure tous les tokens)
        cv = CountVectorizer(ngram_range=n_range, min_df=1)
        X = cv.fit_transform(self.df['processed_text'])
        
        # Calcul des fréquences totales
        total_counts = np.asarray(X.sum(axis=0)).flatten()
        tokens = cv.get_feature_names_out()
        
        # Création du DataFrame
        freq_df = pd.DataFrame({
            'Token': tokens,
            'Frequency': total_counts,
            'Segments_containing': np.asarray((X > 0).sum(axis=0)).flatten()  # Nombre de segments contenant le token
        })
        
        # Tri par fréquence décroissante
        freq_df = freq_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
        
        # Ajout du rang
        freq_df.insert(0, 'Rank', range(1, len(freq_df) + 1))
        
        try:
            freq_df.to_excel(output_file, index=False)
            print(f"   > Token frequencies exported to: {output_file}")
            print(f"   > Total unique tokens: {len(freq_df)}")
            print(f"   > Total token instances: {freq_df['Frequency'].sum()}")
        except Exception as e:
            print(f"   > Error exporting token frequencies: {e}")
        
        return freq_df


if __name__ == "__main__":
    
    print("\nCONFIGURATION")
    print("1. Lemmatization")
    print("2. Stemming")
    norm_choice = input("Choice (1/2) : ").strip()
    norm_mode = 'lemmatization' if norm_choice == '1' else 'stemming'

    print("\n1. Unigram")
    print("2. Bigram")
    print("3. Trigram")
    ngram_choice = input("Choice (1/2/3) : ").strip()
    ngram_map = {'1': 'unigram', '2': 'bigram', '3': 'trigram'}
    ngram_mode = ngram_map.get(ngram_choice, 'unigram')

    # Demande la taille des segments
    try:
        chunk_size = int(input("\n>> Chunk size (words per segment, default=100) : ").strip() or "100")
    except:
        chunk_size = 100

    miner = TextMinerTXT(
        '.patagonia/patagonia-progress-report-2025.txt',  # à remplacer
        chunk_size=chunk_size,
        ngram_type=ngram_mode, 
        normalization=norm_mode
    )

    if miner.load_and_process():
        miner.show_word_frequencies()
        miner.vectorize_tfidf_manual(n_docs=5)
        miner.show_elbow_method(max_k=15)
        
        try:
            k = int(input("\n>> Desired number of clusters (e.g., 5) : "))
        except:
            k = 5
        
        miner.perform_clustering(n_clusters=k)
        
        output_name = f"segments_clustered_txt_{ngram_mode}.xlsx"  # à remplacer
        miner.save_results(output_name)
        
        miner.visualize_pca()
        
        # --- EXPORT GEPHI (COSINE SIMILARITY) ---
        print("\n" + "="*50)
        print("GEPHI EXPORT OPTIONS")
        print("="*50)
        gephi_choice = input("\n>> Export data to Gephi? (y/n) : ").strip().lower()
        
        if gephi_choice == 'y':
            try:
                threshold = float(input(">> Similarity threshold (0.0-1.0, default=0.20) : ").strip() or "0.20")
                threshold = max(0.0, min(1.0, threshold))
            except:
                threshold = 0.20
            
            output_dir = input(">> Output directory (default='.') : ").strip() or '.'
            
            miner.export_to_gephi(
                output_dir=output_dir,
                name_column='name',
                similarity_threshold=threshold
            )
            
            matrix_choice = input("\n>> Export full similarity matrix to Excel? (y/n) : ").strip().lower()
            if matrix_choice == 'y':
                matrix_file = os.path.join(output_dir, f"similarity_matrix_txt_{ngram_mode}.xlsx")
                miner.export_similarity_matrix(output_file=matrix_file)
        
        # --- EXPORT TOKEN FREQUENCIES ---
        print("\n" + "="*50)
        print("TOKEN FREQUENCIES EXPORT")
        print("="*50)
        token_choice = input("\n>> Export all tokens with frequencies to Excel? (y/n) : ").strip().lower()
        
        if token_choice == 'y':
            token_output_dir = input(">> Output directory (default='.') : ").strip() or '.'
            token_file = os.path.join(token_output_dir, f"token_frequencies_txt_{ngram_mode}.xlsx")
            miner.export_token_frequencies(output_file=token_file)
        
        print("\nFINISHED")
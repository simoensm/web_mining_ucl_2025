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

class TextMiner:
    def __init__(self, file_path, column_name='description_clean', ngram_type='unigram', normalization='lemmatization'): 
        self.file_path = file_path
        self.column_name = column_name
        self.ngram_type = ngram_type
        self.normalization = normalization
        
        # Choix entre Lemmantisation et Racinisation, la lemmantisation transforme un mot à sa forme de dictionnaire, la racinisation coupe le mot à sa racine
        if self.normalization == 'stemming':
            self.stemmer = PorterStemmer()
        elif self.normalization == 'lemmatization':
            self.lemmatizer = WordNetLemmatizer()
            
        self.stop_words = set(stopwords.words('english')) # Bilbiothèque des Stop Words NLTK déjà existante

        # Mots à ignorer car non pertinents : à compléter manuellement ou à l'aide de l'IA par rapport aux résultats obtenus
        noise_words = {
            'intro', 'details', 'specs', 'features', 'materials', 'care', 'instructions',
            'weight', 'country', 'origin', 'made', 'factory', 'certified',
            'machine', 'wash', 'warm', 'cold', 'bleach', 'dry', 'tumble', 'iron',
            'oz', 'g', 'lbs', 'premium', 'product', 'regular', 'fit', 'size', 'color'
        }
        self.stop_words.update(noise_words) # Ajout des mots "bruit" à la liste des stop words
        
        # Variables pour stocker les résultats temporaires
        self.vectorizer = None
        self.tfidf_matrix = None
        self.tf_matrix = None     
        self.idf_vector = None    
        self.cosine_sim = None
        self.df = None
        self.feature_names = None



    def preprocess(self, text): # Nettoye (encore) grossièrement le texte
        text = str(text).lower() # Met en minuscules
        text = re.sub(r'[^\w\s]', '', text) # Enlèvre les signes de ponctuation
        text = re.sub(r'\d+', '', text) # Enlève les chiffres
        tokens = word_tokenize(text) # Tokenisation
        
        cleaned = [] # Liste des mots nettoyés

        for t in tokens:
            if t not in self.stop_words and len(t) > 2: # Ignore les stop words et les mots de moins de 3 caractères
                if self.normalization == 'stemming': 
                    word_final = self.stemmer.stem(t) # Racinisation
                elif self.normalization == 'lemmatization':
                    word_final = self.lemmatizer.lemmatize(t) # Lemmantisation
                else:
                    word_final = t
                cleaned.append(word_final) # Ajoute le mot racinisé/lemmantisé à la liste

        return " ".join(cleaned) # Retourne sous forme de chaîne

    def load_and_process(self):
        print(f"\n1. Loading & Pre-Process({self.normalization.upper()})") 
        try:
            self.df = pd.read_excel(self.file_path) # Charge le fichier Excel
        except FileNotFoundError:
            print(f"Error: Unable to find {self.file_path}")
            return False
            
        self.df = self.df.dropna(subset=[self.column_name]).reset_index(drop=True) # Supprime les lignes vides
        print(f"   > {len(self.df)} documents loaded.")
        print("   > Applying pre-processing...")
        self.df['processed_text'] = self.df[self.column_name].apply(self.preprocess) # Fonction preprocess
        return True

    def show_word_frequencies(self): # Affiche les 20 mots les plus fréquents et génère le nuage de mots
        print(f"\n2. Frequency Analysis ({self.ngram_type.upper()})")
        if self.ngram_type == 'trigram': n_range = (3, 3) # Choix des n-grammes utilisés pour l'analyse (trigram)
        elif self.ngram_type == 'bigram': n_range = (2, 2) # Choix des n-grammes utilisés pour l'analyse (bigram)
        else: n_range = (1, 1) # Choix des n-grammes utilisés pour l'analyse (unigram)

        cv = CountVectorizer(ngram_range=n_range, min_df=2) # Compte le nombre d'occurrences
        X = cv.fit_transform(self.df['processed_text']) 
        
        sum_words = X.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True) # Total des colonnes pour avoir le total des occurences
        
        print("\n>> Top 20 Most Frequent Words:")
        for word, freq in words_freq[:20]: # Affiche les 20 mots les plus fréquents
            print(f"   - {word}: {freq}")

        # Génération du nuage de mots
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(words_freq)) #On initialise la toile de pixels 800*400 et on demande un fond blanc
        plt.figure(figsize=(10, 5)) # Taille de la figure peinte sur matplotlib
        plt.imshow(wordcloud, interpolation='bilinear') #pour un rendu plus lisse, plus net
        plt.axis('off') # Pas d'axes pour le nuage de mots
        plt.title(f"Nuage de Mots ({self.ngram_type} - {self.normalization})") # Titre du graphique
        plt.show() # Affiche le nuage de mots

    def vectorize_tfidf_manual(self, n_docs=5):
        print(f"\n3. TF-IDF Vectorization")
        
        if self.ngram_type == 'trigram': n_range = (3, 3)
        elif self.ngram_type == 'bigram': n_range = (2, 2)
        else: n_range = (1, 1)

        
        self.vectorizer = CountVectorizer(ngram_range=n_range, min_df=3, max_df=0.85) # Compte le nombre de mots
        # min_df=3 : ignore les mots qui apparaissent dans moins de 3 documents (trop rares)
        # max_df=0.85 : ignore les mots qui apparaissent dans plus de 85% des documents (trop communs)

        raw_matrix_sparse = self.vectorizer.fit_transform(self.df['processed_text'])
        raw_counts = raw_matrix_sparse.toarray()
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"   > matrix calculated: {raw_counts.shape} (Documents x Tokens)")

        # TF (Fréquence Relative) => (Nombre d'occurrences du mot) / (Nombre max d'occurrences dans le corpus
        max_counts_per_doc = raw_counts.max(axis=1)
        max_counts_per_doc[max_counts_per_doc == 0] = 1 
        self.tf_matrix = raw_counts / max_counts_per_doc[:, np.newaxis]

        # IDF (Specificité) => log(Nombre total descriptions / Nombre de descriptions contenant le mot)
        N_docs = raw_counts.shape[0]
        doc_freq = (raw_counts > 0).sum(axis=0)  #Eviter la division par zero car erreurs
        doc_freq[doc_freq == 0] = 1
        self.idf_vector = np.log(N_docs / doc_freq)

        # TF-IDF Final => TF * IDF
        self.tfidf_matrix = self.tf_matrix * self.idf_vector
        print(f"   > Final TF-IDF matrix ready.")

        # Exportation de la matrice TF-IDF complète sur excel
        try:
            filename = f"tfidf_matrix_{self.ngram_type}.xlsx"
            print(f"   > Exporting full matrix to {filename}")
            
            df_tfidf = pd.DataFrame(self.tfidf_matrix, columns=self.feature_names)
            
            
            df_tfidf.to_excel(filename, index=False)
            print(f"   > Export successful: {filename}")
        except Exception as e:
            print(f"   > Error exporting TF-IDF matrix: {e}")
    
        subset_tf = self.tf_matrix[:n_docs, :]
        active_indices = np.where(subset_tf.sum(axis=0) > 0)[0]
        
        if len(active_indices) > 8: 
            active_indices = active_indices[:8]
        elif len(active_indices) == 0:
            active_indices = np.arange(min(8, len(self.feature_names)))

        feature_subset = self.feature_names[active_indices]

         # Affichage des matrices TF, IDF et TF-IDF

        print("\n   > TF (Relative Frequency):")
        print(pd.DataFrame(self.tf_matrix[:n_docs, active_indices], columns=feature_subset).round(3))

        print("\n   > IDF (Logarithm):")
        print(pd.DataFrame([self.idf_vector[active_indices]], columns=feature_subset).round(3))

        print("\n   > TF-IDF Final:")
        print(pd.DataFrame(self.tfidf_matrix[:n_docs, active_indices], columns=feature_subset).round(3))

    def show_elbow_method(self, max_k=10): # Affiche la méthode du coude pour déterminer le nombre optimal de clusters
        print(f"\n4. Determining Number of Clusters (Elbow Method)")
        inertias = []
        K_range = range(1, max_k + 1)
        
        print("   > Calculating inertias...", end='')
        for k in K_range:
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            km.fit(self.tfidf_matrix)
            inertias.append(km.inertia_)
            print(".", end='')
        print(" Done.")

        # Affichage du graphique
        plt.figure(figsize=(8, 4))
        plt.plot(K_range, inertias, 'bx-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.grid(True)
        plt.show()

    def perform_clustering(self, n_clusters):
        print(f"\n5. Clustering (K={n_clusters})")
        self.kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42) # Application de K-Means
        self.kmeans_model.fit(self.tfidf_matrix)
        self.df['cluster'] = self.kmeans_model.labels_
        
        order_centroids = self.kmeans_model.cluster_centers_.argsort()[:, ::-1] # Mots clés par cluster
        
        for i in range(n_clusters):
            print(f"\n   CLUSTER {i} :")
            top_terms = [self.feature_names[ind] for ind in order_centroids[i, :10]] # 10 mots les plus représentatifs
            print(f"   Keywords : {', '.join(top_terms)}")
            print(f"   Size : {len(self.df[self.df['cluster'] == i])} products")

    def save_results(self, output_file):
        print(f"\n6. SAVING")
        try:
            self.df.to_excel(output_file, index=False) # Enregistre les résultats dans un nouveau fichier Excel
            print(f"   > File saved successfully: {output_file}")
            print(f"   > Added column: 'cluster'")
        except Exception as e:
            print(f"   > Error during saving: {e}")

    def visualize_pca(self): # PCA pour projeter les données en 2D
        print("\n7. PCA Visualization")
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(self.tfidf_matrix)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=self.df['cluster'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title('PCA Projection (based on manual TF-IDF)')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()

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

    miner = TextMiner('all_dataset.xlsx', ngram_type=ngram_mode, normalization=norm_mode) # à remplacer

    if miner.load_and_process():
        miner.show_word_frequencies()
        miner.vectorize_tfidf_manual(n_docs=5)
        miner.show_elbow_method(max_k=15)
        
        try:
            k = int(input("\n>> Desired number of clusters (e.g., 5) : "))
        except:
            k = 5
        
        miner.perform_clustering(n_clusters=k)
        
        output_name = f"all_products_clustered_{ngram_mode}.xlsx" # à remplacer
        miner.save_results(output_name)
        
        miner.visualize_pca()
        
        print("\nFINISHED")
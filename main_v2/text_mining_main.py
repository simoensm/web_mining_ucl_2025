import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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

class TextMiner:
    def __init__(self, file_path, column_name='description_clean', ngram_type='unigram', normalization='lemmatization'):
        self.file_path = file_path
        self.column_name = column_name
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
            'oz', 'g', 'lbs', 'premium', 'product', 'regular', 'fit', 'size', 'color'
        }
        self.stop_words.update(noise_words)
        
        self.vectorizer = None
        self.tfidf_matrix = None
        self.tf_matrix = None     
        self.idf_vector = None    
        self.cosine_sim = None
        self.df = None
        self.feature_names = None

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

    def load_and_process(self):
        print(f"\n1. Loading & Pre-Process({self.normalization.upper()})")
        try:
            self.df = pd.read_excel(self.file_path)
        except FileNotFoundError:
            print(f"Error: Unable to find {self.file_path}")
            return False
            
        self.df = self.df.dropna(subset=[self.column_name]).reset_index(drop=True)
        print(f"   > {len(self.df)} documents loaded.")
        print("   > Applying pre-processing...")
        self.df['processed_text'] = self.df[self.column_name].apply(self.preprocess)
        return True

    def show_word_frequencies(self):
        print(f"\n2. Frequency Analysis ({self.ngram_type.upper()})")
        if self.ngram_type == 'trigram': n_range = (3, 3)
        elif self.ngram_type == 'bigram': n_range = (2, 2)
        else: n_range = (1, 1)

        cv = CountVectorizer(ngram_range=n_range, min_df=2)
        X = cv.fit_transform(self.df['processed_text'])
        
        sum_words = X.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        
        print("\n>> Top 20 Most Frequent Words:")
        for word, freq in words_freq[:20]:
            print(f"   - {word}: {freq}")

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(words_freq))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Nuage de Mots ({self.ngram_type} - {self.normalization})")
        plt.show()

    def vectorize_tfidf_manual(self):
        print(f"\n3. TF-IDF Vectorizatioin")
        
        if self.ngram_type == 'trigram': n_range = (3, 3)
        elif self.ngram_type == 'bigram': n_range = (2, 2)
        else: n_range = (1, 1)
        
        self.vectorizer = CountVectorizer(ngram_range=n_range, min_df=3, max_df=0.85)
        raw_matrix_sparse = self.vectorizer.fit_transform(self.df['processed_text'])
        raw_counts = raw_matrix_sparse.toarray()
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"   > matrix calculated: {raw_counts.shape} (Documents x Tokens)")

        # 1. TF (Fréquence Relative)
        max_counts_per_doc = raw_counts.max(axis=1)
        max_counts_per_doc[max_counts_per_doc == 0] = 1
        self.tf_matrix = raw_counts / max_counts_per_doc[:, np.newaxis]
        print("   > TF (Relative Frequency) calculated.")

        # 2. IDF (Specificity)
        N_docs = raw_counts.shape[0]
        doc_freq = (raw_counts > 0).sum(axis=0)
        doc_freq[doc_freq == 0] = 1
        self.idf_vector = np.log(N_docs / doc_freq)
        print("   > IDF (Logarithm) calculated.")

        # 3. TF-IDF Final
        self.tfidf_matrix = self.tf_matrix * self.idf_vector
        print(f"   > Final TF-IDF matrix ready.")
        
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def explain_tfidf_breakdown(self, n_docs=5):
        print("\n" + "="*60)
        print("   TF-IDF SCORE BREAKDOWN (COURSE METHOD)")
        print("="*60)
        
        subset_tf = self.tf_matrix[:n_docs, :]
        active_indices = np.where(subset_tf.sum(axis=0) > 0)[0]
        if len(active_indices) > 8: active_indices = active_indices[:8]
            
        feature_subset = self.feature_names[active_indices]
        
        print("\n[A] TF MATRIX (Relative Frequency)")
        print(pd.DataFrame(self.tf_matrix[:n_docs, active_indices], columns=feature_subset).round(3))

        print("\n[B] IDF VECTOR (Specificity)")
        print(pd.DataFrame([self.idf_vector[active_indices]], columns=feature_subset).round(3))

        print("\n[C] FINAL MATRIX (TF * IDF)")
        print(pd.DataFrame(self.tfidf_matrix[:n_docs, active_indices], columns=feature_subset).round(3))
        print("="*60 + "\n")

    def show_elbow_method(self, max_k=10):
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

        plt.figure(figsize=(8, 4))
        plt.plot(K_range, inertias, 'bx-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.grid(True)
        plt.show()

    def perform_clustering(self, n_clusters):
        print(f"\n5. Clustering (K={n_clusters})")
        self.kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
        self.kmeans_model.fit(self.tfidf_matrix)
        self.df['cluster'] = self.kmeans_model.labels_
        
        order_centroids = self.kmeans_model.cluster_centers_.argsort()[:, ::-1]
        
        for i in range(n_clusters):
            print(f"\n   CLUSTER {i} :")
            top_terms = [self.feature_names[ind] for ind in order_centroids[i, :10]]
            print(f"   Keywords : {', '.join(top_terms)}")
            print(f"   Size : {len(self.df[self.df['cluster'] == i])} products")

    def save_results(self, output_file):
        """ Save the DataFrame enriched with clusters to an Excel file """
        print(f"\n--- 6. SAVING ---")
        try:
            self.df.to_excel(output_file, index=False)
            print(f"   > File saved successfully: {output_file}")
            print(f"   > Added column: 'cluster'")
        except Exception as e:
            print(f"   > Error during saving: {e}")

    def visualize_pca(self):
        print("\n--- 7. PCA Visualization ---")
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

    miner = TextMiner('.patagonia/patagonia_dataset.xlsx', ngram_type=ngram_mode, normalization=norm_mode)

    if miner.load_and_process():
        miner.show_word_frequencies()
        miner.vectorize_tfidf_manual()
        miner.explain_tfidf_breakdown(n_docs=5)
        miner.show_elbow_method(max_k=15)
        
        try:
            k = int(input("\n>> Desired number of clusters (e.g., 5) : "))
        except:
            k = 5
        
        miner.perform_clustering(n_clusters=k)
        
        output_name = f".patagonia/patagonia_clustered_{ngram_mode}.xlsx"
        miner.save_results(output_name)
        
        miner.visualize_pca()
        
        print("\nFINISHED")
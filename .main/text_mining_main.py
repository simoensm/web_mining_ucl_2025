

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
            'intro', 'details', 'specs', 'features', 'materials', 'care', 'instructions', 'weight', 'country', 'origin', 'made', 'factory', 'certified','machine', 'wash', 'warm', 'cold', 'bleach', 'dry', 'tumble', 'iron','oz', 'g', 'lbs', 'premium', 'product', 'regular', 'fit', 'size', 'color', 'cool', 'intentionally', 'saying', "finish", "visit", "guide"
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
        
        if self.ngram_type == 'trigram': n_range = (3, 3) # Choix des n-grammes utilisés pour l'analyse (trigram)
        elif self.ngram_type == 'bigram': n_range = (2, 2) # Choix des n-grammes utilisés pour l'analyse (bigram)
        else: n_range = (1, 1) # Choix des n-grammes utilisés pour l'analyse (unigram)

        
        self.vectorizer = CountVectorizer(ngram_range=n_range, min_df=3, max_df=0.80) # Compte le nombre de mots
        # min_df=3 : ignore les mots qui apparaissent dans moins de 3 documents (trop rares)
        # max_df=0.85 : ignore les mots qui apparaissent dans plus de 85% des documents (trop communs)

        raw_matrix_sparse = self.vectorizer.fit_transform(self.df['processed_text']) 
        raw_counts = raw_matrix_sparse.toarray() 
        self.feature_names = self.vectorizer.get_feature_names_out() 
        
        print(f"   > matrix calculated: {raw_counts.shape} (Documents x Tokens)")

        # TF (Fréquence Relative) => (Nombre d'occurrences du mot) / (Nombre max d'occurrences dans le corpus)
        max_counts_per_doc = raw_counts.max(axis=1)
        max_counts_per_doc[max_counts_per_doc == 0] = 1 
        self.tf_matrix = raw_counts / max_counts_per_doc[:, np.newaxis]

        # IDF (Specificité) => log(Nombre total descriptions / Nombre de descriptions contenant le mot)
        N_docs = raw_counts.shape[0]
        doc_freq = (raw_counts > 0).sum(axis=0)  #Eviter la division par zero car erreurs
        doc_freq[doc_freq == 0] = 1
        self.idf_vector = np.log(N_docs / doc_freq) # Formule de l'IDF

        # TF-IDF Final => TF * IDF
        self.tfidf_matrix = self.tf_matrix * self.idf_vector
        print(f"   > Final TF-IDF matrix ready.")

        self.tfidf_matrix = normalize(self.tfidf_matrix, norm='l2', axis=1) # On utilise la normalisation L2 pour que la somme des carrés des valeurs de chaque ligne soit égale à 1, ça eviter les différences de longueurs et rend la distance euclidienne équivalente à la similarité cosinus.
        print(f"   > Final TF-IDF matrix normalized and ready.")

        # Exportation de la matrice TF-IDF complète sur excel
        try:
            filename = f"tfidf_matrix_{self.ngram_type}.xlsx"
            print(f"   > Exporting full matrix to {filename}")
            
            df_tfidf = pd.DataFrame(self.tfidf_matrix, columns=self.feature_names) # Crée une table pandas à partir de la matrice TF-IDF
            
            
            df_tfidf.to_excel(filename, index=False) # Enregistre la table ainsi créée dans un fichier Excel
            print(f"   > Export successful: {filename}")
        except Exception as e:
            print(f"   > Error exporting TF-IDF matrix: {e}") #Pour gérer les erreurs d'exportation

        
    
        # --- PRÉVISUALISATION DANS LA CONSOLE ---
        # Le code ci-dessous sert juste à afficher un joli petit tableau dans le terminal pour vérifier que ça a marché, sans afficher les 5000 colonnes 

        subset_tf = self.tf_matrix[:n_docs, :]  #On prend les n_docs premiers documents
        active_indices = np.where(subset_tf.sum(axis=0) > 0)[0]  #On cherche les colonnes actives (avec des valeurs non nulles)
        # On limite l'affichage à 8 colonnes pour ne pas surcharger la console
        if len(active_indices) > 8:  
            active_indices = active_indices[:8]
        elif len(active_indices) == 0:
            active_indices = np.arange(min(8, len(self.feature_names))) #Si toutes les colonnes sont nulles, on prend les 8 premières colonnes par défaut

        feature_subset = self.feature_names[active_indices] # Extraction des noms des mots choisis 

         # Affichage des matrices TF, IDF et TF-IDF

        print("\n   > TF (Relative Frequency):")
        print(pd.DataFrame(self.tf_matrix[:n_docs, active_indices], columns=feature_subset).round(3)) #Arrondi à 3 décimales pour une meilleure lisibilité x3

        print("\n   > IDF (Logarithm):")
        print(pd.DataFrame([self.idf_vector[active_indices]], columns=feature_subset).round(3)) 

        print("\n   > TF-IDF Final:")
        print(pd.DataFrame(self.tfidf_matrix[:n_docs, active_indices], columns=feature_subset).round(3)) 

    def show_elbow_method(self, max_k=10): #Affiche la méthode du coude pour déterminer le nombre optimal de clusters, max_k est le nombre maximum de clusters à tester
        print(f"\n4. Determining Number of Clusters (Elbow Method)")
        inertias = [] #Liste pour stocker les inerties 
        K_range = range(1, max_k + 1) #Crée une plage de k de 1 à max_k
        
        print("   > Calculating inertias...", end='')
        for k in K_range: #On teste chaque valeur de k et on calcule l'inertie
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42) #Initialisation du k-Means, n_init = 10 va lancer l'algorithme 10 fois avec des centroides différents et garder le meilleur
            km.fit(self.tfidf_matrix)                                               #random_state va permettre de reproduire les mêmes résultats à chaque exécution
            inertias.append(km.inertia_) #L'inertie est la somme des distances au carré entre chaque point et le centroïde de son cluster
            print(".", end='') #pour indiquer la progression
        print(" Done.")

        # Affichage du graphique
        plt.figure(figsize=(8, 4)) # Taille de la figure
        plt.plot(K_range, inertias, 'bx-') # 'bx-' signifie des points bleus avec des lignes
        plt.xlabel('Number of Clusters (k)') # Label de l'axe x
        plt.ylabel('Inertia') # Label de l'axe y
        plt.title('Elbow Method') # Titre du graphique
        plt.grid(True) # Grille pour une meilleure lisibilité
        plt.show() # Affiche le graphique

    def perform_clustering(self, n_clusters):
        print(f"\n5. Clustering (K={n_clusters})")
        self.kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42) # Application dU K-Means
        self.kmeans_model.fit(self.tfidf_matrix) #Entraîne le modèle K-Means sur la matrice TF-IDF
        self.df['cluster'] = self.kmeans_model.labels_ #Ajoute les labels de cluster au DataFrame original
        
        # Calcul du Silhouette Score pour évaluer la qualité du clustering
        if n_clusters > 1:
            sil_score = silhouette_score(self.tfidf_matrix, self.kmeans_model.labels_)  #Calcule le score de silhouette
            print(f"\n   > Silhouette Score: {sil_score:.4f}") # Affiche le score de silhouette avec 4 décimales
        else:
            print("\n   > Minimum : 2 clusters.")
        
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
        
        plt.figure(figsize=(10, 6)) # Taille de la figure
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=self.df['cluster'], cmap='viridis', alpha=0.6) # Nuage de points coloré par cluster, cmap 'viridis' pour une bonne visibilité, alpha pour la transparence
        plt.colorbar(scatter, label='Cluster ID') # Barre de couleur pour indiquer les clusters
        plt.title('PCA Projection (based on manual TF-IDF)') # Titre du graphique
        plt.grid(True, linestyle='--', alpha=0.3) # Grille pour une meilleure lisibilité
        plt.show() # Affiche le graphique

    def compute_cosine_similarity(self, export=False, output_file='similarity_matrix.xlsx'): # Calcule la matrice de similarité cosinus
        print("\n8. Computing Cosine Similarity Matrix") 
        if self.tfidf_matrix is None: #Vérifie que la matrice TF-IDF est créée
            print("   > Error: TF-IDF matrix not computed. Run vectorize_tfidf_manual() first.")
            return None
        
        self.cosine_sim = cosine_similarity(self.tfidf_matrix) #Calcule la similarité cosinus entre tous les documents
        print(f"   > Cosine similarity matrix computed: {self.cosine_sim.shape}") #Affiche la taille de la matrice
        
        # Export optionnel de la matrice de similarité
        if export:
            print(f"   > Exporting similarity matrix to {output_file}...") #Export de la matrice de similarité
            if 'name' in self.df.columns:
                labels = self.df['name'].tolist() #Etiquettes basées sur les noms des produits
            else:
                labels = [f"Doc_{i}" for i in range(len(self.df))] #Etiquettes par défaut
            
            sim_df = pd.DataFrame(self.cosine_sim, index=labels, columns=labels) #Création du tableau pour l'export
            try: 
                sim_df.to_excel(output_file)
                print(f"   > Similarity matrix exported successfully: {output_file}") #Message de succès
            except Exception as e:
                print(f"   > Error exporting similarity matrix: {e}") #Message d'erreur en cas d'échec
        
        return self.cosine_sim

    def export_to_gephi(self, output_dir='.', name_column='name', similarity_threshold=0.20): #Exporte les données pour Gephi
        print(f"\n9. Export to Gephi (Cosine Similarity)")
        
        # Vérification des prérequis
        if self.tfidf_matrix is None:
            print("   > Error: TF-IDF matrix not computed. Run vectorize_tfidf_manual() first.") #Erreur si la matrice TF-IDF n'est pas calculée
            return False
        
        if self.df is None:
            print("   > Error: Data not loaded. Run load_and_process() first.") #Erreur si les données ne sont pas chargées
            return False
        
        # Calcul de la similarité cosinus si pas encore fait
        if self.cosine_sim is None:
            self.compute_cosine_similarity()
        
        # Création du répertoire de sortie si nécessaire
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"   > Created output directory: {output_dir}")
        
        # --- CRÉATION DU FICHIER NODES ---
        print("   > Generating Nodes file...")
        
        # Vérifie si la colonne de noms existe
        if name_column not in self.df.columns:
            print(f"   > Warning: Column '{name_column}' not found. Using index as label.") #Avertissement si la colonne n'existe pas
            labels = [f"Product_{i}" for i in range(len(self.df))] #Utilise les indices comme étiquettes si la colonne n'existe pas
        else:
            labels = self.df[name_column].tolist() #Utilise les noms des produits comme étiquettes
        
        # Création du DataFrame nodes
        nodes_data = {
            'Id': range(len(self.df)),
            'Label': labels
        } #Dictionnaire pour les données des nœuds, avec Id et Label
        
        # Ajout du cluster si disponible
        if 'cluster' in self.df.columns:
            nodes_data['Cluster'] = self.df['cluster'].tolist() #Ajoute la colonne de cluster si elle existe
        
        nodes_df = pd.DataFrame(nodes_data) #Création du tableau des nœuds
        nodes_file = os.path.join(output_dir, f"gephi_nodes_{self.ngram_type}.csv") #Nom du fichier des nœuds
        nodes_df.to_csv(nodes_file, index=False, encoding='utf-8') #Enregistrement du fichier des nœuds
        print(f"   > Nodes file saved: {nodes_file} ({len(nodes_df)} nodes)") #Message de succès avec le nombre de nœuds
        
        # --- CRÉATION DU FICHIER EDGES ---
        print(f"   > Generating Edges file (threshold: {similarity_threshold})...")
        
        # Extraction de la partie triangulaire supérieure (évite les doublons A-B et B-A)
        upper_tri = np.triu(self.cosine_sim, k=1)
        
        # Trouve les paires avec similarité au-dessus du seuil
        rows, cols = np.where(upper_tri > similarity_threshold) 
        weights = upper_tri[rows, cols] #Poids des connexions qui représentent la similarité entre les documents
        
        print(f"   > Found {len(weights)} connections above threshold")
        
        # Création du DataFrame edges
        edges_df = pd.DataFrame({
            'Source': rows,
            'Target': cols,
            'Weight': np.round(weights, 4),  # Arrondi à 4 décimales pour la lisibilité
            'Type': 'Undirected'
        }) #Dictionnaire pour les données des arêtes, avec Source, Target, Weight et Type (graph non orienté)
        
        edges_file = os.path.join(output_dir, f"gephi_edges_{self.ngram_type}.csv") #Nom du fichier des arêtes
        edges_df.to_csv(edges_file, index=False, encoding='utf-8') #Enregistrement du fichier des arêtes
        print(f"   > Edges file saved: {edges_file} ({len(edges_df)} edges)") #Message de succès avec le nombre d'arêtes
        
        # --- STATISTIQUES DE SIMILARITÉ ---
        print("\n   > Similarity Statistics:")
        if len(weights) > 0:
            print(f"      - Min similarity: {weights.min():.4f}") #Affiche les statistiques de similarité (min, max, mean, median)
            print(f"      - Max similarity: {weights.max():.4f}") 
            print(f"      - Mean similarity: {weights.mean():.4f}")
            print(f"      - Median similarity: {np.median(weights):.4f}")
        else:
            print("      - No connections found above threshold.") #Si aucune connexion n'est trouvée au-dessus du seuil, suggère de baisser le seuil
            print("      - Try lowering the similarity_threshold parameter.")
        
        print("\n   > Gephi Export Complete!") #Message de fin
        print(f"      Import '{nodes_file}' as Nodes Table") #Indique comment importer les fichiers dans Gephi
        print(f"      Import '{edges_file}' as Edges Table") 
        
        return True

    def export_similarity_matrix(self, output_file='similarity_matrix.xlsx'): #Exporte la matrice de similarité complète vers Excel
        print(f"\n10. Export Similarity Matrix")
        
        if self.cosine_sim is None:
            self.compute_cosine_similarity() #Calcule la matrice de similarité si elle n'est pas déjà calculée
        
        # Création des labels pour les lignes et colonnes
        if 'name' in self.df.columns:
            labels = self.df['name'].tolist() #Utilise les noms des produits comme étiquettes
        else:
            labels = [f"Doc_{i}" for i in range(len(self.df))] #Utilise les indices comme étiquettes si la colonne n'existe pas
        
        # Création du DataFrame avec la matrice de similarité
        sim_df = pd.DataFrame(self.cosine_sim, index=labels, columns=labels)
        
        try:
            sim_df.to_excel(output_file)
            print(f"   > Similarity matrix exported to: {output_file}") #Message de succès
        except Exception as e:
            print(f"   > Error exporting similarity matrix: {e}") #Message d'erreur en cas d'échec
        
        return sim_df

    def export_token_frequencies(self, output_file='token_frequencies.xlsx'): #Exporte tous les tokens avec leurs fréquences dans un fichier Excel
        print(f"\n11. Export Token Frequencies")
        
        if self.df is None:
            print("   > Error: Data not loaded. Run load_and_process() first.") #Erreur si les données ne sont pas chargées
            return None
        
        # Définir la plage de n-grammes
        if self.ngram_type == 'trigram': n_range = (3, 3) #Choix des n-grammes utilisés pour l'analyse (trigram)
        elif self.ngram_type == 'bigram': n_range = (2, 2) #Choix des n-grammes utilisés pour l'analyse (bigram)
        else: n_range = (1, 1) #Choix des n-grammes utilisés pour l'analyse (unigram)
        
        # CountVectorizer pour compter toutes les occurrences (min_df=1 pour inclure tous les tokens)
        cv = CountVectorizer(ngram_range=n_range, min_df=1)
        X = cv.fit_transform(self.df['processed_text']) #Matrice des occurrences
        
        # Calcul des fréquences totales
        total_counts = np.asarray(X.sum(axis=0)).flatten() #Somme des colonnes pour obtenir la fréquence totale de chaque token
        tokens = cv.get_feature_names_out() #Liste des tokens
        
        # Création du DataFrame
        freq_df = pd.DataFrame({
            'token': tokens,
            'frequency': total_counts,
            'documents_containing': np.asarray((X > 0).sum(axis=0)).flatten()  # Nombre de documents contenant le token
        }) #Dictionnaire pour les données des tokens
        
        # Tri par fréquence décroissante
        freq_df = freq_df.sort_values(by='frequency', ascending=False).reset_index(drop=True)
        
        
        try:
            freq_df.to_excel(output_file, index=False) #Enregistrement du fichier Excel
            print(f"   > Token frequencies exported to: {output_file}") #Message de succès
            print(f"   > Total unique tokens: {len(freq_df)}") #Statistiques sur les tokens (total unique, total instances)
            print(f"   > Total token instances: {freq_df['frequency'].sum()}") 
        except Exception as e:
            print(f"   > Error exporting token frequencies: {e}") #Message d'erreur en cas d'échec
        
        return freq_df

if __name__ == "__main__":
    
    print("\nCONFIGURATION")
    print("1. Lemmatization") #Choix entre lemmantisation et racinisation
    print("2. Stemming") 
    norm_choice = input("Choice (1/2) : ").strip()
    norm_mode = 'lemmatization' if norm_choice == '1' else 'stemming'

    print("\n1. Unigram") #Choix entre unigram, bigram et trigram
    print("2. Bigram")
    print("3. Trigram")
    ngram_choice = input("Choice (1/2/3) : ").strip()
    ngram_map = {'1': 'unigram', '2': 'bigram', '3': 'trigram'}
    ngram_mode = ngram_map.get(ngram_choice, 'unigram')

    miner = TextMiner('.patagonia/patagonia_dataset.xlsx', ngram_type=ngram_mode, normalization=norm_mode) #à remplacer
    #Exécution des étapes principales
    if miner.load_and_process():
        miner.show_word_frequencies()
        miner.vectorize_tfidf_manual(n_docs=5)
        miner.show_elbow_method(max_k=15)
        
        try:
            k = int(input("\n>> Desired number of clusters (e.g., 5) : ")) #Demande le nombre de clusters à l'utilisateur
        except:
            k = 5
        
        miner.perform_clustering(n_clusters=k) #Exécution du clustering
        
        output_name = f".patagonia/patagonia_products_clustered_{ngram_mode}.xlsx" # à remplacer
        miner.save_results(output_name) #Sauvegarde des résultats
        
        miner.visualize_pca() #Visualisation PCA
        

        gephi_choice = input("\n>> Export data to Gephi? (y/n) : ").strip().lower() #Demande si l'utilisateur veut exporter les données pour Gephi
        
        if gephi_choice == 'y': #Si oui, demande les paramètres d'exportation
            # Demande le seuil de similarité
            try:
                threshold = float(input(">> Similarity threshold (0.0-1.0, default=0.20) : ").strip() or "0.20")
                threshold = max(0.0, min(1.0, threshold))  # Clamp entre 0 et 1
            except:
                threshold = 0.20
            
            # Demande le nom de la colonne contenant les noms des produits
            name_col = input(">> Column name for product labels (default='name') : ").strip() or 'name'
            
            # Demande le répertoire de sortie
            output_dir = input(">> Output directory (default='.patagonia') : ").strip() or '.patagonia'
            
            # Export vers Gephi
            miner.export_to_gephi(
                output_dir=output_dir,
                name_column=name_col,
                similarity_threshold=threshold
            )
            
            # Option d'export de la matrice complète de similarité
            matrix_choice = input("\n>> Export full similarity matrix to Excel? (y/n) : ").strip().lower() #Demande si l'utilisateur veut exporter la matrice de similarité complète
            if matrix_choice == 'y':
                matrix_file = os.path.join(output_dir, f"similarity_matrix_{ngram_mode}.xlsx") 
                miner.export_similarity_matrix(output_file=matrix_file)


        token_choice = input("\n>> Export all tokens with frequencies to Excel? (y/n) : ").strip().lower() #Demande si l'utilisateur veut exporter les fréquences des tokens
        
        if token_choice == 'y':
            token_output_dir = input(">> Output directory (default='.patagonia') : ").strip() or '.patagonia'
            token_file = os.path.join(token_output_dir, f"token_frequencies_patagonia_{ngram_mode}.xlsx")
            miner.export_token_frequencies(output_file=token_file)
        
        print("\nFINISHED")
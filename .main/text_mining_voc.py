import pandas as pd #Pour les tables de données
import re #Pour le Regex
import nltk 
from nltk.corpus import stopwords #Pour les stopwords pré-définis
from nltk.stem import PorterStemmer, WordNetLemmatizer #Pour le stemming et lemmatization
from nltk.tokenize import word_tokenize #Pour la tokenization
from nltk.util import ngrams #Pour les n-grams
#S'assure que toutes les ressources NLTK nécessaires sont téléchargées
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
# Classe principale pour l'analyse des produits
class ProductAnalyzer:
    def __init__(self, file_path, column_name='description_clean', ngram_type='unigram', normalization='lemmatization'): #Initialisation des paramètres
        self.file_path = file_path      
        self.column_name = column_name
        self.ngram_type = ngram_type #On travaille en unigram 
        self.normalization = normalization #On travaille en lemmatization car cela fait plus de sens et c'est plus précis
        
        if self.normalization == 'stemming':
            self.stemmer = PorterStemmer()
        elif self.normalization == 'lemmatization':
            self.lemmatizer = WordNetLemmatizer() #Initialisation du lemmatizer, ce qu'on fait ici
        
        self.stop_words = set(stopwords.words('english')) #Utilisation des stopwords en anglais
        
        # Détermination du vocabulaire par catégorie : les mots qu'on va enlever dans chacune des catégories
        self.categories_keywords = {
            'ESG_DURABILITE': {
                'activism', 'bcome', 'biosoft', 'blended', 'bluesign', 'carbon', 
                'certified', 'chain', 'circular', 'climate', 'corp', 'cottonrecycled', 
                'downcycled', 'eco', 'economy', 'ecosystem', 'efficiency', 
                'emissions', 'environment', 'environmental', 'ethical', 'fair', 
                'fair trade', 'fishery', 'footprint', 'fossil', 'fsc', 'gots', 
                'greenhouse', 'grs', 'hemprecycled', 'incinerators', 'initiatives', 
                'labor', 'landfills', 'liters', 'net', 'netplus', 'netting', 'ocean', 
                'oceancycle', 'ocs', 'organic', 'people', 'peta', 'pfas', 
                'pfcspfas', 'planet', 'plastic', 'pollution', 'postconsumer', 'rcs', 
                'recyc', 'recycled', 'regenerative', 'renewable', 'resources', 
                'responsib', 'responsibilitee', 'responsible', 'rubber', 
                'rwscertified', 'saved', 'sewn', 'social', 'solar', 'standard', 
                'stewardship', 'supply', 'sustainab', 'traceab', 'traceable', 
                'trade', 'used', 'vegan', 'waste', 'wastewater', 'worker'
            },
            'TECHNIQUE_PHYSIQUE': {
                'abrasionresistant', 'adjustablesnap', 'ankle', 'articulated', 
                'backzip', 'baffle', 'baffled', 'bifit', 'binding', 'branding', 
                'breathable', 'button', 'buttonclosure', 'buttoned', 'buttonfront', 
                'buttonsnap', 'buttonthrough', 'chin', 'closerfitting', 'closure', 
                'collar', 'collared', 'compressible', 'control', 'cropped', 'cuff', 
                'cufftothigh', 'dobby', 'doublecuff', 'doublesnap', 'drawcord', 
                'drawcordadjustable', 'drawcords', 'drawstring', 'dropin', 
                'dropped', 'droptail', 'durable', 'dwr', 'elastic', 'elasticated', 
                'elasticized', 'embroidery', 'fastening', 'fill', 'filling', 
                'fillpower', 'finish', 'fit', 'fitted', 'fivepocket', 'flap', 
                'flatseam', 'foursnap', 'frontzip', 'fullzip', 'gasket', 'glove', 
                'glovefriendly', 'guard', 'gusset', 'gusseted', 'h2no', 
                'halfelastic', 'halfzip', 'heiq', 'hem', 'hemline', 'high', 'hood', 
                'hooded', 'hoodie', 'hoodless', 'hoody', 'hookandloop', 'inseam', 
                'insole', 'insulated', 'knee', 'laces', 'layer', 'layering', 'light', 
                'lightweight', 'liner', 'linerfree', 'linerless', 'lining', 'loft', 
                'longsleeved', 'loop', 'loose', 'metalbutton', 'mid', 'midi', 
                'midlayer', 'midsole', 'mobility', 'mock', 'moisturewicking', 
                'neck', 'odor', 'onseam', 'outseam', 'oversized', 'packable', 
                'parka', 'placket', 'pocket', 'pocketrouted', 'print', 'pure', 
                'quarterzip', 'quilt', 'quilting', 'raglan', 'regular', 
                'regularfit', 'relaxed', 'relaxedfit', 'repellent', 'reversible', 
                'ribbed', 'ribknit', 'rise', 'round', 'scuff', 'seam', 'seaming', 
                'seamless', 'seamsealed', 'securezip', 'shankbutton', 'shell', 
                'shirttail', 'shortsleeved', 'silhouette', 'singlebutton', 
                'singleseam', 'singlesnap', 'sleeve', 'sleeveless', 'sleeves', 
                'slim', 'slimfit', 'slimfitting', 'slimzip', 'snap', 
                'snapadjustable', 'snapclosure', 'snapfront', 'snaponoff', 'snapt', 
                'snaptab', 'sole', 'standup', 'stoppers', 'storm', 'straight', 
                'stretch', 'stretchy', 'taped', 'terryloop', 'threequartersleeved', 
                'tophem', 'trims', 'twosnap', 'ultralight', 'unbuttoned', 
                'verticalzip', 'verticalzippered', 'waist', 'waistband', 
                'waistbandclosure', 'waistbelt', 'warmth', 'water-repellent', 
                'waterproof', 'waterproofbreathable', 'weatherresistant', 'wick', 
                'wicking', 'wide', 'windproof', 'yoke', 'zip', 'zipfly', 'zipneck', 
                'zipout', 'zipped', 'zipper', 'zippered', 'zipperfly', 'zipsecured', 
                'zipthrough'
            },
            'MATERIAUX_TEXTILES': {
                'acetate', 'airmesh', 'baggies', 'bio', 'bottles', 'brushedtricot', 
                'canvas', 'capilene', 'cashmere', 'chiffon', 'coating', 'corduroy', 
                'corozo', 'cotton', 'cottonelastane', 'cottonrecycled', 'crepe', 
                'denim', 'denimstyle', 'doublefabric', 'down', 'downdrift', 
                'downinsulated', 'downlike', 'duck', 'ecovero', 'elastane', 
                'elasthane', 'elasticized', 'elastomultiester', 'eucotton', 'eva', 
                'fabric', 'fabricstrap', 'feather', 'fiber', 'fibre', 'flannel', 
                'fleece', 'fleecelike', 'fleecelined', 'gore-tex', 'gridfleece', 
                'heather', 'hemp', 'herringbone', 'insulation', 'jacquard', 
                'jersey', 'jerseyknit', 'knit', 'knitfleece', 'knitted', 'lace', 
                'laminate', 'laminated', 'leather', 'lenzing', 'linen', 'lyocell', 
                'material', 'materials', 'membrane', 'merino', 'mesh', 'meshback', 
                'meshlined', 'microfiber', 'microfleece', 'microfleecelined', 
                'microgridfleece', 'modal', 'mohair', 'nylon', 'nylonbound', 
                'nyloncoated', 'nylonelastane', 'pertex', 'pet', 'pile', 'plaid', 
                'plumafill', 'polartec', 'polyester', 'polyesterelastane', 
                'polyestermesh', 'polyurethane', 'powermesh', 'primaloft', 'puff', 
                'pulp', 'rib', 'ribbing', 'ripstop', 'selffabric', 'shearling', 
                'sorona', 'spandex', 'stretchmesh', 'suede', 'sweat', 'synchilla', 
                'synthetic', 'taffeta', 'tencel', 'textile', 'textile-to-textile', 
                'thermogreen', 'tpu', 'tpufilm', 'tricot', 'tricotlined', 'twill', 
                'twilllined', 'velour', 'velvet', 'viscose', 'wool', 'woolblend', 
                'woven', 'yarn', 'yulex'
            }
        }

    def preprocess(self, text): # Prétraitement du texte : tokenization,normalisation, choix des n-grams
        if pd.isna(text) or text == "":
            return []
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
        
        if self.ngram_type == 'bigram':
            return list(ngrams(cleaned, 2))
        elif self.ngram_type == 'trigram':
             return list(ngrams(cleaned, 3))
        
        return cleaned

    def classify_token(self, token): # Vérifie à quelle catégorie appartient le token

        if isinstance(token, tuple): # Pour bigrams et trigrams
            token = " ".join(token)
            
        if token in self.categories_keywords['ESG_DURABILITE']: #Vérifie si le token est dans la catégorie ESG_DURABILITE
            return 'ESG_DURABILITE'
        if token in self.categories_keywords['TECHNIQUE_PHYSIQUE']: #Vérifie si le token est dans la catégorie TECHNIQUE_PHYSIQUE
            return 'TECHNIQUE_PHYSIQUE'
        if token in self.categories_keywords['MATERIAUX_TEXTILES']: #Vérifie si le token est dans la catégorie MATERIAUX_TEXTILES
            return 'MATERIAUX_TEXTILES'
        return 'OTHER' # Catégorie par défaut

    def run_analysis(self): #Exécute l'analyse principale
        print("\nLoading Data...")
        try:
            df = pd.read_excel(self.file_path)
        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' not found.") #Gestion d'erreur si le fichier n'est pas trouvé
            return
            
        if self.column_name not in df.columns:
            print(f"Error: Column '{self.column_name}' not found in Excel file.") #Gestion d'erreur si la colonne n'est pas trouvée
            print(f"Available columns: {list(df.columns)}")
            return

        df = df.dropna(subset=[self.column_name]).reset_index(drop=True) #Supprime les lignes avec des valeurs manquantes dans la colonne spécifiée
        print(f"Processing {len(df)} products...")

        results = [] #Liste pour stocker les résultats de chaque produit

        for index, row in df.iterrows(): 

            tokens = self.preprocess(row[self.column_name]) 
            total_items = len(tokens) #Nombre total de tokens après prétraitement
            
            counts = {
                'ESG_DURABILITE': 0,
                'TECHNIQUE_PHYSIQUE': 0,
                'MATERIAUX_TEXTILES': 0,
                'OTHER': 0
            } #Initialisation des compteurs pour chaque catégorie
            
            
            for token in tokens:
                category = self.classify_token(token)
                counts[category] += 1 #Incrémente le compteur de la catégorie correspondante
          
            if total_items > 0: # Calcule les pourcentage d'appartenance aux catégories
                pct_esg = (counts['ESG_DURABILITE'] / total_items) * 100
                pct_tech = (counts['TECHNIQUE_PHYSIQUE'] / total_items) * 100
                pct_mat = (counts['MATERIAUX_TEXTILES'] / total_items) * 100
                pct_other = (counts['OTHER'] / total_items) * 100
            else:
                pct_esg = pct_tech = pct_mat = pct_other = 0.0 #Évite la division par zéro

    
            product_name = row['name'] if 'name' in row else f"Product_{index}" #Nom du produit
            source_file = row['source_file'] if 'source_file' in row else "Unknown" #Nom du fichier source

            results.append({
                'Product Name': product_name,
                'Source File': source_file,
                'Total Keywords': total_items,
                'ESG (Count)': counts['ESG_DURABILITE'],
                'ESG (%)': round(pct_esg, 2),
                'Technical (Count)': counts['TECHNIQUE_PHYSIQUE'],
                'Technical (%)': round(pct_tech, 2),
                'Material (Count)': counts['MATERIAUX_TEXTILES'],
                'Material (%)': round(pct_mat, 2),
                'Other (Count)': counts['OTHER'],
                'Other (%)': round(pct_other, 2)
            }) #Ajoute les résultats du produit à la liste

        result_df = pd.DataFrame(results) #Convertit la liste des résultats en tableau 
        
        print("\nAverage Scores Per source_file") # Affiche les scores moyens par fichier source
        if 'Source File' in result_df.columns:
            summary_df = result_df.groupby('Source File')[
                ['ESG (%)', 'Technical (%)', 'Material (%)', 'Other (%)'] #Calcule la moyenne des pourcentages par fichier source
            ].mean().reset_index()
            print(summary_df.round(2)) #Arrondit les valeurs à 2 décimales pour une meilleure lisibilité
        
        output_file = "all_products_category_analysis.xlsx" # à remplacer
        result_df.to_excel(output_file, index=False) #Export des résultats dans un fichier Excel
        
        print("\n--- ANALYSIS COMPLETE ---")
        print(f">> Results saved to: {output_file}")

if __name__ == "__main__":
    print("\nCONFIGURATION")
    
    print("1. Lemmatization") #Choix entre Lemmatisation et Racinisation
    print("2. Stemming")
    norm_choice = input("Choice (1/2) : ").strip()
    norm_mode = 'lemmatization' if norm_choice == '1' else 'stemming'

    print("\n1. Unigram (Single words)") #Choix du type de n-gram
    print("2. Bigram (Pairs of words)")
    print("3. Trigram (Groups of 3)")
    ngram_choice = input("Choice (1/2/3) : ").strip()
    ngram_map = {'1': 'unigram', '2': 'bigram', '3': 'trigram'}
    ngram_mode = ngram_map.get(ngram_choice, 'unigram')

    file_path = 'all_products_combined.xlsx' # à remplacer
    
    analyzer = ProductAnalyzer(
        file_path=file_path, 
        column_name='description_clean',
        ngram_type=ngram_mode, 
        normalization=norm_mode
    ) #Initialisation de l'analyseur de produits
    
    analyzer.run_analysis() #Exécution de l'analyse
    
    print("\nFINISHED")
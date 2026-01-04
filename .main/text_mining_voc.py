import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams 

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

class ProductAnalyzer:
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
        
        # Détermination du vocabulaire par catégorie
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

    def preprocess(self, text):
        if pd.isna(text) or text == "":
            return []
            
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
        
        if self.ngram_type == 'bigram':
            return list(ngrams(cleaned, 2))
        elif self.ngram_type == 'trigram':
             return list(ngrams(cleaned, 3))
        
        return cleaned

    def classify_token(self, token): # Vérifie à quelle catégorie appartient le token

        if isinstance(token, tuple): # Pour bigrams et trigrams
            token = " ".join(token)
            
        if token in self.categories_keywords['ESG_DURABILITE']:
            return 'ESG_DURABILITE'
        if token in self.categories_keywords['TECHNIQUE_PHYSIQUE']:
            return 'TECHNIQUE_PHYSIQUE'
        if token in self.categories_keywords['MATERIAUX_TEXTILES']:
            return 'MATERIAUX_TEXTILES'
        return 'OTHER' # Catégorie par défaut

    def run_analysis(self):
        print("\nLoading Data...")
        try:
            df = pd.read_excel(self.file_path)
        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' not found.")
            return
            
        if self.column_name not in df.columns:
            print(f"Error: Column '{self.column_name}' not found in Excel file.")
            print(f"Available columns: {list(df.columns)}")
            return

        df = df.dropna(subset=[self.column_name]).reset_index(drop=True)
        print(f"Processing {len(df)} products...")

        results = []

        for index, row in df.iterrows():

            tokens = self.preprocess(row[self.column_name])
            total_items = len(tokens)
            
            counts = {
                'ESG_DURABILITE': 0,
                'TECHNIQUE_PHYSIQUE': 0,
                'MATERIAUX_TEXTILES': 0,
                'OTHER': 0
            }
            
            
            for token in tokens:
                category = self.classify_token(token)
                counts[category] += 1
          
            if total_items > 0: # Calcule les pourcentage d'appartenance aux catégories
                pct_esg = (counts['ESG_DURABILITE'] / total_items) * 100
                pct_tech = (counts['TECHNIQUE_PHYSIQUE'] / total_items) * 100
                pct_mat = (counts['MATERIAUX_TEXTILES'] / total_items) * 100
                pct_other = (counts['OTHER'] / total_items) * 100
            else:
                pct_esg = pct_tech = pct_mat = pct_other = 0.0

    
            product_name = row['name'] if 'name' in row else f"Product_{index}"
            source_file = row['source_file'] if 'source_file' in row else "Unknown"

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
            })

        result_df = pd.DataFrame(results)
        
        print("\nAverage Scores Per source_file")
        if 'Source File' in result_df.columns:
            summary_df = result_df.groupby('Source File')[
                ['ESG (%)', 'Technical (%)', 'Material (%)', 'Other (%)']
            ].mean().reset_index()
            print(summary_df.round(2))
        
        output_file = "all_products_category_analysis.xlsx" # à remplacer
        result_df.to_excel(output_file, index=False)
        
        print("\n--- ANALYSIS COMPLETE ---")
        print(f">> Results saved to: {output_file}")

if __name__ == "__main__":
    print("\nCONFIGURATION")
    
    print("1. Lemmatization")
    print("2. Stemming")
    norm_choice = input("Choice (1/2) : ").strip()
    norm_mode = 'lemmatization' if norm_choice == '1' else 'stemming'

    print("\n1. Unigram (Single words)")
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
    )
    
    analyzer.run_analysis()
    
    print("\nFINISHED")
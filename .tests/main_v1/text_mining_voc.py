import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- NLTK Resource Management ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print(">> Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')

class ProductAnalyzer:
    def __init__(self, file_path, column_name='description'):
        self.file_path = file_path
        self.column_name = column_name
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # --- UPDATED KEYWORD DEFINITIONS ---
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
        """Cleans and tokenizes text."""
        text = str(text).lower()
        # Remove specific boilerplate (simplified)
        text = re.sub(r'\[.*?\]', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        tokens = word_tokenize(text)
        cleaned = []
        for t in tokens:
            if t not in self.stop_words and len(t) > 2:
                word_final = self.lemmatizer.lemmatize(t)
                cleaned.append(word_final)
        return cleaned

    def classify_token(self, token):
        """Determines the category of a single token."""
        if token in self.categories_keywords['ESG_DURABILITE']:
            return 'ESG_DURABILITE'
        if token in self.categories_keywords['TECHNIQUE_PHYSIQUE']:
            return 'TECHNIQUE_PHYSIQUE'
        if token in self.categories_keywords['MATERIAUX_TEXTILES']:
            return 'MATERIAUX_TEXTILES'
        return 'OTHER'

    def run_analysis(self):
        print("\n--- LOADING DATA ---")
        try:
            df = pd.read_excel(self.file_path)
        except FileNotFoundError:
            print("Error: File not found.")
            return
            
        df = df.dropna(subset=[self.column_name]).reset_index(drop=True)
        print(f"Processing {len(df)} products...")

        results = []

        for index, row in df.iterrows():
            # Get and preprocess text
            tokens = self.preprocess(row[self.column_name])
            total_words = len(tokens)
            
            # Initialize counters
            counts = {
                'ESG_DURABILITE': 0,
                'TECHNIQUE_PHYSIQUE': 0,
                'MATERIAUX_TEXTILES': 0,
                'OTHER': 0
            }
            
            # Count words per category
            for token in tokens:
                category = self.classify_token(token)
                counts[category] += 1
            
            # Calculate percentages
            if total_words > 0:
                pct_esg = (counts['ESG_DURABILITE'] / total_words) * 100
                pct_tech = (counts['TECHNIQUE_PHYSIQUE'] / total_words) * 100
                pct_mat = (counts['MATERIAUX_TEXTILES'] / total_words) * 100
                pct_other = (counts['OTHER'] / total_words) * 100
            else:
                pct_esg = pct_tech = pct_mat = pct_other = 0.0

            # Prepare row output
            product_name = row['name'] if 'name' in row else f"Product_{index}"
            
            # --- MODIFICATION: Extract source_file ---
            source_file = row['source_file'] if 'source_file' in row else "Unknown"

            results.append({
                'Product Name': product_name,
                'Source File': source_file,  # Add source_file to the output row
                'Total Keywords': total_words,
                'ESG (Count)': counts['ESG_DURABILITE'],
                'ESG (%)': round(pct_esg, 2),
                'Technical (Count)': counts['TECHNIQUE_PHYSIQUE'],
                'Technical (%)': round(pct_tech, 2),
                'Material (Count)': counts['MATERIAUX_TEXTILES'],
                'Material (%)': round(pct_mat, 2),
                'Other (Count)': counts['OTHER'],
                'Other (%)': round(pct_other, 2)
            })

        # Convert to DataFrame
        result_df = pd.DataFrame(results)
        
        # --- MODIFICATION: Calculate Average by Source File ---
        print("\n--- AVERAGE SCORES PER SOURCE FILE ---")
        if 'Source File' in result_df.columns:
            # Group by Source File and calculate mean for the percentage columns
            summary_df = result_df.groupby('Source File')[
                ['ESG (%)', 'Technical (%)', 'Material (%)', 'Other (%)']
            ].mean().reset_index()
            
            # Print the summary in a readable format
            print(summary_df.round(2))
        
        # Save to Excel
        output_file = "products_category_analysis.xlsx"
        result_df.to_excel(output_file, index=False)
        
        print("\n--- ANALYSIS COMPLETE ---")
        # print(result_df) # Optional: comment out if dataframe is too large
        print(f"\n>> Results saved to: {output_file}")

# --- EXECUTION ---
if __name__ == "__main__":
    # Replace 'cleaned_all_products.xlsx' with your actual filename
    analyzer = ProductAnalyzer('cleaned_all_products.xlsx', column_name='description')
    analyzer.run_analysis()
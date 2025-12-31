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
        # These lists include terms from the original list + terms found in the Excel file
        self.categories_keywords = {
            'ESG_DURABILITE': {
                'bluesign', 'certified', 'cottonrecycled', 'eco', 'ecosystem', 
                'environment', 'environmental', 'ethical', 'fair', 'fair trade', 
                'footprint', 'hemprecycled', 'netplus', 'ocean', 'oceancycle', 
                'organic', 'pfas', 'pfcspfas', 'pollution', 'postconsumer', 
                'recyc', 'recycled', 'responsib', 'responsibilitee', 'responsible', 
                'rwscertified', 'sustainab', 'traceab', 'traceable', 'trade', 
                'waste', 'wastewater'
            },
            'TECHNIQUE_PHYSIQUE': {
                'adjustablesnap', 'articulated', 'backzip', 'baffle', 'baffled', 
                'bifit', 'binding', 'button', 'buttonclosure', 'buttoned', 
                'buttonfront', 'buttonsnap', 'buttonthrough', 'closerfitting', 
                'closure', 'collar', 'collared', 'cuff', 'cufftothigh', 
                'doublecuff', 'doublesnap', 'drawcord', 'drawcordadjustable', 
                'drawcords', 'elastic', 'elasticized', 'fit', 'fitted', 
                'fivepocket', 'flatseam', 'foursnap', 'frontzip', 'fullzip', 
                'glove', 'glovefriendly', 'gusset', 'gusseted', 'halfelastic', 
                'halfzip', 'hem', 'hemline', 'hood', 'hooded', 'hoodless', 
                'hoody', 'hookandloop', 'inseam', 'liner', 'linerfree', 
                'linerless', 'longsleeved', 'loop', 'metalbutton', 'onseam', 
                'outseam', 'parka', 'pocket', 'pocketrouted', 'quarterzip', 
                'regularfit', 'relaxedfit', 'ribknit', 'scuff', 'seam', 
                'seaming', 'seamless', 'seamsealed', 'securezip', 'shankbutton', 
                'shortsleeved', 'silhouette', 'singlebutton', 'singleseam', 
                'singlesnap', 'sleeve', 'slimfit', 'slimfitting', 'slimzip', 
                'snap', 'snapadjustable', 'snapclosure', 'snapfront', 'snaponoff', 
                'snapt', 'snaptab', 'terryloop', 'threequartersleeved', 'tophem', 
                'twosnap', 'unbuttoned', 'verticalzip', 'verticalzippered', 
                'waistband', 'waistbandclosure', 'zip', 'zipfly', 'zipneck', 
                'zipout', 'zipped', 'zipper', 'zippered', 'zipperfly', 
                'zipsecured', 'zipthrough'
            },
            'MATERIAUX_TEXTILES': {
                'airmesh', 'brushedtricot', 'canvas', 'cotton', 'cottonelastane', 
                'cottonrecycled', 'denim', 'denimstyle', 'doublefabric', 'down', 
                'downdrift', 'downinsulated', 'downlike', 'elastane', 'elasticized', 
                'fabric', 'fabricstrap', 'fiber', 'fibre', 'fleece', 'fleecelike', 
                'fleecelined', 'gridfleece', 'insulation', 'jersey', 'jerseyknit', 
                'knitfleece', 'mesh', 'meshback', 'meshlined', 'microfiber', 
                'microfleece', 'microfleecelined', 'microgridfleece', 'nylon', 
                'nylonbound', 'nyloncoated', 'nylonelastane', 'polyester', 
                'polyesterelastane', 'polyestermesh', 'powermesh', 'ripstop', 
                'selffabric', 'spandex', 'stretchmesh', 'synthetic', 'taffeta', 
                'textile', 'tricot', 'tricotlined', 'twill', 'twilllined', 
                'wool', 'woolblend'
            }
        }
        
        # Flatten sets for faster single-word lookup if needed, 
        # though we iterate through tokens and check membership.

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
        return cleaned # Returns list of tokens

    def classify_token(self, token):
        """Determines the category of a single token."""
        # Priority check: ESG > Technical > Material (to handle overlaps if any)
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
            # Try to find a name column, default to index if missing
            product_name = row['name'] if 'name' in row else f"Product_{index}"
            
            results.append({
                'Product Name': product_name,
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
        
        # Save to Excel
        output_file = "products_category_analysis.xlsx"
        result_df.to_excel(output_file, index=False)
        
        print("\n--- ANALYSIS COMPLETE ---")
        print(result_df)
        print(f"\n>> Results saved to: {output_file}")

# --- EXECUTION ---
if __name__ == "__main__":
    # Replace 'cleaned_all_products.xlsx' with your actual filename
    analyzer = ProductAnalyzer('cleaned_all_products.xlsx', column_name='description')
    analyzer.run_analysis()
import pypdf
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Initialisation NLTK ---
def download_nltk_data():
    resources = ['stopwords', 'punkt', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)

# Téléchargement des ressources nécessaires
download_nltk_data()

def nettoyer_texte(texte, langue='english'):
    """
    Nettoie le texte : minuscule, suppression ponctuation/chiffres, retrait des stopwords.
    """
    if not texte:
        return ""

    # 1. Conversion en minuscules
    texte = texte.lower()

    # 2. Suppression des caractères non alphabétiques (garde uniquement lettres et espaces)
    # Note : pour l'anglais/français, on garde a-z. J'ai ajouté les accents au cas où.
    texte = re.sub(r'[^a-zà-ÿ\s]', '', texte)

    # 3. Tokenization (découpage en mots)
    tokens = word_tokenize(texte, language=langue)

    # 4. Filtrage des stopwords (mots vides comme "the", "is", "le", "de")
    stop_words = set(stopwords.words(langue))
    mots_filtres = [mot for mot in tokens if mot not in stop_words and len(mot) > 1]

    # 5. Reconstitution du texte
    return " ".join(mots_filtres)

def convertir_pdf_en_txt_propre(chemin_pdf, chemin_sortie, langue='english'):
    print(f"--- Traitement du fichier : {chemin_pdf} ---")
    
    texte_complet = ""
    
    try:
        reader = pypdf.PdfReader(chemin_pdf)
        nombre_pages = len(reader.pages)
        print(f"Nombre de pages détectées : {nombre_pages}")

        for i, page in enumerate(reader.pages):
            contenu = page.extract_text()
            if contenu:
                # Remplace les sauts de ligne par des espaces pour éviter de couper les phrases
                texte_complet += contenu.replace('\n', ' ') + " "
            
            # Petit indicateur de progression
            if (i + 1) % 10 == 0:
                print(f"Page {i + 1} traitée...")

    except Exception as e:
        print(f"Erreur lecture PDF : {e}")
        return

    print("Nettoyage du texte en cours (suppression stopwords, ponctuation)...")
    texte_final = nettoyer_texte(texte_complet, langue)

    # Sauvegarde
    try:
        with open(chemin_sortie, 'w', encoding='utf-8') as f:
            f.write(texte_final)
        
        print(f"--- Terminé ! ---")
        print(f"Fichier sauvegardé sous : {chemin_sortie}")
        print(f"Aperçu du début : {texte_final[:200]}...")
        
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier : {e}")

if __name__ == "__main__":
    # Nom exact du fichier téléchargé
    input_file = "Armedangels/Social-plan-link-8.pdf" 
    output_file = "report_armedangels_2021.txt"
    
    # Le rapport est en anglais, on utilise donc 'english'
    convertir_pdf_en_txt_propre(input_file, output_file, langue='english')
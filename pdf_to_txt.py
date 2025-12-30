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

download_nltk_data()

def reparer_texte_espaces(texte):
    """
    Répare le texte où les lettres sont séparées par des espaces.
    Exemple: "T o  h e l p" -> "To help"
    """
    if not texte:
        return ""

    # 1. On remplace les "vrais" espaces (souvent doubles ou tabulations dans les PDF espacés)
    # par un marqueur temporaire unique "###".
    # On cherche 2 espaces ou plus consécutifs.
    texte_temp = re.sub(r'\s{2,}', '###', texte)
    
    # 2. Si le texte ne contenait pas de double espace, il est possible que pypdf
    # ait tout extrait avec un seul espace. On essaie une regex plus intelligente :
    # On supprime l'espace UNIQUEMENT s'il est entre deux lettres (ex: "T o").
    if '###' not in texte_temp:
        # Regex: (Lettre) + Espace + (Lettre) -> On garde les lettres, on vire l'espace
        # Attention: c'est une méthode de secours, elle peut être imparfaite.
        texte_repare = re.sub(r'(?<=[a-zA-Z])\s(?=[a-zA-Z])', '', texte)
        return texte_repare

    # 3. On supprime tous les espaces simples restants (ceux entre les lettres "T o")
    texte_temp = texte_temp.replace(' ', '')

    # 4. On remet les vrais espaces (via le marqueur ###)
    texte_final = texte_temp.replace('###', ' ')
    
    return texte_final

def nettoyer_et_convertir_pdf(chemin_pdf, chemin_sortie, langue='french'):
    print(f"--- Traitement du fichier : {chemin_pdf} ---")
    
    # 1. Extraction du texte brut
    texte_brut = ""
    try:
        reader = pypdf.PdfReader(chemin_pdf)
        for page in reader.pages:
            contenu = page.extract_text()
            if contenu:
                # On remplace les sauts de ligne par des espaces doubles pour aider la détection
                texte_brut += contenu.replace('\n', '  ') + "  "
    except Exception as e:
        print(f"Erreur lecture PDF : {e}")
        return

    # 2. Réparation des mots espacés (L a  m a i s o n -> La maison)
    print("Réparation du texte espacé...")
    texte_repare = reparer_texte_espaces(texte_brut)

    # 3. Conversion minuscule et nettoyage caractères
    texte = texte_repare.lower()
    # On garde lettres et espaces
    texte = re.sub(r'[^a-zà-ÿ\s]', '', texte)

    # 4. Tokenization (Découpage en mots)
    tokens = word_tokenize(texte, language=langue)

    # 5. Filtrage des stopwords
    stop_words = set(stopwords.words(langue))
    mots_filtres = [mot for mot in tokens if mot not in stop_words and len(mot) > 1]

    # 6. Reconstitution
    texte_final = " ".join(mots_filtres)

    # 7. Sauvegarde
    with open(chemin_sortie, 'w', encoding='utf-8') as f:
        f.write(texte_final)
    
    print(f"Terminé ! Résultat sauvegardé dans : {chemin_sortie}")
    # Affiche un aperçu pour vérifier
    print(f"Aperçu du début : {texte_final[:200]}...")

if __name__ == "__main__":
    input_file = "patagonia-progress-report-2025.pdf" 
    output_file = "patagonia-progress-report-2025.txt"
    
    # Utilisation de 'english' pour Patagonia
    nettoyer_et_convertir_pdf(input_file, output_file, langue='english')
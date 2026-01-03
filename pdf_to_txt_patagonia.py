import pypdf #Librairie pour lire les fichiers PDF
import re #Librairie pour les expressions régulières = nettoyage de texte
import nltk #Librairie de traitement de texte  
from nltk.corpus import stopwords #On importe la liste des stopwords = le, la , les, un, une, des, etc.
from nltk.tokenize import word_tokenize #Pour découper le texte en mots

# --- Initialisation NLTK ---
def download_nltk_data():
    resources = ['stopwords', 'punkt', 'punkt_tab'] #Liste des ressources nécessaires, note : 'punkt_tab' pour couper les textes avec tabulations
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}') #Vérifie si la ressource est déjà téléchargée
        except LookupError: #Si la ressource n'est pas trouvée, on la télécharge après vérification dans un autre dossier
            try:
                nltk.data.find(f'corpora/{resource}') 
            except LookupError:
                nltk.download(resource, quiet=True)

download_nltk_data() #Appel de la fonction = On télécharge les ressources NLTK nécessaires 

def reparer_texte_espaces(texte): 
    """
    Répare le texte où les lettres sont séparées par des espaces.
    Exemple: "T o  h e l p" -> "To help"
    """
    if not texte:
        return "" # Si le texte est vide, on retourne une chaîne vide

    #1. On remplace les "vrais" espaces (souvent doubles ou tabulations dans les PDF espacés)
    #par un marqueur temporaire unique "###".
    #On cherche 2 espaces ou plus consécutifs.
    texte_temp = re.sub(r'\s{2,}', '###', texte)
    
    #2. Si le texte ne contenait pas de double espace, il est possible que pypdf
    #ait tout extrait avec un seul espace. On essaie une regex plus intelligente :
    #On supprime l'espace UNIQUEMENT s'il est entre deux lettres (ex: "T o").
    if '###' not in texte_temp:
        #Regex: (Lettre) + Espace + (Lettre) -> On garde les lettres, on vire l'espace
        texte_repare = re.sub(r'(?<=[a-zA-Z])\s(?=[a-zA-Z])', '', texte)
        return texte_repare

    #3. On supprime tous les espaces simples restants (ceux entre les lettres "T o")
    texte_temp = texte_temp.replace(' ', '')

    #4. On remet les vrais espaces (via le marqueur ###)
    texte_final = texte_temp.replace('###', ' ')
    
    return texte_final #On retourne le texte "réparé"

def nettoyer_et_convertir_pdf(chemin_pdf, chemin_sortie, langue='french'): #Fonction principale de nettoyage et conversion PDF -> TXT
    print(f"--- Traitement du fichier : {chemin_pdf} ---")
    
    #1. Extraction du texte brut
    texte_brut = ""
    try:
        reader = pypdf.PdfReader(chemin_pdf) #Ouvre le PDF
        for page in reader.pages: #Parcours chaque page
            contenu = page.extract_text() #Extrait le texte de la page
            if contenu:
                # On remplace les sauts de ligne par des espaces doubles pour aider la détection
                # Un PDF coupe les lignes n'importe où. Si on recolle de manière brut, la fin d'une ligne va se coller au début de la suivante ("mot\nSuite" -> "motSuite").
                # On remplace le saut de ligne (\n) par deux espaces pour éviter ça.
                texte_brut += contenu.replace('\n', '  ') + "  "
    except Exception as e:
        print(f"Erreur lecture PDF : {e}") 
        return

    #2. Réparation des mots espacés (L a  m a i s o n -> La maison) - Appel de la fonction vue plus haut
    print("Réparation du texte espacé...")
    texte_repare = reparer_texte_espaces(texte_brut)

    #3. Conversion minuscule et nettoyage caractères - PATAGONIA et patagonia compteront comme identiques
    texte = texte_repare.lower()
    #On garde les lettres et les espaces par filtrage Regex
    texte = re.sub(r'[^a-zà-ÿ\s]', '', texte)

    #4. Tokenization (Découpage en mots), on transforme les chaines de caractères en liste de mots
    tokens = word_tokenize(texte, language=langue)

    #5. Filtrage des stopwords
    stop_words = set(stopwords.words(langue)) #On charge la liste des stopwords pour la langue choisie
    mots_filtres = [mot for mot in tokens if mot not in stop_words and len(mot) > 1] #On enlève les stopwords et les mots d'une lettre

    #6. Reconstruction du texte final = on recolle les mots avec des espaces pour faire un texte "propre"
    texte_final = " ".join(mots_filtres)

    #7. Sauvegarde dans un fichier texte
    with open(chemin_sortie, 'w', encoding='utf-8') as f:
        f.write(texte_final)
    
    print(f"Terminé ! Résultat sauvegardé dans : {chemin_sortie}")
    # Affiche un aperçu pour vérifier
    print(f"Aperçu du début : {texte_final[:200]}...")

if __name__ == "__main__":
    input_file = "report_ecoalf_2022.pdf" 
    output_file = "report_ecoalf_2022.txt"
    
    #Utilisation de 'english' pour Patagonia
    nettoyer_et_convertir_pdf(input_file, output_file, langue='english')
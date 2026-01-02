import asyncio  #Comme pour le script de collecte des liens, nous utilisons asyncio pour plus d'efficacité.
import pandas as pd    #Va nous servir à créer un fichier Excel à la fin.
from playwright.async_api import async_playwright  #Encore une fois, nous utilisons la version asynchrone de Playwright pour une meilleure performance.
from bs4 import BeautifulSoup  #Nécessaire pour analyser le HTML et extraire les informations.

class ProductDetailScraper:
    def __init__(self, input_file="womens_product_links.txt", output_file="patagonia_women_final.xlsx"):
        self.input_file = input_file   #Nom du fichier contenant les liens à scraper
        self.output_file = output_file  #Nom du fichier Excel de sortie
        self.data = []  #Liste pour stocker les données extraites = Mémoire temporaire avant de sauvegarder dans Excel

###Une page Patagonia typique a une description principale et une section "accordéon" pour les détails. Nous devons séparer ce qui nous sera utile des autres informations.
    async def get_description(self, soup):
        """
        Combines the main description and the accordion content, 
        EXCLUDING the 'fit' section as requested.
        """
        full_text = []  #Liste pour stocker les différentes parties de la description

        #1. Récupération de la description principale
        intro = soup.select_one('div.pdp__content-description') #Sélectionne la partie HTML contenant la description principale
        if intro:                                               #Vérifie si l'élément a bien été trouvé pour éviter un crash
            full_text.append(f"[INTRO]\n{intro.get_text(strip=True)}") #Ajoute le texte nettoyé des espaces superflus à la liste avec un label [INTRO]

        #2. Récupération des détails techniques (excluant la section "coupe" = fit)
        details_wrapper = soup.select_one('div.accordion-group--wrapper') #Sélectionne la section contenant les détails en accordéon
        if details_wrapper:                                               #Vérifie si l'élément a bien été trouvé pour éviter un crash
        #Nettoyage : supprimer la section "fit" si elle existe   
            fit_section = details_wrapper.select_one('div.accordion-group[data-pdp-accordion-fit=""]')
            if fit_section:
                fit_section.decompose() #Supprime la section "fit" du HTML avant d'extraire le texte

           
            raw_text = details_wrapper.get_text(separator='\n', strip=True) #Récupère tout le texte brut avec des sauts de ligne entre les sections
            
        
            clean_text = "\n".join([line for line in raw_text.split('\n') if line.strip()]) #Nettoie le texte en supprimant les lignes vides ou inutiles
            full_text.append(f"\n[DETAILS]\n{clean_text}") #Ajoute le texte nettoyé à la liste avec un label [DETAILS]

        return "\n".join(full_text) #Retourne la description complète en une seule chaîne de caractères
    
### Boucle = pour chaque URL, il navigue, récupère le HTML, le transforme en "soupe", extrait les infos, et stocke les données.
    async def run(self):
        try:
            with open(self.input_file, "r") as f:   #Ouvre le fichier texte en mode lecture
                urls = [line.strip() for line in f.readlines() if line.strip()]  #Nettoie les retours à la ligne (\n) pour avoir des URLs propres
            print(f"Loaded {len(urls)} links from file.") #Affiche le nombre de liens chargés
        except FileNotFoundError: #Gestion d'erreur si le fichier n'existe pas
            print(f"Error: Could not find {self.input_file}. Run the link harvester first!")
            return

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()   #Ouvre un nouvel onglet dans le navigateur Chrome de façon visible "headless=False"

            for i, url in enumerate(urls):  #Boucle sur chaque URL avec son index
                print(f"[{i+1}/{len(urls)}] Scraping: {url}") #Affiche la progression du scraping
                try:
                    #Navigation vers la page produit
                    await page.goto(url, timeout=60000, wait_until="domcontentloaded")  #Navigue vers l'URL avec un timeout de 60 secondes et attend que la page soit chargée
                    
                    #Gestion du pop-up de cookies
                    if i == 0:
                        try: await page.click('#onetrust-accept-btn-handler', timeout=2000)  #Accepte les cookies si le bouton est présent
                        except: pass  #Ignore si le bouton n'est pas trouvé

                    #Récupération du contenu HTML de la page
                    content = await page.content()  #Récupère le HTML complet de la page
                    soup = BeautifulSoup(content, 'html.parser')  #Transforme le HTML en "soupe" pour l'analyse

### Extraction des informations
                    #1. Nom du produit
                    h1 = soup.select_one('h1#product-title')
                    name = h1.get_text(strip=True) if h1 else "N/A"

                    #2. Prix du produit
                    price_span = soup.select_one('span[itemprop="price"]')
                    if not price_span:
                        price_span = soup.select_one('.prices .value')
                    price = price_span.get_text(strip=True) if price_span else "N/A"

                    #3. Liste des catégories ("breadcrumb")- Exemple : Men's > Shop by Category > Fleece > Jackets > Nom du produit
                    breadcrumb = soup.select_one('ol.breadcrumb') #Sélectionne le breadcrumb (Note : on parle de "Fil d'Ariane" en français)
                    if breadcrumb:
                        cat_list = [li.get_text(strip=True) for li in breadcrumb.find_all('li')] #Extrait le texte de chaque élément de la liste
                        category = " > ".join(cat_list) #Joint les catégories avec ">" comme séparateur
                    else:
                        category = "N/A" #Valeur par défaut si le breadcrumb n'est pas trouvé

                    #4. Description complète (appel de la fonction définie plus haut)
                    description = await self.get_description(soup)

                    #Stockage des données extraites dans la liste
                    self.data.append({
                        "Name": name,
                        "Price": price,
                        "Category": category,
                        "Description": description,
                        "URL": url
                    })            #Ajoute un dictionnaire avec les données : nom du produit, prix, catégorie, description, URL

                except Exception as e:
                    print(f"  Error: {e}") #Gestion des erreurs pour chaque URL individuelle

                if (i+1) % 10 == 0:
                    self.save()         #Sauvegarde intermédiaire tous les 10 produits

            await browser.close() #Ferme le navigateur une fois toutes les URLs traitées
            self.save() #Sauvegarde finale
            print("Done!")

    def save(self): #Fonction pour sauvegarder les données dans un fichier Excel
        df = pd.DataFrame(self.data) #Convertit la liste de dictionnaires en un tableau structuré (DataFrame)
        df.to_excel(self.output_file, index=False) #Sauvegarde le DataFrame dans un fichier Excel sans les index
        print(f"  > Saved data to {self.output_file}") #Message de confirmation de sauvegarde

if __name__ == "__main__": #Lancement du script
    scraper = ProductDetailScraper() 
    asyncio.run(scraper.run())
import asyncio #Comme pour le script de collecte des liens, nous utilisons asyncio pour plus d'efficacité.
import pandas as pd #Va nous servir à créer un fichier Excel à la fin.
from playwright.async_api import async_playwright #Encore une fois, nous utilisons la version asynchrone de Playwright pour une meilleure performance.
from bs4 import BeautifulSoup  #Nécessaire pour analyser le HTML et extraire les informations.
import os

INPUT_FILE = ".patagonia/patagonia_women_links.txt" #Nom du fichier contenant les liens à scraper, pouvant être fait pour les hommes et les femmes
OUTPUT_FILE = ".patagonia/patagonia_women_products.xlsx" #Nom du fichier Excel de sortie

#Une page Patagonia typique a une description principale et une section "accordéon" pour les détails. Nous devons séparer ce qui nous sera utile des autres informations.

async def get_raw_description(soup):
    full_text = []  #Liste pour stocker les données extraites = Mémoire temporaire avant de sauvegarder dans Excel
    
    #1. Récupération de la description principale
    intro = soup.select_one('div.pdp__content-description')  #Sélectionne la partie HTML contenant la description principale
    if intro:
        full_text.append(intro.get_text(separator=' ', strip=True)) #Vérifie si l'élément a bien été trouvé pour éviter un crash

    #2. Récupération des détails techniques dans la section accordéon
    details_wrapper = soup.select_one('div.accordion-group--wrapper') #Sélectionne la section contenant les détails en accordéon (technique)
    if details_wrapper:
        # On garde tout le texte brut pour le nettoyer plus tard
        raw_text = details_wrapper.get_text(separator=' ', strip=True)
        full_text.append(raw_text)  # On ajoute le texte brut des détails à la liste

    return " ".join(full_text)


# Boucle = pour chaque URL, il navigue, récupère le HTML, le transforme en "soupe", extrait les infos, et stocke les données.
async def scrape_products():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.") #Gestion d'erreur si le fichier n'existe pas
        return

    with open(INPUT_FILE, "r") as f:  #Ouvre le fichier texte en mode lecture
        urls = [line.strip() for line in f.readlines() if line.strip()] #Nettoie les retours à la ligne (\n) pour avoir des URLs propres

    data = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page() #Ouvre un nouvel onglet dans le navigateur Chrome de façon visible "headless=False"

        print(f"Starting Scraping of {len(urls)} products") #Affiche la progression du scraping

        for i, url in enumerate(urls):
            try:
                await page.goto(url, timeout=60000, wait_until="domcontentloaded") #Navigue vers l'URL avec un timeout de 60 secondes et attend que la page soit chargée
                
                # Cookie banner (seulement au début)
                if i == 0:
                    try: await page.click('#onetrust-accept-btn-handler', timeout=2000) #Accepte les cookies si le bouton est présent
                    except: pass #Ignore si le bouton n'est pas trouvé

                #Récupération du contenu HTML de la page
                content = await page.content() #Récupère le HTML complet de la page
                soup = BeautifulSoup(content, 'html.parser') #Transforme le HTML en "soupe" pour l'analyse

                #1. Extraction basique
                h1 = soup.select_one('h1#product-title')
                name = h1.get_text(strip=True) if h1 else "N/A"
                
                #2. Liste des catégories ("breadcrumb")- Exemple : Men's > Shop by Category > Fleece > Jackets > Nom du produit
                breadcrumb = soup.select_one('ol.breadcrumb')
                category = " > ".join([li.get_text(strip=True) for li in breadcrumb.find_all('li')]) if breadcrumb else "N/A" #Joint les catégories avec ">" comme séparateur et valeur par défaut si le breadcrumb n'est pas trouvé

                #3. Description complète (appel de la fonction définie plus haut)
                description = await get_raw_description(soup)

                #4. Stockage des données extraites dans la liste
                data.append({
                    "name": name,
                    "category": category,
                    "description": description,
                    "url": url
                }) #Ajoute un dictionnaire avec les données : nom du produit, prix, catégorie, description, URL
                
                print(f"[{i+1}/{len(urls)}] OK: {name}")

            except Exception as e:
                print(f"[{i+1}/{len(urls)}] ERROR: {url} - {e}") #Gestion des erreurs pour chaque URL individuelle

        await browser.close() #Ferme le navigateur une fois toutes les URLs traitées

    #Sauvegarde
    df = pd.DataFrame(data) #Convertit la liste de dictionnaires en un tableau structuré (DataFrame)
    df.to_excel(OUTPUT_FILE, index=False) #Sauvegarde le DataFrame dans un fichier Excel sans les index
    print(f"Saved raw data to {OUTPUT_FILE}") 

if __name__ == "__main__": #Lancement du script
    asyncio.run(scrape_products())
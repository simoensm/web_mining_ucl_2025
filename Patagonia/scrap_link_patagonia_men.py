##### Note importante : nous sommes partis du principe que nous allions faire le scraping pour un premier site (Patagonia) et ensuite appliquer le même code pour d'autres sites similaires.
##### Nous n'allons commenter, dès lors, que ce code pour la partie scraping, et non les autres scripts similaires.

### Explication de la bibliothèque Playwright (NB : cette bibliothèque doit être installée via pip avant d'exécuter ce script et n'est pas utilisable dans la version 
### Playwright est un peu comme un "robot marionnettiste" qui ouvre un vrai navigateur Chrome, clique sur des boutons et lit le contenu de la page.

import asyncio
from playwright.async_api import async_playwright #Ici, nous utilisons la version asynchrone de Playwright, ce qui signifie que notre code peut faire plusieurs choses en même temps, rendant le processus plus rapide.

async def harvest_mens_links():
    url = "https://eu.patagonia.com/gb/en/shop/mens"  #L'URL de la page que nous voulons scraper, la logique est la même pour la section femmes.
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)   #headless=False permet de voir le navigateur en action
        context = await browser.new_context()
        page = await context.new_page()
        
        print(f"--- Navigate to {url} ---")
        await page.goto(url, timeout=60000, wait_until="domcontentloaded") #Le robot ouvre Patagonia et attend que la page soit complètement chargée.
        try:
            await page.click('#onetrust-accept-btn-handler', timeout=3000)  #Il essaye d'accepter les cookies si une bannière apparaît.
            print("  > Closed Cookie Banner")
        except:
            print("  > No cookie banner found")   #S'il n'y a pas de bannière, il continue simplement.

        print("--- Starting Scroll & Load Process ---")
        
        previous_count = 0
        retries = 0
        
        while True:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")  #Boucle infinie pour faire défiler la page vers le bas
            await asyncio.sleep(2)                                                 #Attendre 2 secondes pour que le contenu se charge
                                                                                   #Pourquoi ? car les sites modernes ne chargent pas tout d'un coup
            load_more = page.locator("button.load-more, .search-results-footer button")
            if await load_more.count() > 0 and await load_more.first.is_visible():
                try:
                    await load_more.first.click(force=True)                     
                    await asyncio.sleep(2)
                    retries = 0
                except:
                    pass
            
            current_count = await page.locator('.product-tile__wrapper').count()
            
            if current_count > previous_count:
                print(f"  > Products loaded: {current_count}")
                previous_count = current_count
                retries = 0
            else:
                retries += 1
                print(f"  > No new products ({retries}/3)...")  #Si aucun nouveau produit n'est chargé après plusieurs tentatives, on arrête.
            
            if retries >= 3:
                print("--- Reached end of page ---")
                break

        print("--- Extracting and Cleaning Links ---")
        
        raw_links = await page.eval_on_selector_all(
            '.product-tile__wrapper a[itemprop="url"]', 
            "elements => elements.map(e => e.href)"                
        )                                                #Récupérer tous les liens des produits sur la page
        

        unique_clean_links = set()
        
        for link in raw_links:
            clean_link = link.split('?')[0]  #Nettoyer les liens en supprimant les paramètres inutiles (tout après '?') 
            unique_clean_links.add(clean_link)   # = Eviter d'avoir plusieurs version d'un même produit ! (couleurs différentes mais même description)
        
        final_list = sorted(list(unique_clean_links))
        
        print(f"\nFOUND {len(final_list)} UNIQUE PRODUCTS (Colors merged).")  #Afficher le nombre de produits uniques trouvés
        
        with open("mens_product_links.txt", "w") as f:
            for link in final_list:
                f.write(link + "\n")  #Enregistrer les liens nettoyés dans un fichier texte avec un lien par ligne.
                
        print(f"Saved cleaned list to 'mens_product_links.txt'")
        await browser.close()   #Fermer le navigateur une fois le travail terminé.

if __name__ == "__main__":
    asyncio.run(harvest_mens_links())   #Lancer la fonction principale pour démarrer le scraping des liens.
import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import urljoin, urlparse, urlencode, parse_qs
import time
import re

# ==============================================================================
# ⚙️ CONFIGURATION UTILISATEUR
# ==============================================================================
# Change l'URL ici pour tester sur un autre site (ex: "https://shop.mango.com" ou autre Shopify)
BASE_URL = "https://en.nikinclothing.com" 

# Limites (Mettre None pour illimité) 
MAX_PRODUCTS_PER_CATEGORY = 10  # Combien de produits max par collection ?
MAX_PAGES_TO_SCAN = 3           # Combien de pages de pagination max explorer ?

# Nom du fichier de sortie
OUTPUT_CSV = "resultats_scraping_universel.csv"

# ==============================================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.7,en;q=0.5",
}

def get_soup(url: str):
    """Récupère le HTML d'une page avec gestion d'erreur."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"❌ Erreur connexion {url}: {e}")
        return None

# -------------------------------------------------------------
# 1) Découverte générique des collections
# -------------------------------------------------------------
def discover_collections(base_url: str) -> dict:
    """
    Scanne la page d'accueil pour trouver des collections.
    Essaie de deviner si c'est Homme ou Femme via l'URL.
    """
    print(f"🔍 Scan du site : {base_url} ...")
    soup = get_soup(base_url)
    if not soup:
        return {}

    collections = {"Men": set(), "Women": set(), "Uncategorized": set()}

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Pattern typique Shopify : /collections/mais-pas-products
        if "/collections/" in href and "/products/" not in href:
            full_url = urljoin(base_url, href)
            
            # Filtres anti-bruit (cartes cadeaux, nouveautés, etc.)
            ignored_keywords = ["all", "new-in", "sale", "gift", "last-chance", "bestsellers"]
            if any(k in href.lower() for k in ignored_keywords):
                continue

            lower_href = href.lower()
            if "men" in lower_href and "women" not in lower_href:
                collections["Men"].add(full_url)
            elif "women" in lower_href:
                collections["Women"].add(full_url)
            else:
                collections["Uncategorized"].add(full_url)
    
    # Conversion en listes triées
    return {k: sorted(list(v)) for k, v in collections.items()}

# -------------------------------------------------------------
# 2) Gestion de la pagination (Universelle Shopify)
# -------------------------------------------------------------
def build_next_page_url(collection_url: str, page_num: int) -> str:
    parsed = urlparse(collection_url)
    query = parse_qs(parsed.query)
    query["page"] = [str(page_num)]
    new_query = urlencode(query, doseq=True)
    return parsed._replace(query=new_query).geturl()

def get_product_links_from_page(url: str) -> list[str]:
    """Extrait tous les liens contenant '/products/' sur une page."""
    soup = get_soup(url)
    if not soup:
        return []
    
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/products/" in href:
            # Nettoyage d'URL pour éviter les doublons type /collections/xyz/products/abc
            # On veut juste https://site.com/products/abc
            if "collections" in href:
                # On essaie d'extraire la partie /products/...
                parts = href.split("/products/")
                if len(parts) > 1:
                    clean_href = "/products/" + parts[1]
                    links.add(urljoin(BASE_URL, clean_href))
            else:
                links.add(urljoin(BASE_URL, href))
    return list(links)

# -------------------------------------------------------------
# 3) Extraction UNIVERSELLE (Via Meta Tags)
# -------------------------------------------------------------
def scrape_generic_product(url: str, category_context: str, gender_context: str) -> dict:
    soup = get_soup(url)
    if not soup:
        return {}

    # --- A. Extraction via Meta Tags (Le plus fiable pour tous les sites) ---
    def get_meta(property_name):
        tag = soup.find("meta", property=property_name) or soup.find("meta", attrs={"name": property_name})
        return tag["content"].strip() if tag and tag.get("content") else None

    name = get_meta("og:title")
    description = get_meta("og:description")
    
    # Prix : souvent dans og:price:amount ou product:price:amount
    price = get_meta("og:price:amount") or get_meta("product:price:amount")
    currency = get_meta("og:price:currency") or get_meta("product:price:currency") or "€"

    # Image URL
    image_url = get_meta("og:image")

    # --- B. Fallback (Si les meta tags manquent) ---
    if not name:
        h1 = soup.find("h1")
        name = h1.get_text(strip=True) if h1 else "Inconnu"

    if not price:
        # Regex pour trouver un prix dans le texte visible (ex: 29.90 €)
        # Cherche un motif chiffre + symbole monétaire ou l'inverse
        text_body = soup.get_text()
        found_price = re.search(r'(\d+[.,]\d{2})\s?€|€\s?(\d+[.,]\d{2})', text_body)
        if found_price:
            price = found_price.group(0) # Prend toute la chaîne trouvée
        else:
            price = "N/A"

    if not description:
        # Cherche la description standard Shopify
        desc_div = soup.find(class_="product-description") or soup.find(class_="rte") or soup.find(id="description")
        description = desc_div.get_text(" ", strip=True) if desc_div else "Pas de description détectée"

    # Nettoyage final
    # Si le nom contient " - SiteName", on l'enlève
    site_name = urlparse(BASE_URL).netloc.replace("www.", "").split(".")[0]
    if name and site_name.lower() in name.lower():
        name = name.split("|")[0].strip()

    return {
        "name": name,
        "price": f"{price} {currency}" if price and "€" not in str(price) and currency else price,
        "category": category_context,
        "gender": gender_context,
        "description": description[:300] + "..." if description else "", # On tronque pour lisibilité
        "image_url": image_url,
        "product_url": url
    }

# -------------------------------------------------------------
# 4) Logique principale de boucle
# -------------------------------------------------------------
def process_category(category_url: str, gender: str, existing_urls: set):
    """
    Parcourt les pages d'une catégorie jusqu'à atteindre les limites de l'utilisateur.
    """
    collected_products = []
    
    # Extraction du nom de la catégorie depuis l'URL (pour le CSV)
    path_parts = urlparse(category_url).path.strip("/").split("/")
    cat_name = path_parts[-1] if path_parts else "general"

    page = 1
    total_scraped_in_cat = 0

    print(f"\n📂 Traitement : {cat_name} ({gender})")

    while True:
        # 1. Vérif limite pages
        if MAX_PAGES_TO_SCAN and page > MAX_PAGES_TO_SCAN:
            break
        
        # 2. Construction URL et récupération liens
        current_page_url = build_next_page_url(category_url, page)
        print(f"   📄 Page {page}...")
        
        links = get_product_links_from_page(current_page_url)
        if not links:
            break # Plus de produits ou page vide

        # 3. Scraping des produits
        for link in links:
            # Vérif limite produits par catégorie
            if MAX_PRODUCTS_PER_CATEGORY and total_scraped_in_cat >= MAX_PRODUCTS_PER_CATEGORY:
                return collected_products # On arrête tout pour cette catégorie

            # On évite de scraper 2 fois la même URL (économie de temps)
            if link in existing_urls:
                continue
            
            existing_urls.add(link)
            
            # Action !
            data = scrape_generic_product(link, cat_name, gender)
            if data.get("name"):
                collected_products.append(data)
                total_scraped_in_cat += 1
                print(f"      ✅ {data['name'][:30]}... ({data['price']})")
                time.sleep(0.2) # Petite pause politesse
            
        page += 1
    
    return collected_products

# -------------------------------------------------------------
# 5) MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Découverte des catégories
    cats = discover_collections(BASE_URL)
    
    all_data = []
    seen_urls = set() # Pour éviter les doublons globaux

    # Liste des genres à traiter
    target_groups = ["Men", "Women"] 
    # Si le site n'a pas men/women dans les URLs, tout sera dans "Uncategorized", on l'ajoute :
    if not cats["Men"] and not cats["Women"]:
        target_groups.append("Uncategorized")

    # 2. Lancement du scraping
    for group in target_groups:
        urls = cats.get(group, [])
        if not urls: continue
        
        print(f"\n--- Démarrage Groupe : {group} ({len(urls)} collections) ---")
        
        for coll_url in urls:
            products = process_category(coll_url, group, seen_urls)
            all_data.extend(products)

    # 3. Sauvegarde
    if all_data:
        keys = ["name", "price", "category", "gender", "description", "image_url", "product_url"]
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_data)
        print(f"\n✨ Terminé ! {len(all_data)} produits sauvegardés dans '{OUTPUT_CSV}'.")
    else:
        print("\n⚠️ Aucun produit trouvé. Vérifiez l'URL ou la structure du site.")
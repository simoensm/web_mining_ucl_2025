import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import urljoin, urlparse, urlencode, parse_qs
import time
import re
import json
import random

# --- CONFIGURATION ---
BASE_URL = "https://organicbasics.com"
# On démarre sur une page qui liste généralement toutes les collections ou le menu principal
START_URL = "https://organicbasics.com/collections/all" 

MAX_PRODUCTS_PER_CATEGORY = 20  
MAX_PAGES_TO_SCAN = 3
OUTPUT_CSV = "resultats_organicbasics.csv"

# ⚠️ AJOUTEZ VOS PROXIES ICI
# Format: "http://user:pass@ip:port" ou "http://ip:port"
PROXY_LIST = [
    # "http://user:password@123.45.67.89:8080",
    # "http://user:password@98.76.54.32:8080",
]

# Liste d'User-Agents pour varier les signatures
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0"
]

def get_random_header():
    """Génère un header avec un User-Agent aléatoire."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9", # Organic Basics est en anglais par défaut
    }

def get_soup(url: str, retries=3):
    """
    Récupère le HTML avec rotation de proxy et gestion des retries.
    """
    attempt = 0
    while attempt < retries:
        proxies = {}
        if PROXY_LIST:
            proxy_url = random.choice(PROXY_LIST)
            proxies = {"http": proxy_url, "https": proxy_url}
            # print(f"  🔄 Utilisation du proxy : {proxy_url}") # Décommentez pour debug

        try:
            resp = requests.get(
                url, 
                headers=get_random_header(), 
                proxies=proxies, 
                timeout=15
            )
            
            # Gestion spécifique 403/429 (Bannissement)
            if resp.status_code in [403, 429]:
                print(f"  ⛔ Bloqué ({resp.status_code}) sur {url}. Changement de proxy...")
                attempt += 1
                time.sleep(2)
                continue
                
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
            
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️ Erreur proxy (tentative {attempt+1}/{retries}): {e}")
            attempt += 1
            time.sleep(1)

    print(f"❌ Échec définitif pour l'URL : {url}")
    return None

def discover_collections(start_url: str) -> dict:
    """
    Scan adapté pour Organic Basics. 
    On cherche les liens contenant '/collections/' et on les classe par genre.
    """
    print(f"🔍 Scan des collections depuis : {start_url} ...")
    soup = get_soup(start_url)
    
    # Structure par défaut si le scan échoue
    default_collections = {
        "Men": ["https://organicbasics.com/collections/men"],
        "Women": ["https://organicbasics.com/collections/women"]
    }

    if not soup:
        return default_collections

    collections = {"Men": set(), "Women": set(), "Other": set()}
    
    # Mots clés à exclure
    ignored = ["all", "impact", "materials", "reviews", "shipping", "returns", "gift-card"]

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/collections/" in href and "/products/" not in href:
            full_url = urljoin(BASE_URL, href)
            
            # Filtre les liens inutiles
            if any(k in href.lower() for k in ignored):
                continue
            
            lower_href = href.lower()
            
            # Classification simple basée sur l'URL
            if "men" in lower_href and "women" not in lower_href:
                collections["Men"].add(full_url)
            elif "women" in lower_href:
                collections["Women"].add(full_url)
            else:
                collections["Other"].add(full_url)
    
    # Si le scan automatique ne trouve rien (ex: chargement JS du menu), on utilise les défauts
    if not collections["Men"] and not collections["Women"]:
        print("⚠️ Scan automatique vide, utilisation des collections par défaut.")
        return default_collections

    return {k: sorted(list(v)) for k, v in collections.items()}

def build_next_page_url(collection_url: str, page_num: int) -> str:
    """Gère la pagination Shopify standard."""
    parsed = urlparse(collection_url)
    query = parse_qs(parsed.query)
    query["page"] = [str(page_num)]
    new_query = urlencode(query, doseq=True)
    return parsed._replace(query=new_query).geturl()

def get_product_links_from_page(url: str) -> list[str]:
    """Extrait les liens produits."""
    soup = get_soup(url)
    if not soup:
        return []
    
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/products/" in href:
            # Nettoyage URL
            clean_path = href.split("?")[0]
            # On retire la partie collection de l'URL pour avoir l'URL canonique
            if "collections" in clean_path:
                parts = clean_path.split("/products/")
                if len(parts) > 1:
                    clean_path = "/products/" + parts[1]
            
            links.add(urljoin(BASE_URL, clean_path))
            
    return list(links)

def scrape_product_shopify(url: str, category_context: str, gender_context: str) -> dict:
    """Scrape les détails d'un produit."""
    soup = get_soup(url)
    if not soup:
        return {}

    product_data = {
        "name": None,
        "price": None,
        "currency": None,
        "category": category_context,
        "gender": gender_context,
        "description": None,
        "image_url": None,
        "product_url": url
    }

    # --- STRATÉGIE JSON-LD ---
    json_ld = soup.find('script', type='application/ld+json')
    if json_ld:
        try:
            content = json_ld.string
            # Parfois le JSON est malformé ou vide
            if content:
                data = json.loads(content)
                if isinstance(data, list):
                    data = next((item for item in data if item.get("@type") == "Product"), {})
                
                if data.get("@type") == "Product":
                    product_data["name"] = data.get("name")
                    product_data["description"] = data.get("description")
                    
                    img = data.get("image")
                    if isinstance(img, list): product_data["image_url"] = img[0]
                    elif isinstance(img, dict): product_data["image_url"] = img.get("url")
                    else: product_data["image_url"] = img

                    offers = data.get("offers")
                    if isinstance(offers, list): offers = offers[0]
                    if offers:
                        product_data["price"] = offers.get("price")
                        product_data["currency"] = offers.get("priceCurrency", "EUR")

        except Exception as e:
            # print(f"Info: JSON-LD parsing error: {e}") 
            pass

    # --- STRATÉGIE META TAGS (Fallback) ---
    def get_meta(prop):
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        return tag["content"].strip() if tag and tag.get("content") else None

    if not product_data["name"]:
        product_data["name"] = get_meta("og:title") or (soup.find("h1").get_text(strip=True) if soup.find("h1") else "Unknown")

    if not product_data["price"]:
        amount = get_meta("og:price:amount") or get_meta("product:price:amount")
        currency = get_meta("og:price:currency") or get_meta("product:price:currency") or "EUR"
        if amount:
            product_data["price"] = amount
            product_data["currency"] = currency

    # Nettoyage description
    if product_data["description"]:
        clean = re.sub(r'<[^>]+>', '', product_data["description"])
        product_data["description"] = " ".join(clean.split())[:300] + "..."

    return product_data

def process_category(category_url: str, gender: str, existing_urls: set):
    """Gère le scraping d'une catégorie entière."""
    collected_products = []
    cat_name = urlparse(category_url).path.strip("/").split("/")[-1]
    
    print(f"\n📂 Catégorie : {cat_name} ({gender})")
    
    page = 1
    total_in_cat = 0

    while True:
        if MAX_PAGES_TO_SCAN and page > MAX_PAGES_TO_SCAN:
            break
        
        current_url = build_next_page_url(category_url, page)
        print(f"  📄 Scan Page {page}...")
        
        links = get_product_links_from_page(current_url)
        if not links:
            break

        new_links_found = False
        for link in links:
            if MAX_PRODUCTS_PER_CATEGORY and total_in_cat >= MAX_PRODUCTS_PER_CATEGORY:
                return collected_products

            if link in existing_urls:
                continue
            
            new_links_found = True
            existing_urls.add(link)
            
            data = scrape_product_shopify(link, cat_name, gender)
            if data.get("name"):
                collected_products.append(data)
                total_in_cat += 1
                price_str = f"{data['price']} {data['currency']}" if data['price'] else "N/A"
                print(f"      ✅ {data['name'][:35]:<40} | {price_str}")
                
                # Pause aléatoire pour imiter un humain
                time.sleep(random.uniform(0.5, 1.5))
        
        if not new_links_found:
            # Si tous les liens de la page ont déjà été vus, on arrête (pagination infinie ou fin de liste)
            break
            
        page += 1

    return collected_products

if __name__ == "__main__":
    if not PROXY_LIST:
        print("⚠️  ATTENTION: Aucun proxy configuré dans PROXY_LIST.")
        print("⚠️  Le script utilisera votre IP locale. Risque de blocage si usage intensif.\n")
        time.sleep(2)

    # 1. Découverte
    cats = discover_collections(START_URL)
    
    all_data = []
    seen_urls = set()
    
    # 2. Scraping
    # On priorise Homme/Femme pour Organic Basics
    for group in ["Men", "Women"]:
        urls = cats.get(group, [])
        if not urls: continue
        
        print(f"\n--- 🚀 Groupe : {group} ({len(urls)} collections) ---")
        for url in urls:
            products = process_category(url, group, seen_urls)
            all_data.extend(products)

    # 3. Sauvegarde
    if all_data:
        keys = ["name", "price", "currency", "category", "gender", "description", "image_url", "product_url"]
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_data)
        print(f"\n✨ Terminé ! {len(all_data)} produits sauvegardés dans '{OUTPUT_CSV}'.")
    else:
        print("\n⚠️ Aucun produit trouvé. Vérifiez les sélecteurs ou le blocage anti-bot.")
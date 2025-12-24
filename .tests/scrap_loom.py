import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import urljoin, urlparse, urlencode, parse_qs
import time
import re
import json

# --- CONFIGURATION ---
BASE_URL = "https://www.loom.fr"
START_URL = "https://www.loom.fr/collections/"
OUTPUT_CSV = "resultats_loom_clean.csv"

# Pas de limites
MAX_PRODUCTS = None 
MAX_PAGES = None

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.7,en;q=0.5",
}

def get_soup(url: str):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"❌ Erreur connexion {url}: {e}")
        return None

def clean_text(text):
    if not text: return ""
    # Nettoie les espaces multiples
    return re.sub(r'\s+', ' ', text).strip()

def is_garbage_text(text):
    """Filtre les textes qui viennent du menu ou des bannières promo."""
    garbage_phrases = [
        "Commandez avant le", "recevoir votre colis avant Noël",
        "Qui sommes-nous", "Se connecter", "Mon panier", "Rechercher",
        "Menu", "Panier", "Fermer", "Connexion", "Créer un compte",
        "Vêtements →", "Accessoires →", "Cadeaux à moins de"
    ]
    if len(text) < 3: return True
    for phrase in garbage_phrases:
        if phrase.lower() in text.lower():
            return True
    return False

def get_price_robust(soup, json_data):
    """Logique renforcée pour trouver le prix."""
    # 1. JSON-LD (Prioritaire)
    if json_data:
        offers = json_data.get("offers")
        if isinstance(offers, list): offers = offers[0]
        if isinstance(offers, dict):
            p = offers.get("price")
            if p: return f"{p} EUR"

    # 2. Sélecteurs CSS spécifiques à la zone produit (et pas ailleurs)
    # On cherche dans le conteneur info produit pour éviter les prix des "produits recommandés"
    product_info = soup.select_one('.product-single__meta, .product__info-container, .product-info')
    
    if product_info:
        # Cherche les classes de prix standard Shopify
        price_tag = product_info.select_one('.price-item--regular, .price__current, .product__price')
        if price_tag:
            return clean_text(price_tag.get_text())

    # 3. Fallback Meta
    meta_price = soup.find("meta", property="og:price:amount")
    if meta_price:
        return f"{meta_price['content']} EUR"

    return "Prix non détecté"

def scrape_everything_on_page(url: str, category: str, gender: str) -> dict:
    soup = get_soup(url)
    if not soup: return {}

    # --- ÉTAPE CRUCIALE : CIBLAGE DE LA ZONE PRINCIPALE ---
    # On ignore le Header (Menu) et le Footer. On ne travaille que dans <main> ou la section produit.
    main_content = soup.find('main') or soup.select_one('.product-section') or soup.select_one('#MainContent')
    
    if not main_content:
        # Si on ne trouve pas le main, on utilise soup mais on risque d'avoir le menu
        main_content = soup

    # --- 1. JSON-LD ---
    product_json = {}
    json_ld_tags = soup.find_all('script', type='application/ld+json')
    for tag in json_ld_tags:
        try:
            data = json.loads(tag.string)
            if isinstance(data, list):
                for item in data:
                    if item.get("@type") == "Product":
                        product_json = item
                        break
            elif data.get("@type") == "Product":
                product_json = data
        except: continue

    # --- 2. INFOS DE BASE ---
    name = product_json.get("name")
    if not name:
        h1 = main_content.find("h1")
        name = h1.get_text(strip=True) if h1 else "Inconnu"
    name = name.split("|")[0].strip()

    price = get_price_robust(soup, product_json)

    # --- 3. IMAGES ---
    images = []
    # On cherche les images UNIQUEMENT dans le bloc produit
    img_tags = main_content.select('.product__media img, .product-gallery img')
    for img in img_tags:
        src = img.get('src') or img.get('data-src') or img.get('srcset')
        if src:
            if src.startswith("//"): src = "https:" + src
            src = src.split("?")[0]
            if src not in images and "icon" not in src: # Evite les petites icônes
                images.append(src)
    all_images_str = " | ".join(images[:10])

    # --- 4. TAILLES ---
    sizes = []
    size_labels = main_content.select('.variant-input-wrap label, .product-form__input label, fieldset[name="Size"] label')
    for label in size_labels:
        s_text = clean_text(label.get_text())
        if s_text: sizes.append(s_text)
    sizes_str = ", ".join(list(set(sizes)))

    # --- 5. CONTENU PROPRE (Sans le menu) ---
    full_content = []

    # A. Description (limitée à la zone produit)
    desc_div = main_content.select_one('.product-description, .rte, .product__text')
    if desc_div:
        text = clean_text(desc_div.get_text())
        if not is_garbage_text(text):
            full_content.append(f"[DESCRIPTION]:\n{text}")

    # B. Accordéons (Matière, Lavage, etc.)
    # On cherche les <details> ou accordéons UNIQUEMENT dans le main_content
    accordions = main_content.select('details, .accordion__item, .collapsible-content')
    
    for acc in accordions:
        # Extraction Titre
        title_tag = acc.find('summary') or acc.select_one('.accordion__title') or acc.select_one('.collapsible-trigger')
        title = clean_text(title_tag.get_text()) if title_tag else "Détail"
        
        # Ignorer les menus cachés dans des accordéons (ex: filtre de tri)
        if is_garbage_text(title): continue

        # Extraction Contenu
        content_text = clean_text(acc.get_text())
        # Nettoyage : enlever le titre du contenu pour éviter la répétition
        if title in content_text:
            content_text = content_text.replace(title, "", 1).strip()
        
        if content_text and not is_garbage_text(content_text):
            full_content.append(f"[{title.upper()}]: {content_text}")

    # C. Blocs texte supplémentaires (spécifique Loom)
    extra_blocks = main_content.select('.product-block--text p')
    for block in extra_blocks:
        text = clean_text(block.get_text())
        if len(text) > 20 and not is_garbage_text(text):
            full_content.append(f"[INFO]: {text}")

    final_description_blob = "\n\n".join(full_content)

    return {
        "name": name,
        "price": price,
        "category": category,
        "gender": gender,
        "sizes_available": sizes_str,
        "all_images": all_images_str,
        "full_content": final_description_blob,
        "product_url": url
    }

# --- NAVIGATION (Inchangée) ---

def discover_collections(start_url: str):
    soup = get_soup(start_url)
    if not soup:
        return {"Homme": ["https://www.loom.fr/collections/homme"], "Femme": ["https://www.loom.fr/collections/femme"]}
    collections = {"Homme": set(), "Femme": set(), "Autre": set()}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/collections/" in href and "/products/" not in href:
            full = urljoin(BASE_URL, href)
            ignored = ["all", "tous-les-vetements", "carte-cadeau", "new-in", "last-chance"]
            if any(k in href.lower() for k in ignored): continue
            if "homme" in href.lower() and "femme" not in href.lower(): collections["Homme"].add(full)
            elif "femme" in href.lower(): collections["Femme"].add(full)
            else: collections["Autre"].add(full)
    return {k: sorted(list(v)) for k, v in collections.items()}

def get_product_links(url):
    soup = get_soup(url)
    if not soup: return []
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/products/" in href:
            clean = href.split("?")[0]
            if "collections" in clean:
                parts = clean.split("/products/")
                if len(parts) > 1: clean = "/products/" + parts[1]
            links.add(urljoin(BASE_URL, clean))
    return list(links)

def process_category(url, gender, existing):
    products = []
    page = 1
    while True:
        if MAX_PAGES and page > MAX_PAGES: break
        parsed = urlparse(url)
        q = parse_qs(parsed.query)
        q["page"] = [str(page)]
        page_url = parsed._replace(query=urlencode(q, doseq=True)).geturl()
        links = get_product_links(page_url)
        if not links: break
        
        new_found = False
        for link in links:
            if link in existing: continue
            new_found = True
            existing.add(link)
            data = scrape_everything_on_page(link, url.split("/")[-1], gender)
            if data.get("name"):
                products.append(data)
                desc_prev = data['full_content'][:30].replace("\n", " ") + "..."
                print(f"   👕 {data['name'][:20]:<20} | 💰 {data['price']:<10} | 📝 {desc_prev}")
                time.sleep(0.5)
        
        if not new_found and page > 1: break
        page += 1
    return products

if __name__ == "__main__":
    cats = discover_collections(START_URL)
    all_data = []
    seen = set()
    
    if not cats["Homme"] and not cats["Femme"]:
        cats["Homme"] = ["https://www.loom.fr/collections/homme"]
        cats["Femme"] = ["https://www.loom.fr/collections/femme"]

    for group in ["Homme", "Femme", "Autre"]:
        urls = cats.get(group, [])
        if not urls: continue
        print(f"\n--- 🚀 GROUPE : {group} ---")
        for u in urls:
            all_data.extend(process_category(u, group, seen))

    if all_data:
        keys = ["name", "price", "category", "gender", "sizes_available", "all_images", "full_content", "product_url"]
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_data)
        print(f"\n✅ Terminé ! {len(all_data)} produits sauvegardés.")
    else:
        print("❌ Rien trouvé.")
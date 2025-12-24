import time
import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright
from tqdm import tqdm

# =====================================================
# CONFIGURATION
# =====================================================
BASE_URL = "https://twothirds.com"
HEADERS = {"User-Agent": "Mozilla/5.0 WebMiningBot/1.0"}

OUTPUT_FILE = "twothirds_products.xlsx"
CHECKPOINT_FILE = "scraped_products.csv"

SCROLL_PAUSE = 1.5
MAX_SCROLLS = 30
REQUEST_DELAY = 1.0

# =====================================================
# UTILS
# =====================================================
def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


def get_soup(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        # print(f"🌍 ACCESS {url} → {r.status_code}") # Reduced verbosity
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"❌ FAILED {url}: {e}")
        return None


def clean_text(el):
    return el.get_text(strip=True) if el else None


# =====================================================
# STEP 1: EXTRACT CATEGORIES & SUBCATEGORIES
# =====================================================
def extract_collections():
    soup = get_soup(BASE_URL)
    collections = {}

    if not soup:
        return collections

    for a in soup.select("a[href*='/collections/']"):
        name = clean_text(a)
        href = a.get("href")

        if not name or not href:
            continue

        url = urljoin(BASE_URL, href)

        if "women" in href.lower():
            category = "Women"
        elif "men" in href.lower():
            category = "Men"
        else:
            category = "Other"

        collections.setdefault(category, {})
        collections[category][name] = url

    return collections


# =====================================================
# STEP 2: PLAYWRIGHT SCROLL
# =====================================================
def get_all_products_by_scrolling(collection_url):
    print(f"\n🧭 SCROLLING COLLECTION: {collection_url}")
    product_links = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(collection_url, timeout=60000)
            time.sleep(3)
            
            last_count = 0
            for i in range(MAX_SCROLLS):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(SCROLL_PAUSE)

                links = page.eval_on_selector_all(
                    "a[href*='/products/']",
                    "els => els.map(e => e.href)"
                )

                # Clean links (remove query params)
                links = {l.split("?")[0] for l in links}
                product_links.update(links)

                print(f"   🔄 Scroll {i+1}: {len(product_links)} products found so far...")

                if len(product_links) == last_count:
                    print("   ⛔ No new products detected, stopping scroll")
                    break
                
                last_count = len(product_links)
        except Exception as e:
            print(f"   ⚠️ Scroll Error: {e}")
        finally:
            browser.close()

    print(f"   ✅ TOTAL PRODUCTS FOUND: {len(product_links)}")
    return product_links


# =====================================================
# STEP 3: SCRAPE PRODUCT PAGE
# =====================================================
def scrape_product(url, category, subcategory):
    soup = get_soup(url)
    if not soup:
        return None

    try:
        name = clean_text(soup.find("h1"))
        price = clean_text(soup.select_one("[data-product-price], .price"))
        description = clean_text(soup.select_one(".rte, .product__description"))

        return {
            "Product Name": name,
            "Normalized Name": normalize_text(name),
            "Price": price,
            "Category": category,
            "Subcategory": subcategory,
            "Description": description,
            "Product URL": url
        }

    except Exception as e:
        print(f"❌ Product error {url}: {e}")
        return None


# =====================================================
# STEP 4: SAVING & PROCESSING
# =====================================================
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            df = pd.read_csv(CHECKPOINT_FILE)
            return df.to_dict("records")
        except:
            return []
    return []

def save_checkpoint(data):
    pd.DataFrame(data).to_csv(CHECKPOINT_FILE, index=False)

def export_to_excel(data, filename):
    """
    Handles deduplication and saving to Excel.
    Includes error handling in case the file is open.
    """
    if not data:
        return

    print(f"💾 SAVING EXCEL ({len(data)} items)...", end=" ")
    
    try:
        df = pd.DataFrame(data)
        
        # Deduplicate
        df = df.drop_duplicates(subset=["Product URL"])
        if "Normalized Name" in df.columns:
            df = df.drop_duplicates(subset=["Normalized Name"])
            # Remove the helper column for the final excel
            df = df.drop(columns=["Normalized Name"])
            
        df.to_excel(filename, index=False)
        print("✅ Done.")
    except PermissionError:
        print("\n⚠️  WARNING: Could not save Excel file. Is it open? Skipping save this round.")
    except Exception as e:
        print(f"\n⚠️  Error saving Excel: {e}")


def run_scraper():
    collections = extract_collections()

    print("\n🌐 SITE STRUCTURE")
    for cat, subs in collections.items():
        print(f"\n📁 {cat}")
        for s in subs:
            print(f"   └── {s}")

    scraped_data = load_checkpoint()
    scraped_urls = {p["Product URL"] for p in scraped_data}
    results = scraped_data.copy()
  
    for category, subcats in collections.items():
        for subcategory, collection_url in subcats.items():

            print(f"\n🔎 CATEGORY → {category} / {subcategory}")

            product_links = get_all_products_by_scrolling(collection_url)

            # Convert to list to iterate
            new_links = [l for l in product_links if l not in scraped_urls]

            if not new_links:
                print("   ✨ All items in this category already scraped.")
                continue

            for link in tqdm(new_links, desc="Scraping Items"):
                product = scrape_product(link, category, subcategory)
                if product:
                    results.append(product)
                    scraped_urls.add(link)
                    
                    # Save CSV checkpoint frequently (every item)
                    save_checkpoint(results)

                time.sleep(REQUEST_DELAY)

            # === SAVE EXCEL AFTER EVERY SUBCATEGORY ===
            export_to_excel(results, OUTPUT_FILE)

    # Final Save
    print("\n🏁 SCRAPING FINISHED.")
    export_to_excel(results, OUTPUT_FILE)


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    run_scraper()
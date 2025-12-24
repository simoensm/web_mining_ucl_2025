import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin, urlparse
import time

# -----------------------------
# Configuration
# -----------------------------
BASE_URL = ' https://ecoalf.com/'  # Replace with the website you want to scrape
OUTPUT_FILE = 'products.xlsx'
DELAY_BETWEEN_REQUESTS = 1  # seconds, to avoid overloading the server

# -----------------------------
# Helper Functions
# -----------------------------

def get_soup(url):
    """
    Fetches the content of a URL and returns a BeautifulSoup object.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_product_info(product_url, category=None, subcategory=None):
    """
    Extracts product information from a product page.
    """
    soup = get_soup(product_url)
    if not soup:
        return None

    try:
        # Replace these selectors with the ones relevant to the target site
        name = soup.select_one('h1.product-title').get_text(strip=True)
    except:
        name = None

    try:
        price = soup.select_one('.price').get_text(strip=True)
    except:
        price = None

    try:
        description = soup.select_one('.product-description').get_text(strip=True)
    except:
        description = None

    return {
        'Name': name,
        'Price': price,
        'Category': category,
        'Subcategory': subcategory,
        'Description': description,
        'Link': product_url
    }

def get_all_links(soup, base_url):
    """
    Returns all unique, internal links found in the soup.
    """
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(base_url, href)
        # Keep only internal links
        if urlparse(full_url).netloc == urlparse(base_url).netloc:
            links.add(full_url)
    return links

def find_products_in_category(category_url, category=None, subcategory=None):
    """
    Finds all product pages in a given category/subcategory URL.
    Handles pagination automatically.
    """
    products = []
    page_url = category_url

    while page_url:
        soup = get_soup(page_url)
        if not soup:
            break

        # Extract product links from the category page
        # Replace '.product-link' with the site's product link selector
        product_links = [urljoin(BASE_URL, a['href']) for a in soup.select('.product-link')]
        for link in product_links:
            info = extract_product_info(link, category, subcategory)
            if info:
                products.append(info)
            time.sleep(DELAY_BETWEEN_REQUESTS)

        # Handle pagination (assumes a "next" link exists)
        next_page = soup.select_one('a.next')
        if next_page:
            page_url = urljoin(BASE_URL, next_page['href'])
        else:
            page_url = None

    return products

# -----------------------------
# Main Scraper
# -----------------------------

def scrape_website(base_url):
    all_products = []
    visited_links = set()
    to_visit = [base_url]

    while to_visit:
        current_url = to_visit.pop(0)
        if current_url in visited_links:
            continue

        visited_links.add(current_url)
        soup = get_soup(current_url)
        if not soup:
            continue

        # Add new internal links to the queue
        links = get_all_links(soup, base_url)
        to_visit.extend(links - visited_links)

        # Heuristic: if this page is a category or subcategory page, scrape products
        # Adjust selectors based on the website's navigation/menu structure
        if '/category/' in current_url or '/shop/' in current_url:
            # Attempt to parse category/subcategory from URL
            parts = urlparse(current_url).path.strip('/').split('/')
            category = parts[-2] if len(parts) >= 2 else None
            subcategory = parts[-1] if len(parts) >= 1 else None

            print(f"Scraping category: {category}, subcategory: {subcategory}")
            products = find_products_in_category(current_url, category, subcategory)
            all_products.extend(products)

        time.sleep(DELAY_BETWEEN_REQUESTS)

    return all_products

# -----------------------------
# Run Scraper and Save to Excel
# -----------------------------
if __name__ == '__main__':
    print(f"Starting scraping {BASE_URL}")
    products = scrape_website(BASE_URL)
    print(f"Scraped {len(products)} products.")

    # Save to Excel
    df = pd.DataFrame(products)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Data saved to {OUTPUT_FILE}")

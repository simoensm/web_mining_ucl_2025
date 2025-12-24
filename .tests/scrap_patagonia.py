import asyncio
import pandas as pd
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import sys

class PatagoniaScraper:
    def __init__(self, start_url, output_file='patagonia_products.xlsx'):
        self.start_url = start_url
        self.output_file = output_file
        self.visited_urls = set()
        self.product_data = []
        self.queue = [start_url]
        self.allowed_categories = ['mens', 'womens'] # Filter for Men and Women only
        
        # State for the "Test Line" feature
        self.first_product_verified = False 
        self.products_found_count = 0

    async def handle_popups(self, page):
        """
        Attempts to close common popups like Cookie banners or Sign-up modals.
        """
        try:
            # Common selector for OneTrust cookie banners (Patagonia uses this often)
            await page.click('#onetrust-accept-btn-handler', timeout=2000)
            print("  [Info] Cookie banner accepted.")
        except:
            pass # No banner found, continue

        try:
            # unexpected email signup popups (generic close button attempts)
            await page.click('button[class*="close"]', timeout=1000)
            print("  [Info] Popup closed.")
        except:
            pass

    def get_category_info(self, url, soup):
        """
        Infers Category and Subcategory from the URL structure or Breadcrumbs.
        Patagonia URL format usually: /shop/category/subcategory/product-name
        """
        parsed = urlparse(url)
        path_segments = [s for s in parsed.path.split('/') if s]
        
        # Default fallback
        category = "Unknown"
        subcategory = "Unknown"

        # Strategy 1: URL Analysis
        # Example: /gb/en/shop/mens-jackets-vests/....
        for segment in path_segments:
            if 'mens' in segment and 'womens' not in segment:
                category = "Men"
            elif 'womens' in segment:
                category = "Women"
        
        # Strategy 2: Breadcrumbs (More accurate if available)
        breadcrumbs = soup.select('.breadcrumbs .breadcrumb-element')
        if breadcrumbs:
            texts = [b.get_text(strip=True) for b in breadcrumbs]
            # Usually: Home > Men > Jackets > Product
            if len(texts) >= 2:
                # Often the second item is the main category
                if "Men" in texts or "Women" in texts:
                    category = next((t for t in texts if t in ["Men", "Women"]), category)
                # The item before the last one is often the subcategory
                if len(texts) > 2:
                    subcategory = texts[-2]

        return category, subcategory

    async def scrape_product(self, page, url):
        """
        Extracts details from a single product page.
        """
        print(f"  [Scraping Product] {url}")
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        try:
            # 1. Product Name
            # Patagonia uses h1 usually with class 'product-name'
            name_tag = soup.find('h1', class_='product-name')
            name = name_tag.get_text(strip=True) if name_tag else "N/A"

            # 2. Price
            # Prices can be complex (sales, ranges). We look for the main sales price container.
            price_tag = soup.select_one('.prices .value, .sales .value, span[itemprop="price"]')
            price = price_tag.get_text(strip=True) if price_tag else "N/A"

            # 3. Description
            # Usually in a div with 'description' class
            desc_tag = soup.select_one('.description-text, div[itemprop="description"]')
            description = desc_tag.get_text(strip=True) if desc_tag else "No description found"

            # 4. Categories
            category, subcategory = self.get_category_info(url, soup)

            product_info = {
                "Product Name": name,
                "Price": price,
                "Category": category,
                "Subcategory": subcategory,
                "Description": description[:500] + "..." if len(description) > 500 else description, # Truncate for display
                "Link": url
            }

            return product_info

        except Exception as e:
            print(f"  [Error] Failed to extract data from {url}: {e}")
            return None

    def is_valid_link(self, link):
        """
        Filters links to ensure we only follow relevant paths (Men/Women shops).
        """
        if not link or link.startswith('javascript') or link.startswith('#'):
            return False
        
        # Parse link to check domain matches (stay on patagonia)
        # and ensure it's in the 'shop' section and is Men or Women
        if "patagonia.com" in link and "/shop/" in link:
            lower_link = link.lower()
            # Only process if it contains mens or womens keywords
            if any(cat in lower_link for cat in self.allowed_categories):
                return True
        return False

    async def run(self):
        async with async_playwright() as p:
            # Launch browser (headless=False so you can see it working initially)
            browser = await p.chromium.launch(headless=False) 
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            page = await context.new_page()

            print(f"--- Starting BFS Crawl at {self.start_url} ---")
            
            # Initial Load
            try:
                await page.goto(self.start_url, timeout=60000)
                await self.handle_popups(page)
            except Exception as e:
                print(f"Critical Error loading homepage: {e}")
                return

            while self.queue:
                current_url = self.queue.pop(0)
                
                if current_url in self.visited_urls:
                    continue
                
                self.visited_urls.add(current_url)
                
                try:
                    await page.goto(current_url, wait_until='domcontentloaded')
                    
                    # Logic to identify if this is a Product Page
                    # Product pages usually don't end in a slash or have specific IDs, 
                    # but identifying by 'Add to Cart' button or specific selectors is safer.
                    is_product = await page.locator('.add-to-cart, button[data-action="AddToCart"]').count() > 0

                    if is_product:
                        data = await self.scrape_product(page, current_url)
                        if data:
                            # --- TEST LINE LOGIC START ---
                            if not self.first_product_verified:
                                print("\n" + "="*50)
                                print("TEST MODE: First Product Scraped. Please Verify.")
                                print(f"Name: {data['Product Name']}")
                                print(f"Price: {data['Price']}")
                                print(f"Cat/Subcat: {data['Category']} / {data['Subcategory']}")
                                print(f"Desc: {data['Description'][:100]}...")
                                print("="*50)
                                user_input = input("Does this look correct? (y/n) to continue or abort: ").strip().lower()
                                if user_input != 'y':
                                    print("Aborting script based on user input.")
                                    await browser.close()
                                    return
                                self.first_product_verified = True
                                print("Continuing full scrape...\n")
                            # --- TEST LINE LOGIC END ---

                            self.product_data.append(data)
                            self.products_found_count += 1
                            # Save periodically to avoid data loss
                            if self.products_found_count % 10 == 0:
                                self.save_to_excel()

                    # Extract new links for BFS (Navigation & Pagination)
                    # We look for links in the main content area to avoid footer legal links etc.
                    links = await page.eval_on_selector_all('a[href]', "elements => elements.map(e => e.href)")
                    
                    for link in links:
                        # Normalize URL
                        link = urljoin(current_url, link)
                        # Remove query parameters for cleaner BFS (optional, depends on site pagination)
                        # Keeping query params often necessary for pagination (?start=30)
                        
                        if self.is_valid_link(link) and link not in self.visited_urls and link not in self.queue:
                            self.queue.append(link)
                            # Print structure discovery
                            if "/shop/" in link:
                                # Clean up URL for display
                                display_name = link.split('/shop/')[-1].replace('/', ' > ').replace('-', ' ').title()
                                # print(f"  [Structure Found] {display_name[:50]}...")

                except Exception as e:
                    print(f"  [Error] Processing {current_url}: {e}")
                    continue

            await browser.close()
            self.save_to_excel()
            print("Scraping Completed.")

    def save_to_excel(self):
        if not self.product_data:
            print("No data to save.")
            return
        
        df = pd.DataFrame(self.product_data)
        df.to_excel(self.output_file, index=False)
        print(f"  [Data Saved] {len(self.product_data)} products saved to {self.output_file}")

# --- Entry Point ---
if __name__ == "__main__":
    # URL for Patagonia Europe
    target_url = "https://eu.patagonia.com/gb/en/home/"
    
    scraper = PatagoniaScraper(target_url)
    asyncio.run(scraper.run())
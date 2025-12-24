import asyncio
import random
import os
import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup

class PatagoniaForceScraper:
    def __init__(self):
        # 1. We start ONLY with these two URLs
        self.start_urls = [
            "https://eu.patagonia.com/gb/en/shop/mens",
            "https://eu.patagonia.com/gb/en/shop/womens"
        ]
        self.product_urls = set() # To avoid duplicates
        self.scraped_data = []
        self.output_file = 'patagonia_products_v3.xlsx'
        
        # Dashboard counters
        self.mens_count = 0
        self.womens_count = 0
        self.first_verified = False

    async def random_sleep(self, min_s=1, max_s=3):
        await asyncio.sleep(random.uniform(min_s, max_s))

    async def close_popups(self, page):
        """Aggressively closes cookie/signup popups."""
        try:
            # Common Patagonia popup selectors
            await page.click('#onetrust-accept-btn-handler', timeout=1000)
        except: pass
        try:
            await page.click('button.close', timeout=500)
        except: pass

    async def scroll_to_load_all(self, page):
        """
        Scrolls down and clicks 'Load More' until all products are visible.
        """
        print(f"   [Navigation] Loading all products on {page.url}...")
        
        previous_height = 0
        no_change_count = 0
        
        while True:
            # Scroll to bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
            
            # Check for 'Load More' button
            try:
                # Patagonia often uses a button with class 'load-more' or similar in the search footer
                load_more = page.locator("button.load-more, .search-results-footer .btn")
                if await load_more.is_visible():
                    print("   [Pagination] Clicking 'Load More'...")
                    await load_more.click(force=True)
                    await asyncio.sleep(3) # Wait for content to load
                    no_change_count = 0 # Reset counter if we clicked
                else:
                    # If no button, check if we are just stuck
                    curr_height = await page.evaluate("document.body.scrollHeight")
                    if curr_height == previous_height:
                        no_change_count += 1
                        if no_change_count >= 2: # Stop if height hasn't changed twice
                            break
                    else:
                        previous_height = curr_height
                        no_change_count = 0
            except:
                break

    async def collect_product_links(self, page, category_name):
        """
        Extracts all product links from the grid after scrolling.
        """
        print(f"   [Extraction] Gathering product links for {category_name}...")
        
        # Selector for the clickable title inside the product card
        # Patagonia usually: .product-tile -> .product-name -> a
        links = await page.eval_on_selector_all(
            '.product-tile .product-name a', 
            "elements => elements.map(e => e.href)"
        )
        
        new_links = 0
        for link in links:
            if link not in self.product_urls:
                self.product_urls.add(link)
                new_links += 1
        
        print(f"   [Found] {new_links} new products in {category_name}. Total Queue: {len(self.product_urls)}")

    async def scrape_detail_page(self, page, url):
        """
        Visits the product page and extracts data.
        """
        try:
            await page.goto(url, wait_until='domcontentloaded')
            await self.close_popups(page)
            
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')

            # 1. NAME
            name = soup.select_one('h1.product-name, h1.pdp-main__title').get_text(strip=True)
            
            # 2. PRICE
            price_tag = soup.select_one('.prices .sales .value, .price .value, span[itemprop="price"]')
            price = price_tag.get_text(strip=True) if price_tag else "N/A"

            # 3. CATEGORY (Infer from URL)
            cat = "Men" if "/mens" in url else "Women" if "/womens" in url else "Unisex"
            
            # 4. DESCRIPTION (Full)
            desc_parts = []
            
            # Overview text
            overview = soup.select_one('div[itemprop="description"], .description-text')
            if overview:
                desc_parts.append(f"OVERVIEW:\n{overview.get_text(strip=True)}")
                
            # Features / Specs / Materials headers
            headers = soup.find_all(['h2', 'h3', 'h4'], string=lambda t: t and any(x in t.lower() for x in ['features', 'specs', 'materials']))
            for h in headers:
                parent = h.find_parent('div')
                if parent:
                    # distinct text from header
                    text = parent.get_text(strip=True).replace(h.get_text(strip=True), "")
                    desc_parts.append(f"\n{h.get_text(strip=True).upper()}:\n{text}")

            full_desc = "\n".join(desc_parts)

            return {
                "Product Name": name,
                "Price": price,
                "Category": cat,
                "Subcategory": "General", # Hard to get specific subcat without breadcrumbs, but URL usually has it
                "Full Description": full_desc,
                "URL": url
            }
            
        except Exception as e:
            # print(f"Error on {url}: {e}")
            return None

    async def run(self):
        async with async_playwright() as p:
            # HEADLESS=FALSE so you can see it working and solve CAPTCHAs
            browser = await p.chromium.launch(headless=False, channel="chrome")
            context = await browser.new_context(viewport={'width': 1280, 'height': 800})
            page = await context.new_page()

            # --- PHASE 1: HARVEST LINKS ---
            for url in self.start_urls:
                try:
                    print(f"\n--- Processing Category: {url} ---")
                    await page.goto(url, timeout=60000)
                    await self.close_popups(page)
                    
                    # 1. Scroll to bottom to get ALL products
                    await self.scroll_to_load_all(page)
                    
                    # 2. Extract all links
                    cat_name = "Men" if "mens" in url else "Women"
                    await self.collect_product_links(page, cat_name)
                    
                except Exception as e:
                    print(f"Skipping category {url} due to error: {e}")

            print(f"\n--- Harversting Complete. Found {len(self.product_urls)} unique products. Starting Scrape. ---")

            # --- PHASE 2: SCRAPE DETAILS ---
            # Convert set to list to iterate
            all_urls = list(self.product_urls)
            
            for i, prod_url in enumerate(all_urls):
                data = await self.scrape_detail_page(page, prod_url)
                
                if data:
                    # === VERIFICATION STEP ===
                    if not self.first_verified:
                        print("\n" + "="*50)
                        print(" TEST LINE: First Product Scraped")
                        print(f" Name: {data['Product Name']}")
                        print(f" Desc: {data['Full Description'][:100]}...")
                        print("="*50)
                        x = input(">> Type 'y' to continue scraping, anything else to abort: ")
                        if x.lower() != 'y': return
                        self.first_verified = True
                        print("Continuing...")
                    # =========================

                    self.scraped_data.append(data)
                    
                    # Stats for console
                    if data['Category'] == 'Men': self.mens_count += 1
                    else: self.womens_count += 1
                    
                    # Dashboard update (simple print per line)
                    print(f"[{i+1}/{len(all_urls)}] Scraped: {data['Product Name'][:30]}...")

                    # Save every 10
                    if len(self.scraped_data) % 10 == 0:
                        df = pd.DataFrame(self.scraped_data)
                        df.to_excel(self.output_file, index=False)

            # Final Save
            df = pd.DataFrame(self.scraped_data)
            df.to_excel(self.output_file, index=False)
            print(f"Done. Saved to {self.output_file}")
            
            await browser.close()

if __name__ == "__main__":
    scraper = PatagoniaForceScraper()
    asyncio.run(scraper.run())
import asyncio
import pandas as pd
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

class EcoalfProductScraper:
    def __init__(self, input_file="ecoalf_men_links.txt", output_file="ecoalf_men_final.xlsx"):
        self.input_file = input_file
        self.output_file = output_file
        self.data = []

    async def get_description(self, soup):
        """
        STRICT FILTER:
        1. Keeps [PRODUCT DETAILS] (from .product__description)
        2. Keeps [SUSTAINABILITY REPORT] (from the specific accordion)
        3. Removes EVERYTHING else (Care, Shipping, Returns, etc.)
        """
        full_text = []

        # --- 1. PRODUCT DETAILS (Main Text) ---
        intro = soup.select_one('.product__description')
        if intro:
            full_text.append(f"[PRODUCT DETAILS]\n{intro.get_text(strip=True)}")

        # --- 2. SUSTAINABILITY REPORT ONLY ---
        # Scan all potential accordion containers (details tags AND divs)
        containers = soup.select('details, div.product__accordion')
        
        for acc in containers:
            # Find the title element
            title_el = acc.select_one('summary, .accordion__title, h2.accordion__title')
            
            if title_el:
                title_text = title_el.get_text(strip=True)
                
                # CHECK: Is this the Sustainability Report?
                # We use a case-insensitive check to be safe
                if "sustainability" in title_text.lower():
                    
                    # Extract the content inside
                    content_el = acc.select_one('.accordion__content, .prose')
                    if content_el:
                        text = content_el.get_text(separator='\n', strip=True)
                        full_text.append(f"\n[SUSTAINABILITY REPORT]\n{text}")
                    
                    # We found sustainability, no need to keep checking this specific accordion
                    # (Logic: If we found it, we add it. If it's not sustainability, we do nothing/drop it)

        return "\n".join(full_text)

    async def run(self):
        try:
            with open(self.input_file, "r") as f:
                urls = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Loaded {len(urls)} links from {self.input_file}.")
        except FileNotFoundError:
            print(f"Error: Could not find {self.input_file}.")
            return

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()

            for i, url in enumerate(urls):
                print(f"[{i+1}/{len(urls)}] Scraping: {url}")
                try:
                    await page.goto(url, timeout=60000, wait_until="domcontentloaded")
                    
                    # Handle Cookie Banner (First iteration only)
                    if i == 0:
                        await asyncio.sleep(2)
                        try:
                            cookie_btn = page.locator('#onetrust-accept-btn-handler')
                            if await cookie_btn.is_visible():
                                await cookie_btn.click()
                                print("  > Cookie banner closed.")
                        except: pass

                    content = await page.content()
                    soup = BeautifulSoup(content, 'html.parser')

                    # --- 1. NAME ---
                    h1 = soup.select_one('h1.product__title')
                    name = h1.get_text(strip=True) if h1 else "N/A"

                    # --- 2. PRICE ---
                    price_div = soup.select_one('.price__regular')
                    price = price_div.get_text(strip=True) if price_div else "N/A"

                    # --- 3. CATEGORY ---
                    breadcrumbs = soup.select('li.breadcrumbs__item')
                    if breadcrumbs:
                        cat_list = [li.get_text(strip=True) for li in breadcrumbs]
                        category = " > ".join(cat_list)
                    else:
                        category = "N/A"

                    # --- 4. DESCRIPTION (Filtered) ---
                    description = await self.get_description(soup)

                    self.data.append({
                        "Name": name,
                        "Price": price,
                        "Category": category,
                        "Description": description,
                        "URL": url
                    })

                except Exception as e:
                    print(f"  Error scraping {url}: {e}")

                if (i+1) % 10 == 0:
                    self.save()

            await browser.close()
            self.save()
            print("Done!")

    def save(self):
        df = pd.DataFrame(self.data)
        df.to_excel(self.output_file, index=False)
        print(f"  > Saved data to {self.output_file}")

if __name__ == "__main__":
    # Ensure these filenames match your needs (Men vs Women)
    scraper = EcoalfProductScraper(input_file="ecoalf_men_links.txt", output_file="ecoalf_men_final.xlsx")
    asyncio.run(scraper.run())
import asyncio
import pandas as pd
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

class ProductDetailScraper:
    def __init__(self, input_file="womens_product_links.txt", output_file="patagonia_women_final.xlsx"):
        self.input_file = input_file
        self.output_file = output_file
        self.data = []

    async def get_description(self, soup):
        """
        Combines the main description and the accordion content, 
        EXCLUDING the 'fit' section as requested.
        """
        full_text = []

        intro = soup.select_one('div.pdp__content-description')
        if intro:
            full_text.append(f"[INTRO]\n{intro.get_text(strip=True)}")

        details_wrapper = soup.select_one('div.accordion-group--wrapper')
        if details_wrapper:
    
            fit_section = details_wrapper.select_one('div.accordion-group[data-pdp-accordion-fit=""]')
            if fit_section:
                fit_section.decompose() 

           
            raw_text = details_wrapper.get_text(separator='\n', strip=True)
            
        
            clean_text = "\n".join([line for line in raw_text.split('\n') if line.strip()])
            full_text.append(f"\n[DETAILS]\n{clean_text}")

        return "\n".join(full_text)

    async def run(self):
        try:
            with open(self.input_file, "r") as f:
                urls = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Loaded {len(urls)} links from file.")
        except FileNotFoundError:
            print(f"Error: Could not find {self.input_file}. Run the link harvester first!")
            return

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()

            for i, url in enumerate(urls):
                print(f"[{i+1}/{len(urls)}] Scraping: {url}")
                try:
                    await page.goto(url, timeout=60000, wait_until="domcontentloaded")
                    
                    if i == 0:
                        try: await page.click('#onetrust-accept-btn-handler', timeout=2000)
                        except: pass

                    content = await page.content()
                    soup = BeautifulSoup(content, 'html.parser')

                    h1 = soup.select_one('h1#product-title')
                    name = h1.get_text(strip=True) if h1 else "N/A"

                    price_span = soup.select_one('span[itemprop="price"]')
                    if not price_span:
                        price_span = soup.select_one('.prices .value')
                    price = price_span.get_text(strip=True) if price_span else "N/A"

                    breadcrumb = soup.select_one('ol.breadcrumb')
                    if breadcrumb:
                        cat_list = [li.get_text(strip=True) for li in breadcrumb.find_all('li')]
                        category = " > ".join(cat_list)
                    else:
                        category = "N/A"

            
                    description = await self.get_description(soup)

                    self.data.append({
                        "Name": name,
                        "Price": price,
                        "Category": category,
                        "Description": description,
                        "URL": url
                    })

                except Exception as e:
                    print(f"  Error: {e}")

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
    scraper = ProductDetailScraper()
    asyncio.run(scraper.run())
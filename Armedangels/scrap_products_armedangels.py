import asyncio
import pandas as pd
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

class ArmedAngelsScraper:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.data = []

    async def get_hidden_content(self, soup, button_text_identifier):
        """
        Finds the content linked to a button (Material or Transparency).
        The button has a 'for' attribute that links to the ID of the content.
        """
        try:
            label = soup.find(lambda tag: tag.name == "p" and button_text_identifier in tag.get_text())
            
            if label:
                button = label.find_parent('theme-modal-button')
                if button and button.has_attr('for'):
                    target_id = button['for']
                    
                    hidden_content = soup.find(id=target_id)
                    if hidden_content:
                        return hidden_content.get_text(separator='\n', strip=True)
            return None
        except Exception:
            return None

    async def run(self):
        try:
            with open(self.input_file, "r") as f:
                urls = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Loaded {len(urls)} links from {self.input_file}.")
        except FileNotFoundError:
            print(f"Error: Could not find {self.input_file}. Make sure to run the harvest script first.")
            return

        async with async_playwright() as p:

            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = await context.new_page()

            for i, url in enumerate(urls):
                print(f"[{i+1}/{len(urls)}] Scraping: {url}")
                try:
                    await page.goto(url, timeout=60000, wait_until="domcontentloaded")

                    if i == 0:
                        try:
                            await asyncio.sleep(2)
                            if await page.get_by_role("button", name="Accept all").is_visible():
                                await page.get_by_role("button", name="Accept all").click()
                            else:
                                await page.click('#usercentrics-root button[data-testid="uc-accept-all-button"]', timeout=2000)
                        except: pass

                    content = await page.content()
                    soup = BeautifulSoup(content, 'html.parser')

                    name_div = soup.select_one('.product-title-wrapper')
                    name = name_div.get_text(strip=True) if name_div else "N/A"

                    price_tag = soup.select_one('.variant-money')
                    price = price_tag.get_text(strip=True) if price_tag else "N/A"

                    breadcrumb_nav = soup.select_one('nav[aria-label="Breadcrumb"]')
                    if breadcrumb_nav:
                        items = [li.get_text(strip=True) for li in breadcrumb_nav.find_all('li')]
                        category = " > ".join([x for x in items if x and x != '/'])
                    else:
                        category = "N/A"

                    desc_parts = []

                    main_desc = soup.select_one('.extended-description')
                    if main_desc:
                        desc_parts.append(f"[OVERVIEW]\n{main_desc.get_text(separator='\n', strip=True)}")

                    material_text = await self.get_hidden_content(soup, "Material")
                    if material_text:
                        desc_parts.append(f"\n[MATERIAL & CARE]\n{material_text}")

                    transparency_text = await self.get_hidden_content(soup, "Transparency")
                    if transparency_text:
                        desc_parts.append(f"\n[TRANSPARENCY]\n{transparency_text}")

                    full_desc = "\n".join(desc_parts)

                    self.data.append({
                        "Name": name,
                        "Price": price,
                        "Category": category,
                        "Description": full_desc,
                        "URL": url
                    })

                except Exception as e:
                    print(f"  Error on {url}: {e}")

                if (i + 1) % 10 == 0:
                    self.save()

            await browser.close()
            self.save()
            print("Scraping Completed.")

    def save(self):
        df = pd.DataFrame(self.data)
        df.to_excel(self.output_file, index=False)
        print(f"  > Data saved to {self.output_file}")

if __name__ == "__main__":

    INPUT_FILE = "armedangels_women_links.txt"
    OUTPUT_FILE = "armedangels_women_final.xlsx"

    scraper = ArmedAngelsScraper(INPUT_FILE, OUTPUT_FILE)
    asyncio.run(scraper.run())
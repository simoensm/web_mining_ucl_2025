import asyncio
import pandas as pd
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import os

# --- CONFIGURATION ---
INPUT_FILE = ".ecoalf/ecoalf_men_links.txt"
OUTPUT_FILE = ".ecoalf/ecoalf_men_products.xlsx"

async def get_raw_description(soup):
    """Extracts product details and sustainability info from accordions."""
    full_text = []

    # 1. Product Description Intro
    intro = soup.select_one('.product__description')
    if intro:
        full_text.append(f"[PRODUCT DETAILS]\n{intro.get_text(separator=' ', strip=True)}")

    # 2. Accordions (looking for Sustainability specific content)
    containers = soup.select('details, div.product__accordion')
    
    for acc in containers:
        title_el = acc.select_one('summary, .accordion__title, h2.accordion__title')
        
        if title_el:
            title_text = title_el.get_text(strip=True)
            
            # Specific check for Sustainability sections
            if "sustainability" in title_text.lower():
                content_el = acc.select_one('.accordion__content, .prose')
                if content_el:
                    text = content_el.get_text(separator=' ', strip=True)
                    full_text.append(f"\n[SUSTAINABILITY REPORT]\n{text}")

    return "\n".join(full_text)

async def scrape_products():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r") as f:
        urls = [line.strip() for line in f.readlines() if line.strip()]

    data = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        print(f"--- Starting Scraping of {len(urls)} products ---")

        for i, url in enumerate(urls):
            try:
                await page.goto(url, timeout=60000, wait_until="domcontentloaded")
                
                # Cookie banner (OneTrust) - Only on first iteration
                if i == 0:
                    try:
                        await asyncio.sleep(2)
                        cookie_btn = page.locator('#onetrust-accept-btn-handler')
                        if await cookie_btn.is_visible():
                            await cookie_btn.click()
                    except: pass

                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')

                # Extraction Basique
                h1 = soup.select_one('h1.product__title')
                name = h1.get_text(strip=True) if h1 else "N/A"

                price_div = soup.select_one('.price__regular')
                price = price_div.get_text(strip=True) if price_div else "N/A"

                # Breadcrumb / Category
                breadcrumbs = soup.select('li.breadcrumbs__item')
                if breadcrumbs:
                    cat_list = [li.get_text(strip=True) for li in breadcrumbs]
                    category = " > ".join(cat_list)
                else:
                    category = "N/A"

                # Description
                description = await get_raw_description(soup)

                data.append({
                    "name": name,
                    "price": price,
                    "category": category,
                    "description": description,
                    "url": url
                })
                
                print(f"[{i+1}/{len(urls)}] OK: {name}")

            except Exception as e:
                print(f"[{i+1}/{len(urls)}] ERROR: {url} - {e}")

        await browser.close()

    # Sauvegarde
    df = pd.DataFrame(data)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"--- DONE. Saved raw data to {OUTPUT_FILE} ---")

if __name__ == "__main__":
    asyncio.run(scrape_products())
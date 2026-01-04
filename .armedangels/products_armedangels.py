import asyncio
import pandas as pd
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import os

INPUT_FILE = ".armedangels/armedangels_men_links.txt"
OUTPUT_FILE = ".armedangels/armedangels_men_products.xlsx"

async def get_raw_description(soup):
    """Combines all description parts (Overview, Details, Material, Transparency) into one text."""
    full_text = []

    main_desc = soup.select_one('.extended-description')
    if main_desc:
        full_text.append(f"[OVERVIEW]\n{main_desc.get_text(separator=' ', strip=True)}")

    def extract_hidden(identifiers):
        if isinstance(identifiers, str): identifiers = [identifiers]
        try:

            label = soup.find(lambda tag: tag.name == "p" and any(x in tag.get_text() for x in identifiers))
            if label:
                button = label.find_parent('theme-modal-button')
                if button and button.has_attr('for'):
                    target_id = button['for']
                    hidden_content = soup.find(id=target_id)
                    if hidden_content:
                        return hidden_content.get_text(separator=' ', strip=True)
        except: pass
        return None

    details_text = extract_hidden(["Details", "Fit", "Cut", "Shape"])

    if not details_text:
        try:
            drawers = soup.select('div.spacing-m.flex.col.gap-m.drawer-body')
            for drawer in drawers:
                text = drawer.get_text(separator=' ', strip=True)
                if any(keyword in text for keyword in ["Fit", "Length", "Model", "Cut", "size"]):
                    details_text = text
                    break
        except: pass

    if details_text:
        full_text.append(f"\n[DETAILS & FIT]\n{details_text}")

    material_text = extract_hidden("Material")
    if material_text:
        full_text.append(f"\n[MATERIAL]\n{material_text}")

    transparency_text = extract_hidden("Transparency")
    if transparency_text:
        full_text.append(f"\n[TRANSPARENCY]\n{transparency_text}")

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
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = await context.new_page()

        print(f"--- Starting Scraping of {len(urls)} products ---")

        for i, url in enumerate(urls):
            try:
                await page.goto(url, timeout=60000, wait_until="domcontentloaded")

                if i == 0:
                    try:
                        await asyncio.sleep(2)
                        if await page.get_by_role("button", name="Accept all").is_visible():
                            await page.get_by_role("button", name="Accept all").click()
                        else:
                            await page.click('#usercentrics-root button[data-testid="uc-accept-all-button"]', timeout=3000)
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

    df = pd.DataFrame(data)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"--- DONE. Saved raw data to {OUTPUT_FILE} ---")

if __name__ == "__main__":
    asyncio.run(scrape_products())
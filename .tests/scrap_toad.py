import asyncio
import random
from playwright.async_api import async_playwright
import pandas as pd

# CONFIGURATION
# The specific collection URL you requested
TARGET_URL = 'https://www.toadandco.com/collections/all-mens-clothing'
MAX_PRODUCTS = 5  # Keep this low to avoid an immediate IP Ban

async def scrape_toad_and_co():
    async with async_playwright() as p:
        # Launch browser in visible mode to debug
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        print(f"Navigating to {TARGET_URL}...")
        await page.goto(TARGET_URL, timeout=60000)

        # 1. Handle Popups (Toad&Co often has a discount modal)
        try:
            # Wait briefly to see if a popup appears (common generic selectors)
            # You might need to manually close it if the script gets stuck here
            await page.wait_for_selector('button[aria-label="Close"]', timeout=4000)
            await page.click('button[aria-label="Close"]')
            print("Closed popup.")
        except:
            print("No popup detected or auto-close failed (ignoring).")

        # 2. Extract Product Links
        # Toad&Co (Shopify) usually keeps product links in anchors containing '/products/'
        await page.wait_for_selector('a[href*="/products/"]', state='visible')
        
        # Get all links that look like product pages
        all_links = await page.eval_on_selector_all(
            'a[href*="/products/"]',
            'elements => elements.map(e => e.href)'
        )

        # Filter duplicates and valid product URLs
        unique_links = list(set([link for link in all_links if "/products/" in link]))
        print(f"Found {len(unique_links)} potential products. Scraping first {MAX_PRODUCTS}...")

        scraped_data = []

        for link in unique_links[:MAX_PRODUCTS]:
            print(f"Scraping: {link}")
            try:
                await page.goto(link, timeout=60000)
                
                # Random "human" pause
                await asyncio.sleep(random.uniform(2, 5))

                # --- EXTRACT DATA (Shopify Selectors) ---
                
                # 1. Name: Usually an H1 tag
                name = await page.locator('h1').first.inner_text()

                # 2. Price: Look for standard Shopify price classes
                price = "N/A"
                # Try the most common Shopify price selectors
                if await page.locator('.price-item--regular').count() > 0:
                    price = await page.locator('.price-item--regular').first.inner_text()
                elif await page.locator('.price__current').count() > 0:
                    price = await page.locator('.price__current').first.inner_text()
                elif await page.locator('.product__price').count() > 0:
                    price = await page.locator('.product__price').first.inner_text()

                # 3. Description: Usually in a Rich Text Editor (.rte) div or .product__description
                description = "N/A"
                if await page.locator('.product__description').count() > 0:
                    description = await page.locator('.product__description').first.inner_text()
                elif await page.locator('.rte').count() > 0:
                    description = await page.locator('.rte').first.inner_text()
                
                # Clean up the text
                description = description.replace('\n', ' ').strip()

                scraped_data.append({
                    'Name': name.strip(),
                    'Price': price.strip(),
                    'Description': description[:200] + "...", # Truncated
                    'URL': link
                })

            except Exception as e:
                print(f"Error scraping {link}: {e}")

        await browser.close()
        return scraped_data

if __name__ == "__main__":
    data = asyncio.run(scrape_toad_and_co())
    
    df = pd.DataFrame(data)
    print("\n--- Scraped Data ---")
    print(df)
    df.to_csv('toadandco_products.csv', index=False)
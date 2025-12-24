import asyncio
import random
from playwright.async_api import async_playwright
import pandas as pd

# CONFIGURATION
# Start with a specific category URL to test (safest approach)
TARGET_URL = 'https://www.patagonia.com/shop/mens-fleece' 
MAX_PRODUCTS = 5  # Limit strictly for testing to avoid IP bans

async def scrape_patagonia():
    async with async_playwright() as p:
        # Launch browser (headless=False lets you see the bot working)
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = await context.new_page()

        print(f"Navigating to {TARGET_URL}...")
        await page.goto(TARGET_URL, timeout=60000)

        # Handle "Cookie" or "Location" popups if they appear
        try:
            # Example: Wait for and close a generic modal if it exists
            # You might need to inspect the site to find the exact ID for the 'Close' button
            await page.wait_for_selector('button[class*="close"]', timeout=5000)
            await page.click('button[class*="close"]')
        except:
            pass

        # Wait for product grid to load
        await page.wait_for_selector('.product-tile', state='visible')

        # Extract product links from the category page
        # Note: Selectors (.product-tile a) depend on the site's current CSS
        product_links = await page.eval_on_selector_all(
            '.product-tile a', 
            'elements => elements.map(e => e.href)'
        )
        
        # Deduplicate links
        product_links = list(set(product_links))
        print(f"Found {len(product_links)} products. Scraping first {MAX_PRODUCTS}...")

        scraped_data = []

        for link in product_links[:MAX_PRODUCTS]:
            print(f"Scraping: {link}")
            try:
                # Go to individual product page
                await page.goto(link, timeout=60000)
                
                # Random delay to mimic human behavior (CRITICAL to avoid bans)
                await asyncio.sleep(random.uniform(2, 5))

                # --- EXTRACT DATA ---
                # These selectors must be updated based on "Inspect Element" on the live site
                
                # 1. Product Name (Usually an H1 tag)
                name = await page.locator('h1.product-name, h1').first.inner_text()
                
                # 2. Price (Look for classes like 'price', 'sales', 'value')
                # Try multiple common selectors
                price = "N/A"
                if await page.locator('.price .value').count() > 0:
                    price = await page.locator('.price .value').first.inner_text()
                elif await page.locator('.sales .value').count() > 0:
                    price = await page.locator('.sales .value').first.inner_text()
                
                # 3. Description
                description = "N/A"
                # Often in a div with class 'description-text' or similar
                if await page.locator('.description-text').count() > 0:
                    description = await page.locator('.description-text').first.inner_text()
                else:
                    # Fallback: grab meta description
                    description = await page.get_attribute('meta[name="description"]', 'content')

                scraped_data.append({
                    'Name': name.strip(),
                    'Price': price.strip(),
                    'Description': description.strip()[:200] + "...", # Truncate for display
                    'URL': link
                })

            except Exception as e:
                print(f"Failed to scrape {link}: {e}")

        await browser.close()
        
        return scraped_data

# Run the scraper
if __name__ == "__main__":
    data = asyncio.run(scrape_patagonia())
    
    # Save to CSV
    df = pd.DataFrame(data)
    print("\n--- Scraped Data ---")
    print(df)
    df.to_csv('patagonia_products.csv', index=False)
#Même principe que pour le script des hommes, mais cette fois pour la section femmes du site Patagonia.
import asyncio
from playwright.async_api import async_playwright

async def harvest_womens_links():
    url = "https://eu.patagonia.com/gb/en/shop/womens"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        print(f"--- Navigate to {url} ---")
        await page.goto(url, timeout=60000, wait_until="domcontentloaded")
      
        try:
            await page.click('#onetrust-accept-btn-handler', timeout=3000)
            print("  > Closed Cookie Banner")
        except:
            print("  > No cookie banner found")

        print("--- Starting Scroll & Load Process ---")
        
        previous_count = 0
        retries = 0
        
        while True:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
            
            load_more = page.locator("button.load-more, .search-results-footer button")
            if await load_more.count() > 0 and await load_more.first.is_visible():
                try:
                    await load_more.first.click(force=True)
                    await asyncio.sleep(2)
                    retries = 0
                except:
                    pass
            
            current_count = await page.locator('.product-tile__wrapper').count()
            
            if current_count > previous_count:
                print(f"  > Products loaded: {current_count}")
                previous_count = current_count
                retries = 0
            else:
                retries += 1
                print(f"  > No new products ({retries}/3)...")
            
            if retries >= 3:
                print("--- Reached end of page ---")
                break

        print("--- Extracting and Cleaning Links ---")
        
        raw_links = await page.eval_on_selector_all(
            '.product-tile__wrapper a[itemprop="url"]', 
            "elements => elements.map(e => e.href)"
        )
        
        unique_clean_links = set()
        
        for link in raw_links:
            clean_link = link.split('?')[0]
            unique_clean_links.add(clean_link)
        
        final_list = sorted(list(unique_clean_links))
        
        print(f"\nFOUND {len(final_list)} UNIQUE PRODUCTS (Colors merged).")
        
        output_file = "womens_product_links.txt"
        with open(output_file, "w") as f:
            for link in final_list:
                f.write(link + "\n")
                
        print(f"Saved cleaned list to '{output_file}'")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(harvest_womens_links())
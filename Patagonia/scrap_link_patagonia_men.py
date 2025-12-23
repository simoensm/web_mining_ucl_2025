import asyncio
from playwright.async_api import async_playwright

async def harvest_mens_links():
    url = "https://eu.patagonia.com/gb/en/shop/mens"
    
    async with async_playwright() as p:
        # headless=False so you can see the process
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        print(f"--- Navigate to {url} ---")
        await page.goto(url, timeout=60000, wait_until="domcontentloaded")
        
        # 1. Close Cookie Banner
        try:
            await page.click('#onetrust-accept-btn-handler', timeout=3000)
            print("  > Closed Cookie Banner")
        except:
            print("  > No cookie banner found")

        # 2. Scroll and Load All Products
        print("--- Starting Scroll & Load Process ---")
        
        previous_count = 0
        retries = 0
        
        while True:
            # Scroll to bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
            
            # Click "Load More" if exists
            load_more = page.locator("button.load-more, .search-results-footer button")
            if await load_more.count() > 0 and await load_more.first.is_visible():
                try:
                    await load_more.first.click(force=True)
                    await asyncio.sleep(2)
                    retries = 0
                except:
                    pass
            
            # Check if new products appeared
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

        # 3. Extract and CLEAN Links
        print("--- Extracting and Cleaning Links ---")
        
        raw_links = await page.eval_on_selector_all(
            '.product-tile__wrapper a[itemprop="url"]', 
            "elements => elements.map(e => e.href)"
        )
        
        # --- NEW LOGIC: REMOVE DUPLICATES AND COLORS ---
        unique_clean_links = set()
        
        for link in raw_links:
            # Split URL at '?' to remove query params (colors, sizes, etc.)
            clean_link = link.split('?')[0]
            unique_clean_links.add(clean_link)
        
        # Sort for neatness
        final_list = sorted(list(unique_clean_links))
        
        print(f"\nFOUND {len(final_list)} UNIQUE PRODUCTS (Colors merged).")
        
        # 4. Save to file
        with open("mens_product_links.txt", "w") as f:
            for link in final_list:
                f.write(link + "\n")
                
        print(f"Saved cleaned list to 'mens_product_links.txt'")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(harvest_mens_links())
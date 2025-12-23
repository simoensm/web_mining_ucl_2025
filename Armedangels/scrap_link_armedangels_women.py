import asyncio
from playwright.async_api import async_playwright

async def harvest_armedangels_women():
    # 1. New URL for Women
    url = "https://www.armedangels.com/en-be/collections/women"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        print(f"--- Navigate to {url} ---")
        await page.goto(url, timeout=60000, wait_until="domcontentloaded")
        
        # 2. Handle Cookie Banner
        print("  > Checking for Cookie Banner...")
        try:
            await asyncio.sleep(3)
            accept_btn = page.get_by_role("button", name="Accept all")
            if await accept_btn.count() > 0:
                await accept_btn.click()
                print("  > Cookie Banner Accepted (Method A)")
            else:
                await page.click('#usercentrics-root button[data-testid="uc-accept-all-button"]', timeout=3000)
                print("  > Cookie Banner Accepted (Method B)")
        except:
            print("  > No cookie banner found or could not click.")

        # 3. Scroll and Load All Products
        print("--- Starting Scroll & Load Process ---")
        
        previous_link_count = 0
        retries = 0
        
        while True:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
            
            try:
                load_more = page.locator("button:has-text('Load more'), button.btn--load-more")
                if await load_more.count() > 0 and await load_more.first.is_visible():
                    print("  > Clicking 'Load More'...")
                    await load_more.first.click(force=True)
                    await asyncio.sleep(3)
                    retries = 0
            except:
                pass
            
            current_links = await page.locator('a[href*="/det/"]').count()
            
            if current_links > previous_link_count:
                print(f"  > Products loaded: {current_links}")
                previous_link_count = current_links
                retries = 0
            else:
                retries += 1
                print(f"  > No new products ({retries}/4)...")
            
            if retries >= 4:
                print("--- Reached end of page ---")
                break

        # 4. Extract and CLEAN Links
        print("--- Extracting and Cleaning Links ---")
        
        raw_links = await page.eval_on_selector_all(
            'a[href*="/det/"]', 
            "elements => elements.map(e => e.href)"
        )
        
        unique_clean_links = set()
        
        for link in raw_links:
            if "/det/" in link:
                clean_link = link.split('?')[0]
                unique_clean_links.add(clean_link)
        
        final_list = sorted(list(unique_clean_links))
        
        print(f"\nFOUND {len(final_list)} UNIQUE PRODUCTS.")
        
        output_file = "armedangels_women_links.txt"
        with open(output_file, "w") as f:
            for link in final_list:
                f.write(link + "\n")
                
        print(f"Saved cleaned list to '{output_file}'")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(harvest_armedangels_women())
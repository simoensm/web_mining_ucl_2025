import asyncio
from playwright.async_api import async_playwright

TARGET_URL = "https://ecoalf.com/en/collections/hombre" # mujer
OUTPUT_FILE = ".ecoalf/ecoalf_men_links.txt" # women

async def harvest_links():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        print(f"--- 1. Navigate to {TARGET_URL} ---")
        await page.goto(TARGET_URL, timeout=60000, wait_until="domcontentloaded")
        
        try:
            cookie_btn = page.locator("#onetrust-accept-btn-handler, button:has-text('Accept'), button:has-text('Allow All')")
            if await cookie_btn.count() > 0:
                await cookie_btn.first.click(timeout=3000)
                print("   > Cookie banner closed")
        except:
            print("   > No cookie banner or timed out")

        print("--- 2. Scrolling & Loading ---")
        previous_count = 0
        retries = 0
        
        while True:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)

            load_more = page.locator('text=/View More|Load More/i')
            if await load_more.count() > 0 and await load_more.first.is_visible():
                try:
                    await load_more.first.click(force=True)
                    await asyncio.sleep(2)
                    retries = 0
                except: pass

            current_count = await page.locator('a[href*="/products/"]').count()
            
            if current_count > previous_count:
                print(f"   > Products found: {current_count}")
                previous_count = current_count
                retries = 0
            else:
                retries += 1
                print(f"   > Waiting... ({retries}/3)")
            
            if retries >= 3:
                break

        print("--- 3. Extracting Links ---")
        raw_links = await page.eval_on_selector_all(
            'a[href*="/products/"]', 
            "elements => elements.map(e => e.href)"
        )
        
        unique_links = sorted(list(set([link.split('?')[0] for link in raw_links])))
        
        print(f"   > Found {len(unique_links)} unique products.")
        
        with open(OUTPUT_FILE, "w") as f:
            for link in unique_links:
                f.write(link + "\n")
                
        print(f"--- DONE. Saved to {OUTPUT_FILE} ---")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(harvest_links())
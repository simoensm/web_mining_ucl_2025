import asyncio
from playwright.async_api import async_playwright

TARGET_URL = "https://www.armedangels.com/en-be/collections/women" # women
OUTPUT_FILE = ".armedangels/armedangels_women_links.txt" # women

async def harvest_links():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        print(f"--- 1. Navigate to {TARGET_URL} ---")
        await page.goto(TARGET_URL, timeout=60000, wait_until="domcontentloaded")
        
        try:
            await page.click('#usercentrics-root', timeout=3000) 
            print("   > Cookie banner interaction attempted")
        except:
            print("   > No cookie banner or timed out")

        print("--- 2. Scrolling & Loading ---")
        previous_count = 0
        retries = 0
        
        while True:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)

            load_more = page.locator("button.load-more-selector-placeholder") 
            if await load_more.count() > 0 and await load_more.first.is_visible():
                try:
                    await load_more.first.click(force=True)
                    await asyncio.sleep(2)
                    retries = 0
                except: pass

            current_count = await page.locator("a.product-image[href*='/products/']").count()
            
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
        raw_data = await page.eval_on_selector_all(
            "a.product-image[href*='/products/']", 
            """elements => elements.map(e => {
                const rawId = e.getAttribute('data-colorway-number'); 
                const styleId = rawId ? rawId.split('-')[0] : 'UNKNOWN';
                return {
                    url: e.href,
                    style_id: styleId
                };
            })"""
        )
        
        unique_links = []
        seen_style_ids = set()

        for item in raw_data:
            style_id = item['style_id']
            url = item['url']

            if style_id != 'UNKNOWN':
                if style_id not in seen_style_ids:
                    seen_style_ids.add(style_id)
                    unique_links.append(url)
            else:
                if url not in unique_links:
                    unique_links.append(url)
        
        unique_links = sorted(unique_links)
        print(f"   > Found {len(unique_links)} unique styles (deduplicated by Style ID).")
        
        with open(OUTPUT_FILE, "w") as f:
            for link in unique_links:
                f.write(link + "\n")
                
        print(f"--- DONE. Saved to {OUTPUT_FILE} ---")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(harvest_links())
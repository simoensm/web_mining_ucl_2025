import asyncio
from playwright.async_api import async_playwright

class EcoalfCollector:
    def __init__(self, start_url, output_file):
        self.start_url = start_url
        self.output_file = output_file
        self.seen_slugs = set()
        self.final_links = []

    async def close_cookie_banner(self, page):
        print("  > Handling Cookie Banner...")
        try:
            # Wait for banner to potentially appear
            await asyncio.sleep(2)
            
            # Common OneTrust button ID or generic 'Accept' text
            cookie_btn = page.locator('#onetrust-accept-btn-handler, button:has-text("Accept"), button:has-text("Allow")')
            if await cookie_btn.count() > 0 and await cookie_btn.first.is_visible():
                await cookie_btn.first.click()
                print("  > Clicked Cookie 'Accept' button.")
            
            await asyncio.sleep(1)
        except Exception as e:
            print(f"  > Cookie banner skipped: {e}")

    async def extract_products(self, page):
        """
        Target based on user input: <a href="/en/products/..." class="card-product__color ...">
        We look for ANY <a> tag that contains '/products/' to be safe.
        """
        # Check if any matching links exist
        count = await page.locator('a[href*="/products/"]').count()
        if count == 0:
            return []

        # Extract data
        return await page.evaluate("""
            () => {
                // Select all anchors with '/products/' in the href
                // This matches the href pattern you provided: /en/products/bayonaalf...
                const elements = Array.from(document.querySelectorAll('a[href*="/products/"]'));
                
                return elements.map(e => {
                    return {
                        url: e.href,
                        // Create ID from the last part of URL (e.g., bayonaalf-knit-man...)
                        style_id: e.href.split('/').pop().split('?')[0]
                    };
                });
            }
        """)

    async def run(self):
        async with async_playwright() as p:
            # Launch with headless=False to see the browser
            browser = await p.chromium.launch(headless=False) 
            context = await browser.new_context()
            page = await context.new_page()

            print(f"--- Navigating to {self.start_url} ---")
            await page.goto(self.start_url, timeout=60000, wait_until='domcontentloaded')
            
            await self.close_cookie_banner(page)

            print("--- Waiting for product links ---")
            try:
                # UPDATED: Wait for the specific link pattern you found
                await page.wait_for_selector('a[href*="/products/"]', timeout=30000)
                print("  > Product grid loaded successfully.")
                
            except Exception as e:
                print(f"Error: Timed out waiting for products. {e}")
                await browser.close()
                return

            print("--- Starting Collection ---")
            no_new_products_attempts = 0
            
            while True:
                # 1. Scroll to the bottom to trigger lazy loading
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(2) # Give it time to render

                # 2. Extract
                current_batch = await self.extract_products(page)
                
                new_styles_found = 0
                for item in current_batch:
                    if item['style_id'] not in self.seen_slugs:
                        self.seen_slugs.add(item['style_id'])
                        self.final_links.append(item['url'])
                        new_styles_found += 1
                
                # 3. Status Report
                if new_styles_found > 0:
                    print(f"  > +{new_styles_found} NEW Styles found. (Total: {len(self.final_links)})")
                    no_new_products_attempts = 0
                else:
                    no_new_products_attempts += 1
                    print(f"  > No new styles. Attempt {no_new_products_attempts}/5")

                # 4. Stop if we haven't found anything new in 5 attempts
                if no_new_products_attempts >= 5:
                    print("--- Finished: No new styles found. ---")
                    break

                # 5. Click 'View More' if it exists (Generic check)
                try:
                    # Looks for button or link with text 'View More' or 'Load More'
                    load_more = page.locator('text=/View More|Load More/i')
                    if await load_more.count() > 0 and await load_more.first.is_visible():
                         print("  > Clicking 'View More'...")
                         await load_more.first.click()
                         await asyncio.sleep(3)
                except:
                    pass

            # 6. Save
            print(f"\nSAVING {len(self.final_links)} LINKS TO {self.output_file}...")
            with open(self.output_file, "w") as f:
                for link in sorted(self.final_links):
                    f.write(link + "\n")
            
            print("Done.")
            await browser.close()

if __name__ == "__main__":
    URL = "https://ecoalf.com/en/collections/hombre"
    FILE = "ecoalf_men_links.txt"
    
    collector = EcoalfCollector(URL, FILE)
    asyncio.run(collector.run())
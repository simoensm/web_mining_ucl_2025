import asyncio
from playwright.async_api import async_playwright

class ArmedAngelsProCollector:
    def __init__(self, start_url, output_file):
        self.start_url = start_url
        self.output_file = output_file
        
        self.seen_style_ids = set()
        self.final_links = []

    async def close_cookie_banner(self, page):
        print("  > Handling Cookie Banner...")
        try:
            await page.wait_for_selector('#usercentrics-root', state="attached", timeout=5000)
            await page.evaluate("""() => {
                const root = document.querySelector('#usercentrics-root');
                if (root && root.shadowRoot) {
                    const btn = root.shadowRoot.querySelector('button[data-testid="uc-accept-all-button"]');
                    if (btn) btn.click();
                }
            }""")
            await asyncio.sleep(2)
        except:
            pass

    async def extract_products(self, page):
        """
        Extracts Style IDs directly from the HTML data attributes.
        Returns: list of {url, style_id, full_name}
        """
        return await page.eval_on_selector_all(
            "a.product-image[href*='/products/']", 
            """elements => elements.map(e => {
                const rawId = e.getAttribute('data-colorway-number'); 
                // rawId example: "30002888-1509"
                // We want "30002888" (The Style ID)
                const styleId = rawId ? rawId.split('-')[0] : 'UNKNOWN';
                
                return {
                    url: e.href,
                    style_id: styleId,
                    handle: e.getAttribute('data-product-handle') || 'unknown-handle'
                };
            })"""
        )

    async def run(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()

            print(f"--- Navigating to {self.start_url} ---")
            await page.goto(self.start_url, timeout=60000, wait_until='domcontentloaded')
            await self.close_cookie_banner(page)

            print("--- Starting Pro Collection (Grouping by Style ID) ---")
            
            no_new_products_attempts = 0
            
            while True:
        
                current_batch = await self.extract_products(page)
                
                new_styles_found = 0
                
                for item in current_batch:
                    style_id = item['style_id']
                    url = item['url']
                    handle = item['handle']
                    
                    if style_id != 'UNKNOWN':
                        if style_id not in self.seen_style_ids:
                           
                            self.seen_style_ids.add(style_id)
                            self.final_links.append(url)
                            new_styles_found += 1
                            
                        else:
                            
                            pass
                            
                    else:
                       
                        if url not in self.final_links:
                             self.final_links.append(url)
                             new_styles_found += 1

          
                if new_styles_found > 0:
                    print(f"  > +{new_styles_found} NEW Styles found. (Total Unique: {len(self.final_links)})")
                    no_new_products_attempts = 0
                else:
                    no_new_products_attempts += 1
                    print(f"  > No new styles found. Scrolling... ({no_new_products_attempts}/5)")

                if no_new_products_attempts >= 5:
                    print("--- Finished: No new styles found for 5 consecutive scrolls. ---")
                    break

                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1.5)
                
                try:
                    buttons = page.locator("button")
                    count = await buttons.count()
                    for i in range(count):
                        btn = buttons.nth(i)
                        txt = await btn.inner_text()
                        if txt and any(x in txt.lower() for x in ['load more', 'show more', 'mehr anzeigen']):
                            if await btn.is_visible():
                                await btn.click(force=True)
                                await asyncio.sleep(2)
                                break
                except: pass

            print(f"\nSAVING {len(self.final_links)} UNIQUE STYLE LINKS TO {self.output_file}...")
            
            with open(self.output_file, "w") as f:
                for link in sorted(self.final_links):
                    f.write(link + "\n")
            
            print("Done.")
            await browser.close()

if __name__ == "__main__":

    URL = "https://www.armedangels.com/en-be/collections/men"
    FILE = "armedangels_men_links.txt"
    

    collector = ArmedAngelsProCollector(URL, FILE)
    asyncio.run(collector.run())
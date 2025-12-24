import requests
import time
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

TIMEOUT = 15
ua = UserAgent()


def get_headers():
    return {
        "User-Agent": ua.random,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://www.google.com/"
    }


def check_robots_txt(url):
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    try:
        r = requests.get(robots_url, timeout=TIMEOUT)
        if r.status_code != 200:
            return "No robots.txt"

        if "disallow: /" in r.text.lower():
            return "Blocks all bots"

        return "Allows scraping (partial/full)"

    except Exception:
        return "Robots unreachable"


def looks_like_block_page(html):
    block_signals = [
        "captcha",
        "verify you are human",
        "access denied",
        "unusual traffic",
        "cloudflare",
        "perimeterx"
    ]
    html_lower = html.lower()
    return any(signal in html_lower for signal in block_signals)


def content_is_js_heavy(html):
    soup = BeautifulSoup(html, "html.parser")
    text_length = len(soup.get_text(strip=True))
    script_count = len(soup.find_all("script"))
    return text_length < 200 and script_count > 5


def try_scrape(url):
    report = {
        "url": url,
        "success": False,
        "method": None,
        "notes": [],
        "robots": check_robots_txt(url)
    }

    session = requests.Session()

    for attempt in range(1, 4):
        try:
            headers = get_headers()
            response = session.get(url, headers=headers, timeout=TIMEOUT)
            html = response.text

            if response.status_code != 200:
                report["notes"].append(f"HTTP {response.status_code}")
                time.sleep(attempt * 2)
                continue

            if looks_like_block_page(html):
                report["notes"].append("Blocked by bot protection")
                time.sleep(attempt * 3)
                continue

            if content_is_js_heavy(html):
                report["notes"].append("JS-heavy site → Selenium needed")
                report["method"] = "Selenium / Playwright"
                return report

            # Success
            report["success"] = True
            report["method"] = f"Requests + headers (attempt {attempt})"
            return report

        except requests.exceptions.RequestException as e:
            report["notes"].append(str(e))
            time.sleep(attempt * 2)

    report["method"] = "Blocked or protected"
    return report


def test_websites(websites):
    results = []
    for site in websites:
        print(f"🔍 Testing {site}")
        result = try_scrape(site)
        results.append(result)
    return results


if __name__ == "__main__":
    websites = [
        "https://www.armedangels.com/"
    ]

    results = test_websites(websites)

    print("\n📊 FINAL REPORT\n" + "=" * 60)
    for r in results:
        print(f"""
URL: {r['url']}
Robots.txt: {r['robots']}
Scraping Success: {r['success']}
Recommended Method: {r['method']}
Notes: {', '.join(r['notes']) if r['notes'] else 'None'}
""")
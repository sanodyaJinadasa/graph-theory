"""
Scrape real EV charging station data from GreenEV's public location directory.

SETUP:
    pip install selenium webdriver-manager beautifulsoup4 pandas

This uses Selenium because the station list is likely rendered by JavaScript
(a React/Vue-style app), not present in the raw HTML. If you find a JSON API
endpoint via the browser's Network tab instead, that approach is much faster
and more reliable -- use that instead if you can.

USAGE:
    python scrape_greenev.py
"""

import os

# Unset proxy environment variables for the Python process
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Rest of your script...

from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


import re
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

URL = "https://greenev.lk/location.html"


def scrape_stations(url):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1400,2000")  # taller window helps load more of the list

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(4)  # let the JS app finish its initial render

        # The station list looks scrollable (a side panel with a scrollbar in your
        # screenshot). Try scrolling it multiple times to force lazy-loaded entries
        # to render. We scroll the whole page AND look for an inner scrollable panel.
        for _ in range(15):
            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(0.5)

        # Grab all visible text on the page
        full_text = driver.find_element(By.TAG_NAME, "body").text

    finally:
        driver.quit()

    # Save raw text for debugging/inspection regardless of whether parsing succeeds
    with open("greenev_raw_page_text.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    # Pattern: Name line, then "Contact: ...", then "Status: ...", then a tags line
    pattern = re.compile(
        r'(?P<name>.+?)\s*\n'
        r'Contact:\s*(?P<phone>[\d\s+]+)\s*\n'
        r'Status:\s*(?P<status>.+?)\s*\n'
        r'(?P<tags>.+?)\s*\n',
        re.MULTILINE
    )

    matches = list(pattern.finditer(full_text))
    stations = []
    for m in matches:
        stations.append({
            "name": m.group("name").strip(),
            "phone": m.group("phone").strip(),
            "status": m.group("status").strip(),
            "tags": m.group("tags").strip(),
        })

    return stations


if __name__ == "__main__":
    results = scrape_stations(URL)
    df = pd.DataFrame(results)
    df.to_csv("greenev_stations.csv", index=False)
    print(f"Extracted {len(results)} stations -> greenev_stations.csv")
    print(f"Full raw page text also saved to greenev_raw_page_text.txt for debugging")
    if len(results) == 0:
        print("\nNo matches found. Open greenev_raw_page_text.txt and check whether "
              "the station cards actually appear in that text, and whether the "
              "Contact:/Status: format matches exactly (e.g. maybe there's a dash "
              "'–' instead of ':' somewhere, or extra whitespace).")
    else:
        print(df.head(10).to_string())

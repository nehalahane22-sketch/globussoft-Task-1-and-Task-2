# ============================================================
# TASK 1 - Amazon India Laptop Scraper
# Run this entire file in Google Colab
# ============================================================

# ----------------------------
# STEP 1: Install dependencies
# ----------------------------
# Run this in a Colab cell first:
# !pip install requests beautifulsoup4 pandas lxml

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import random

# ----------------------------
# STEP 2: Setup headers
# (Mimics a real browser so Amazon doesn't block us)
# ----------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

# ----------------------------
# STEP 3: Function to scrape one page
# ----------------------------
def scrape_page(page_number):
    """Scrapes one page of Amazon laptop search results."""
    url = f"https://www.amazon.in/s?k=laptops&page={page_number}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] Could not fetch page {page_number}: {e}")
        return []

    soup = BeautifulSoup(response.content, "lxml")
    
    # All product cards on the page
    product_cards = soup.find_all("div", {"data-component-type": "s-search-result"})
    print(f"  Found {len(product_cards)} products on page {page_number}")
    
    products = []

    for card in product_cards:
        # ---- Title ----
        title_tag = card.find("h2")
        title = title_tag.get_text(strip=True) if title_tag else "N/A"

        # ---- Price ----
        price_tag = card.find("span", class_="a-price-whole")
        price = price_tag.get_text(strip=True).replace(",", "") if price_tag else "N/A"
        price = f"₹{price}" if price != "N/A" else "N/A"

        # ---- Rating ----
        rating_tag = card.find("span", class_="a-icon-alt")
        rating = rating_tag.get_text(strip=True) if rating_tag else "N/A"

        # ---- Image URL ----
        img_tag = card.find("img", class_="s-image")
        image_url = img_tag["src"] if img_tag and img_tag.get("src") else "N/A"

        # ---- Product URL ----
        link_tag = card.find("a", class_="a-link-normal s-no-outline")
        if not link_tag:
            link_tag = card.find("h2").find("a") if card.find("h2") else None
        product_url = ("https://www.amazon.in" + link_tag["href"]) if link_tag else "N/A"

        # ---- Ad or Organic ----
        # Sponsored products have a specific label
        sponsored_tag = card.find("span", string=lambda t: t and "Sponsored" in t)
        ad_or_organic = "Ad" if sponsored_tag else "Organic"

        products.append({
            "Title":       title,
            "Price":       price,
            "Rating":      rating,
            "Image URL":   image_url,
            "Product URL": product_url,
            "Ad/Organic":  ad_or_organic,
            "Page":        page_number,
        })

    return products

# ----------------------------
# STEP 4: Scrape multiple pages
# ----------------------------
def scrape_amazon_laptops(total_pages=3):
    """
    Scrapes Amazon India laptop listings across multiple pages.
    Args:
        total_pages: number of pages to scrape (each page ~20 products)
    Returns:
        DataFrame with all products
    """
    all_products = []
    
    print(f"Starting scrape for {total_pages} pages...\n")
    
    for page in range(1, total_pages + 1):
        print(f"Scraping page {page}/{total_pages}...")
        page_products = scrape_page(page)
        all_products.extend(page_products)
        
        # Random delay between requests (1.5 to 3.5 seconds)
        # This is polite scraping and avoids getting blocked
        if page < total_pages:
            delay = random.uniform(1.5, 3.5)
            print(f"  Waiting {delay:.1f}s before next page...\n")
            time.sleep(delay)
    
    return pd.DataFrame(all_products)

# ----------------------------
# STEP 5: Save to CSV with timestamp
# ----------------------------
def save_to_csv(df):
    """Saves DataFrame to a CSV file with current timestamp in filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"amazon_laptops_{timestamp}.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved {len(df)} products to: {filename}")
    return filename

# ----------------------------
# STEP 6: MAIN - Run everything
# ----------------------------
if __name__ == "__main__":
    # Scrape 3 pages (~60 products). Change to more if needed.
    df = scrape_amazon_laptops(total_pages=3)
    
    if df.empty:
        print("\n⚠️  No data scraped. Amazon may be blocking requests.")
        print("    Try: adding more headers, using a VPN, or reducing pages.")
    else:
        print(f"\n📦 Total products scraped: {len(df)}")
        print("\nSample data (first 3 rows):")
        print(df.head(3).to_string())
        
        # Save CSV
        saved_file = save_to_csv(df)
        
        # In Colab: download the file automatically
        try:
            from google.colab import files
            files.download(saved_file)
            print(f"📥 Download started for: {saved_file}")
        except ImportError:
            print(f"(Not in Colab - file saved locally as: {saved_file})")

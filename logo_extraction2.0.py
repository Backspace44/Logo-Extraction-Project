'''
LOGO EXTRACTION & CLUSTERING TOOL

Key Functionality:
- Scrapes logos from websites using multiple detection methods
- Converst various formats (SVG/ICO) to PNG
- Groups visually similar logos using histogram analysis

Main Components:
1. URL Handling      - Read input, normalize URLs, get HTML content
2. Logo Detection    - Parse HTML for logos and favicons
3. Download System   - Handle different image formats and conversions
4. Image Analysis    - Histogeram-based similarity clustering
'''

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import os
import urllib.parse
from urllib.parse import urljoin, urlparse
import time
import logging
import random
import re
from playwright.sync_api import sync_playwright
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import cairosvg
import hashlib
import argparse

#------------------------
# CORE CONFIGURATION
#------------------------
NUM_BINS = (16, 16, 16)           # HSV histogram dimensions (Hue, Saturation, Value)
DISTANCE_THRESHOLD = 0.1          # Maximum similarity distance for clustering


#------------------------
# DATA INPUT HANDLING
#------------------------
def read_data(file_path, url_column):
    """
    Read data from either Parquet of CSV file based on file extension.
    Returns list of URLs from the specified column.
    """
    try:
        # Check file existence
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Input file '{file_path}' does not exist")

        # Check read permissions
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Insufficient permissions to read '{file_path}'")

        # Determine file type from extension
        if file_path.lower().endswith('.parquet'):
            df = pd.read_parquet(file_path, engine="pyarrow")
            file_ext = Path(file_path).suffix.lower()
        elif file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
            file_ext = Path(file_path).suffix.lower()
        else:
            raise ValueError("Unsupported file format. Only .parquet and .csv are supported.")

        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Invalid file content: Empty or malformed data frame")
        # Verify URL column exists
        if url_column not in df.columns:
            raise ValueError(f"Column '{url_column}' not found in file.")

        return df[url_column].astype(str).tolist()

    except FileNotFoundError as e:
        logging.error(f"File Error: {str(e)}")
        print(f"ERROR: {str(e)}")
        return []

    except PermissionError as e:
        logging.error(f"Permissions Error: {str(e)}")
        print(f"PERMISSION ERROR: {str(e)}")
        return []

    except pd.errors.ParserError as e:
        logging.error(f"Parsing Error: {str(e)}")
        print(f"FILE FORMAT ERROR: Malformed {file_ext.upper()} file")
        return []

    except (KeyError, ValueError) as e:
        logging.error(f"Data Validation Error: {str(e)}")
        print(f"DATA ERROR: {str(e)}")
        return []

    except Exception as e:
        logging.error(f"Unexpected Error: {str(e)}")
        print(f"UNEXPECTED ERROR: {str(e)}")
        return []


#------------------------
# BROWSER AUTOMATION
#------------------------
def _init_browser(proxies=None):
    """Initialize Playwright browser with random user agent and proxy"""
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(
        headless=True,
        proxy=(
            {"server": proxies['http']}
            if proxies and "http" in proxies
            else None
        ),
        args=[
            '--disable-blink-features=AutomationControlled',
            '--no-sandbox'
        ]
    )

    user_agent = random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
    ])
    context = browser.new_context(
        user_agent=user_agent,
        java_script_enabled=True,
        ignore_https_errors=True
    )
    return browser, context

def get_html(url, retries=3, proxies=None, use_playwright=False):
    """Fetch HTML content using either requests or Playwright.
    Handles reties, SSL errors, and user-agent rotation."""

    if use_playwright:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    timeout=60000,
                    args=['--disable-blink-features=AutomationControlled']
                )
                page = browser.new_page()
                page.goto(url, timeout=60000)
                content = page.content()
                browser.close()
                return content
        except Exception as e:
            print(f"Playwright Error: {e}")
            return None

    user_agents= [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like"
        " Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, "
        "like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome"
        "/92.0.4515.131 Safari/537.36"
    ]

    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br"
    }

    if retries <= 0:
        print(f"Abort {url} after 3 failed attempts")
        return None

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=(8, 25),
            proxies=proxies,
            allow_redirects=True,
            verify=True
        )
        response.raise_for_status()
        return response.text

    except requests.exceptions.SSLError:
        print(f"Retry with HTTP for {url} ({retries-1} remaining tries)")
        return get_html(
            url.replace("https://", "http://", 1),
            retries=retries-1,
            proxies=proxies,
            use_playwright=use_playwright
        )
    except requests.exceptions.RequestException as e:
        print(f"Error: { type(e).__name__} at {url}. Remaining attempts: {retries-1}")
        return get_html(url, retries=retries-1, proxies=proxies, use_playwright=use_playwright) if retries > 0 else None

def normalize_url(url):
    """Ensures URLs have proper HTTP/S prefix and standardized format."""
    url = str(url).strip()
    if not url.startswith(("http://", "https://")):
        return f"https://{url}"
    return url

#------------------------
# LOGO PROCESSING CORE
#------------------------
def extract_logo_url(html, base_url):
    """Extracts logo URLs from HTML using multi-layered detection:
    1. <img> tags with logo-related attributes
    2. Favicon links
    3. OpenGraph/Twitter meta tags
    4. CSS background images
    5. Fallback to favicon.ico"""
    soup = BeautifulSoup(html, 'html.parser')
    parsed_url = urlparse(base_url)

    base_domain = parsed_url.netloc.replace("www.", "", 1)
    base_scheme = parsed_url.scheme or "https"

    logo_url = None

    sources = [
        {
            "tags": ["img"],
            "attr": "src",
            "keywords": [r"logo", r"brand", r"header", r"site"],
            "regex": True
        },

        {
            "tags": ["link"],
            "attr": "href",
            "keywords": [r"icon", r"shortcut"],
            "attr_filters": {"rel": True}
        },

        {
            "tags": ["meta"],
            "attr": "content",
            "keywords": [r"og:image", r"twitter:image"]
        },

        {
            "tags": ["svg"],
            "attr": "data-logo",
            "keywords": [r"logo"]
        },

        {
            "tags": ["div"],
            "attr": "style",
            "keywords": [r"background-image:\s*url\((.*?)\)"],
            "regex": True,
            "is_css": True
        }
    ]

    for source in sources:
        tags = source["tags"]
        attr = source["attr"]
        keywords = source["keywords"]
        regex =source.get("regex", False)
        is_css = source.get("is_css", False)
        attr_filters = source.get("attr_filters", {})

        for tag in tags:
            elements = soup.find_all(tag, {**{attr: True}, **attr_filters})
            for el in elements:
                value = el.get(attr, "").strip()

                if not value:
                    continue

                if is_css:
                    match = re.search(keywords[0], value, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip('"\'')

                match_found = False
                if regex:
                    for pattern in keywords:
                        if re.search(pattern, value, re.IGNORECASE):
                            match_found = True
                            break
                else:
                    if any(kw.lower() in value.lower() for kw in keywords):
                        match_found = True

                if match_found:
                    if value.startswith("//"):
                        return f"{parsed_url.scheme}:{value}"
                    elif value.startswith(("http://", "https://")):
                        return value
                    else:
                        return urljoin(f"{base_scheme}://{parsed_url.netloc}/", value)

    for img in soup.find_all('img', src=True):
        src = img['src']
        if src:
            return urljoin(f"{base_scheme}://{base_domain}", src)

    if not logo_url:
        favicon_link = soup.find("link", rel=lambda x: x and x.lower() in ["icon", "shortcut icon"])

        if favicon_link:
            logo_url = favicon_link.get('href', '')
        else:
            logo_url = f"{base_scheme}://{base_domain}/favicon.ico"

    if logo_url:
        if logo_url.startswith("//"):
            return f"{base_scheme}:{logo_url}"
        elif logo_url.startswith(("http://", "https://")):
            return logo_url
        else:
            return urljoin(f"{base_scheme}://{base_domain}/", logo_url)


    return None

def sanitize_filename(name):
    name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)
    return name if name else "logo"

def download_logo(logo_url, output_folder):
    """Downloads and converts logo images to PNG format.
    Handles SVG (vector), ICO (icon), and standard raster formats."""
    try:
        if not logo_url:
            return None

        if logo_url.startswith("data:image/svg+xml"):
            svg_data = logo_url.split(",", 1)[-1]
            svg_data = urllib.parse.unquote(svg_data)
            name = "inline_" + hashlib.md5(svg_data.encode('utf-8')).hexdigest()
            filename = f"{name}.png"
            save_path = os.path.join(output_folder, filename)

            try:
                cairosvg.svg2png(bytestring=svg_data.encode('utf-8'), write_to=save_path)
                return save_path
            except Exception as e:
                print(f"SVG inline conversion error: {e}")
                return None


        response = requests.get(logo_url, stream=True, timeout=15)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', "")
            ext = 'png'
            is_svg = False
            is_ico = False

            if 'image/jpeg' in content_type:
                ext = 'jpg'
            elif 'image/svg+xml' in content_type:
                ext = 'svg'
                is_svg = True
            elif 'image//x-icon' in content_type or logo_url.endswith('.ico'):
                ext = 'ico'
                is_ico = True

            path = urlparse(logo_url).path
            basename = os.path.basename(path)
            name, _ = os.path.splitext(basename)
            name = sanitize_filename(name)
            filename = f"{name}.{ext}"
            save_path = os.path.join(output_folder, filename)

            if is_svg:
                temp_svg_path = os.path.join(output_folder, f"{name}.svg")
                with open(temp_svg_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                try:
                    cairosvg.svg2png(url=temp_svg_path, write_to=save_path)
                    os.remove(temp_svg_path)
                    return save_path
                except Exception as e:
                    print(f"SVG inline conversion error: {e}")
                    return None

            if is_ico:
                img = Image.open(BytesIO(response.content))

                max_size = max(img.size, key=lambda x: x[0]*x[1])
                img = img.resize(max_size)

                png_path = save_path.replace('.ico', '.png')
                img.save(png_path, "PNG")
                return png_path

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return save_path

    except Exception as e:
        print(f"Error downloading {logo_url}: {str(e)}")
        return None

logging.basicConfig(
    filename='logo_scraper.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#------------------------
# IMAGE ANALYSIS
#------------------------
def extract_histogram(image_path):
    """Generates normalized HSV histogram for image comparison.
    Resizes images to 256x256 for consistent analysis."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        image = cv2.resize(image, (256, 256))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, NUM_BINS, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    except Exception as e:
        print(f"Error on histogram {image_path}: {e}")
        return None

def group_by_histograms(histograms, paths):
    """Clusters images using Bhattacharyya distance between histograms.
    Groups images below DISTANCE_THRESHOLD similarity threshold."""
    used = set()
    clusters = []

    for i, h1 in enumerate(histograms):
        if i in used or h1 is None:
            continue
        group = [paths[i]]
        used.add(i)

        for j, h2 in enumerate(histograms):
            if j in used or h2 is None or i == j:
                continue
            d = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
            if d < DISTANCE_THRESHOLD:
                group.append(paths[j])
                used.add(j)
        clusters.append(group)

    return clusters

def save_clusters(clusters, base_output='grouped_logos'):
    """Organizes clustered logos into numbered subdirectories."""
    Path(base_output).mkdir(exist_ok=True)
    for idx, cluster in enumerate(clusters):
        folder = Path(base_output) / f"cluster_{idx}"
        folder.mkdir(exist_ok=True)
        for logo_path in cluster:
            shutil.copy(logo_path, folder / Path(logo_path).name)

#------------------------
# MAIN WORKFLOW
#------------------------
def main():
    """Orchestrates the full pipeline:
    1. Load URLs from input file
    2. Scrape and download logos
    3. Convert all images to PNG
    4. Cluster visually similar logos
    5. Save results in structured directories"""

    parser = argparse.ArgumentParser(description='Logo Extraction & Clustering Tool')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to input file (Parquet/CSV)')
    parser.add_argument('-c', '--column', default='domain',
                        help='Column name containing URLs (default: domain)')
    parser.add_argument('-o', '--output', default='new_logos',
                        help='Output folder for logos (default: new_logos)')

    args = parser.parse_args()

    PARQUET_FILE = args.input
    URL_COLUMN = args.column
    OUTPUT_FOLDER = args.output

    if not Path(PARQUET_FILE).exists():
        raise FileNotFoundError(f"Input file {PARQUET_FILE} not found!")


    raw_urls = read_data(PARQUET_FILE, URL_COLUMN)
    urls = [normalize_url(url) for url in raw_urls]

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    DELAY_RANGE = (1, 3)

    for url in urls:
        try:
            time.sleep(random.uniform(*DELAY_RANGE))

            print(f"Processing: {url}")

            html = get_html(url)
            if not html:
                print(f"Fallback to Playwright for {url}")
                html = get_html(url, use_playwright=True)

            if not html:
                print(f"Could not retrieve HTML for {url}")
                continue

            logo_url = extract_logo_url(html, url)
            if not logo_url:
                print(f"No logo found for {url}")
                continue

            saved_path = download_logo(logo_url, OUTPUT_FOLDER)
            if saved_path:
                print(f"Logo saved: {saved_path}")

            time.sleep(2)
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")

    all_logos = [str(p) for p in Path(OUTPUT_FOLDER).glob("*.[pj][np]g") if p.is_file()]
    histograms = []

    print("Extracting histograms...")
    for path in tqdm(all_logos):
        hist = extract_histogram(path)
        histograms.append(hist)

    print("Grouping logos based on histograms...")
    clusters = group_by_histograms(histograms, all_logos)

    print(f"Saving {len(clusters)} groups in 'grouped_logos' folder")
    save_clusters(clusters)

    print("Done!")



if __name__ == "__main__":
    main()





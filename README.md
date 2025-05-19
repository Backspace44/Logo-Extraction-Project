# Logo Extraction & Clustering Tool

A Python-based tool for scraping logos from websites, converting them to PNG format, and grouping visually similar logos using histogram analysis.

---

## Features

### Core Functionality
- **Logo Detection**: Multi-layered HTML parsing for logo extraction:
  - `<img>` tags with logo-related attributes
  - Favicon links and meta tags (OpenGraph/Twitter)
  - CSS background images
  - Fallback to default favicon.ico

- **Format Handling**:
  - Supports SVG/ICO/JPG/PNG inputs
  - Automatic conversion to PNG format
  - Inline SVG data URI processing

- **Image Analysis**:
  - HSV histogram-based similarity comparison
  - Configurable clustering thresholds
  - Batch processing of URLs

### Technical Capabilities
- Resilient scraping with dual engines:
  - **BeautifulSoup** for static HTML parsing
  - **Playwright** for JavaScript-rendered pages
- Anti-detection measures:
  - Randomized user agents
  - Request delays
  - Proxy support
- Error resilience:
  - SSL error fallback (HTTPS→HTTP)
  - Retry mechanism (3 attempts)
  - Comprehensive logging

---

## Installation

### Requirements
- Python 3.8+
- Chrome/Chromium (for Playwright)

### Dependencies
```bash
pip install pandas requests beautifulsoup4 pillow playwright opencv-python numpy tqdm cairosvg pathlib shutil argparse

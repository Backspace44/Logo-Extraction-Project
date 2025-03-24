# Logo-Extraction-Project
My attempt at solving the Veridion challenge #1
### 🐝 README.md - Project: Grouping Logos Based on Visual Similarity
This project uses **web scraping**, **image processing**, and **machine learning** to extract, process, and group logos based on their visual similarity.

---

## 📌 Description
The goal of this project is to **automatically download website logos** based on domains listed in a **Parquet** file, **analyze their visual similarity**, and **group them by brand** using **computer vision and clustering algorithms**.

### 👉 What does the program do?
- ✅ **Extracts logos** from web pages using `Selenium` and `BeautifulSoup`
- ✅ **Detects and downloads logos** from meta-tags (`og:image`, `twitter:image`) and images on the page
- ✅ **Automatic conversion** of SVG/AVIF logos to PNG
- ✅ **Verifies image integrity** and removes corrupted ones
- ✅ **Extracts visual features** using `ORB` from OpenCV
- ✅ **Groups logos** based on visual similarity with `DBSCAN`

---

## 📂 Project Structure
```
📁 VeridionProject
│── 📁 logos/                     # Folder where downloaded logos are stored
│── 📄 logo_extraction.py         # Main script for downloading and grouping logos
│── 📄 requirements.txt           # List of required Python packages
│── 📄 README.md                  # Project documentation
│── 📄 logo_clusters.csv          # CSV file with clustering results
│── 📄 logos.snappy.parquet       # Input file with domain list
```

---

## 🚀 Installation and Setup
### 1️⃣ Install Dependencies
Before running the project, install all necessary packages:
```bash
pip install -r requirements.txt
```
📌 If you encounter errors with `cairosvg`, install `cairo`:
```bash
pip install cairosvg
```

### 2️⃣ Selenium Setup
📌 Ensure **Google Chrome** is installed and that `chromedriver` is automatically updated with:
```bash
pip install webdriver-manager
```

---

## 📝 Usage
1️⃣ **Place the `logos.snappy.parquet` file in the project folder**  
2️⃣ **Run the main script:**
```bash
python logo_extraction.py
```
3️⃣ **After execution, you will have:**
   - **Downloaded logos** in `logos/`
   - **Logos grouped by brand** in `logos/brand_X/`
   - **Clustering results** in `logo_clusters.csv`

---

## 🛠️ Main Features

### 👉 1. Automatic Logo Downloading
We use **Selenium + BeautifulSoup** to search for logos on each website:
- **Check meta-tags** (`og:image`, `twitter:image`)
- **Search for logos in the page structure** (`header-logo`, `site-logo`)
- **If no logo is found, download the favicon**

---

### 👉 2. Image Processing and Integrity Check
- ✅ **Automatic conversion**: SVG → PNG, AVIF → PNG
- ✅ **Remove corrupted files** using `PIL.Image.verify()`
- ✅ **Fallback to Selenium if download fails**

---

### 👉 3. Grouping Logos by Brand
- ✅ **Use ORB from OpenCV** to extract visual features
- ✅ **Apply DBSCAN** to group similar logos
- ✅ **Move logos into corresponding folders** (`brand_0/`, `brand_1/`, etc.)

---

## 📊 Results and Generated Files

**The `logos/` folder will contain:**
```
📂 logos/
├── brand_0/
│   ├── logo1.png
│   ├── logo2.png
├── brand_1/
│   ├── logo3.png
│   ├── logo4.png
```

**The `logo_clusters.csv` file will contain:**
| domain                           | cluster |
|----------------------------------|---------|
| mazda-autohaus-kilger-regen.de  | 0       |
| lidl.com.cy                     | 1       |
| toyota-buchreiter-eisenstadt.at | 2       |

---

## 💡 Possible Improvements
- 📌 **Intelligent logo detection** using **Deep Learning (ResNet, MobileNet)**
- 📌 **Improved web scraping** with `playwright` for protected websites
- 📌 **Scalability** – adaptation for processing hundreds of thousands of logos

---

## 📜 Conclusion
This project provides **an automated solution for extracting, analyzing, and grouping logos** based on visual similarity. 🚀  

If you have any questions or suggestions, feel free to contribute! 🛠️✨  




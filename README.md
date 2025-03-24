# Logo-Extraction-Project
My attempt at solving the Veridion challenge #1
### **\ud83d\udc1d README.md - Project: Grouping Logos Based on Visual Similarity**  
This project uses **web scraping**, **image processing**, and **machine learning** to extract, process, and group logos based on their visual similarity.  

---

## **\ud83d\udccc Description**  
The goal of this project is to **automatically download website logos** based on domains listed in a **Parquet** file, **analyze their visual similarity**, and **group them by brand** using **computer vision and clustering algorithms**.

### \ud83d\udc49 **What does the program do?**  
- \u2705 **Extracts logos** from web pages using `Selenium` and `BeautifulSoup`  
- \u2705 **Detects and downloads logos** from meta-tags (`og:image`, `twitter:image`) and images on the page  
- \u2705 **Automatic conversion** of SVG/AVIF logos to PNG  
- \u2705 **Verifies image integrity** and removes corrupted ones  
- \u2705 **Extracts visual features** using `ORB` from OpenCV  
- \u2705 **Groups logos** based on visual similarity with `DBSCAN`  

---

## **\ud83d\udcc2 Project Structure**  
```plaintext
\ud83d\udcc1 VeridionProject
│── \ud83d\udcc1 logos/                     # Folder where downloaded logos are stored
│── \ud83d\udcc4 logo_extraction.py         # Main script for downloading and grouping logos
│── \ud83d\udcc4 requirements.txt           # List of required Python packages
│── \ud83d\udcc4 README.md                  # Project documentation
│── \ud83d\udcc4 logo_clusters.csv          # CSV file with clustering results
│── \ud83d\udcc4 logos.snappy.parquet       # Input file with domain list
```

---

## **\ud83d\ude80 Installation and Setup**  
### **1️⃣ Install Dependencies**  
Before running the project, install all necessary packages:  
```bash
pip install -r requirements.txt
```
\ud83d\udccc If you encounter errors with `cairosvg`, install `cairo`:  
```bash
pip install cairosvg
```

### **2️⃣ Selenium Setup**  
\ud83d\udccc Ensure **Google Chrome** is installed and that `chromedriver` is automatically updated with:  
```bash
pip install webdriver-manager
```

---

## **\ud83d\udcc5 Usage**  
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

## **\ud83d\udee0️ Main Features**  

### **\ud83d\udc49 1. Automatic Logo Downloading**  
We use **Selenium + BeautifulSoup** to search for logos on each website:  
- **Check meta-tags** (`og:image`, `twitter:image`)  
- **Search for logos in the page structure** (`header-logo`, `site-logo`)  
- **If no logo is found, download the favicon**  

---

### **\ud83d\udc49 2. Image Processing and Integrity Check**  
- \u2705 **Automatic conversion**: SVG → PNG, AVIF → PNG  
- \u2705 **Remove corrupted files** using `PIL.Image.verify()`  
- \u2705 **Fallback to Selenium if download fails**  

---

### **\ud83d\udc49 3. Grouping Logos by Brand**  
- \u2705 **Use ORB from OpenCV** to extract visual features  
- \u2705 **Apply DBSCAN** to group similar logos  
- \u2705 **Move logos into corresponding folders** (`brand_0/`, `brand_1/`, etc.)  

---

## **\ud83d\udccb Results and Generated Files**  

**The `logos/` folder will contain:**  
```plaintext
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

## **\ud83d\udca1 Possible Improvements**  
- \ud83d\udccc **Intelligent logo detection** using **Deep Learning (ResNet, MobileNet)**  
- \ud83d\udccc **Improved web scraping** with `playwright` for protected websites  
- \ud83d\udccc **Scalability** – adaptation for processing hundreds of thousands of logos  

---

## **\ud83d\udcda Conclusion**  
This project provides **an automated solution for extracting, analyzing, and grouping logos** based on visual similarity. \ud83d\ude80  

If you have any questions or suggestions, feel free to contribute! \ud83d\udee0️✨  

---
\ud83d\udccc **Author:** *David*  
\ud83d\udccc **Technologies:** Python, Selenium, OpenCV, NumPy, DBSCAN  
\ud83d\udccc **License:** MIT  


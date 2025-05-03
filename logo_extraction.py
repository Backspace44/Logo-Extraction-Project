import os
import cairosvg
import cv2
import numpy as np
import mimetypes
import requests
import pandas as pd
from PIL import UnidentifiedImageError
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Rulează în fundal
options.add_argument("--no-sandbox")
options.add_argument("--disable-gpu")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.6998.89 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from skimage.feature import ORB

# Configurare Selenium
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def is_website_accessible(url):
    """Verifica daca un site este online inainte de a incerca scraping."""
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_logo_url(domain):
    """Caută URL-ul logo-ului pe baza domeniului."""
    url = f"https://{domain}"

    if not is_website_accessible(url):
        print(f"Site-ul {domain} nu este accesibil, trec la urmatorul.")
        return None

    try:
        driver.get(url)
    except Exception as e:
        print(f"Eroare la accesarea {url}: {e}")
        return None

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    # driver.quit()

    # Caută logo în meta tag-uri
    meta_tags = ['og:image', 'twitter:image']
    for tag in meta_tags:
        meta = soup.find('meta', property=tag)
        if meta and 'content' in meta.attrs:
            return urljoin(url, meta['content'])

    # Caută logo în imagini din header/nav
    logo_classes = ['logo', 'header-logo', 'site-logo']
    for img in soup.find_all('img'):
        img_class = img.get('class')
        if img_class and isinstance(img_class, list):
            img_class = " ".join(img_class) # Convertim lista in string
        if img_class and any(cls in img_class for cls in logo_classes):
            logo_url = urljoin(url, img['src'])
            print(f"Gasit logo in {logo_url}")
            return logo_url


    # Fallback: favicon
    url_of_logo = urljoin(url, '/favicon.ico')
    print(f"Fallback favicon: {url_of_logo}")
    return url_of_logo

def is_valid_image(file_path):
    """Verifica daca imaginea descarcata este valida."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def download_image(image_url, save_path, referer_url=None):
    """Descarcă și salvează imaginea, convertind SVG in PNG daca este necesar."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.6998.89 Safari/537.36"
    }
    if referer_url:
        headers["Referer"] = referer_url # Adauga Referer daca e necesar

    try:
        response = requests.get(image_url, stream=True, timeout=10)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')

            # Convertim SVG -: PNG
            if 'image/svg+xml' in content_type:
                print(f"Conversie SVG -> PNG: {image_url}")
                svg_data = response.content
                png_path = save_path
                cairosvg.svg2png(bytestring=svg_data, write_to=png_path)

            # Convertim AVIF -> PNG
            elif 'image/avif' in content_type:
                print(f"Conversie AVIF -> PNG: {image_url}")
                avif_data = io.BytesIO(response.content)
                img = Image.open(avif_data).convert("RGBA")
                img.save(save_path, format="PNG")

            # Salvam PNG, JPEG, WEBP fara conversie
            elif 'image/png' in content_type or 'image/jpeg' in content_type:
                  with open(save_path, 'wb') as file:
                      for chunk in response.iter_content(1024):
                          file.write(chunk)

            else:
                 print(f"{image_url} nu este PNG/JPG/SVG, ignorat.")
                 return False

    except Exception as e:
         print(f"Eroare la descarcarea imaginii {image_url}: {e}")
    return False

def download_image_with_selenium(image_url, save_path):
    """Foloseste Selenium pentru a descarca imaginea daca 'request.get()' esueaza."""

    options = webdriver.ChromeOptions()
    options.add_argument("--headless") # Ruleaza browser-ul in fundal
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


    try:
        driver.get(image_url)

        with open(save_path, 'wb') as file:
            file.write(driver.page_source.encode('utf-8')) # Salvam continutul imaginii

            print(f"Descarcare reusita cu Selenium: {save_path}")
            return True

    except Exception as e:
        print(f"Eroare Selenium la descarcarea imaginii {image_url}: {e}")
        return False

    finally:
        driver.quit()

def extract_features(image_path):
    """Extrage caracteristici din logo folosind ORB."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Imagine invalida sau inexistenta: {image_path}")
        return np.array([])

    orb = ORB(n_keypoints=100)
    orb.detect_and_extract(image)
    return orb.descriptors.flatten() if orb.descriptors is not None else np.array([])


def cluster_logos(feature_vectors):
    """Grupează logo-urile folosind DBSCAN."""
    pca = PCA(n_components=50)
    reduced_features = pca.fit_transform(feature_vectors)
    clustering = DBSCAN(eps=0.5, min_samples=2).fit(reduced_features)
    return clustering.labels_

def extract_features(image_path):
    """Extrage trasaturi vizuale folosind ORB."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Imagine invalida: {image_path}")
        return None

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return descriptors

def cluster_logos(features_list, eps=30, min_samples=2):
    """Grupeaza logo-urile pe baza similaritatii vizuale folosind DBSCAN."""
    if len(features_list) < 2:
        print("Prea putine imagini pentru clustering.")
        return np.array([-1] * len(features_list))  # Eticheta -1 = fara grup

    # Aplicam clustering DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(features_list)

    return clustering.labels_


# Citim fișierul Parquet
df = pd.read_parquet(
    "F:/Teme facultate/Anul II/Proiecte Personale/Projects/Veridion Projects/Logo Similarity/logos.snappy.parquet",
    engine="pyarrow")
output_dir = "logos"
os.makedirs(output_dir, exist_ok=True)

feature_list = []
processed_domains = []

# Procesăm fiecare domeniu
for domain in df['domain']:
    print(f"🔍 Procesăm: {domain}")
    logo_url = get_logo_url(domain)

    if not logo_url:
        print(f"⚠️ Niciun logo găsit pentru {domain}, trecem la următorul.")
        continue

    # Verificam daca URL-ul conduce direct la o imagine
    file_extension = os.path.splitext(logo_url)[1].lower()
    is_direct_image = file_extension in [".png", ".jpg", ".jpeg", ".svg", ".avif", ".webp"]

    # Creăm un nume de fișier sigur
    save_path = os.path.join(output_dir, f"{domain.replace('.', '_')}.png")


    if is_direct_image:
        print(f"Link direct catre imagine: {logo_url}")
        if download_image(logo_url, save_path):
            print(f"Logo descarcat direct: {save_path}")
        else:
            print(f"Eroare la descarcarea imaginii directe pentru {domain}")
        continue

    # Verificăm dacă imaginea există și este validă
    if os.path.exists(save_path) and os.path.getsize(save_path) > 100:
        print(f"✅ Imaginea există deja: {save_path}")
    else:
        print(f"⬇️ Descărcăm logo-ul pentru {domain}...")

        # Dacă fișierul există dar este gol, îl ștergem și descărcăm din nou
        if os.path.exists(save_path):
            os.remove(save_path)

        # Încercăm descărcarea cu requests, apoi cu Selenium dacă eșuează
        if not download_image(logo_url, save_path):
            print(f"❌ Descărcare eșuată cu `requests`, încercăm Selenium...")
            if not download_image_with_selenium(logo_url, save_path):
                print(f"❌ Nu am reușit să descărcăm logo-ul pentru {domain}, trecem la următorul.")
                continue

    # Extragem trăsăturile doar dacă imaginea este validă
    if os.path.exists(save_path) and os.path.getsize(save_path) > 100:
        features = extract_features(save_path)
        feature_list.append(features)
        processed_domains.append(domain)
        print(f"🎯 Logo procesat: {save_path}")
    else:
        print(f"⚠️ Imaginea {save_path} nu s-a descărcat corect, trecem la următorul logo.")

# Grupare logo-uri
def process_and_group_logos(logos_folder="C:/Users/david/PycharmProjects/VeridionProject/logos"):
    """Proceseaza toate logo-urile si le grupeaza in functie de similaritatea vizuala."""

    image_paths = [os.path.join(logos_folder, img) for img in os.listdir(logos_folder) if img.endswith(".png")]

    feature_list = []
    valid_image_paths = []

    for image_path in image_paths:
        features = extract_features(image_path)
        if features is not None:
            feature_list.append(np.mean(features, axis=0))
            valid_image_paths.append(image_path)

    if not feature_list:
        print("Nu s-au gasit imagini valide pentru clustering.")
        return

    # Transformam trasaturile intr-un array pentru clustering
    feature_array = np.array(feature_list)

    # Aplicam clustering pe baza similaritatii vizuale
    labels = cluster_logos(feature_array)

    # Cream foldere pentru fiecare grup
    clusters = {}
    for img_path, label in zip(valid_image_paths, labels):
        cluster_folder = os.path.join(logos_folder, f"brand_{label}")
        os.makedirs(cluster_folder, exist_ok=True)
        shutil.move(img_path, os.path.join(cluster_folder, os.path.basename(img_path)))

        print("Grupare finalizata! Logo-urile au fost organizate in functie de brand.")
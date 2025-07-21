from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import csv
import time

URL = "https://learn.microsoft.com/fr-fr/training/paths/introduction-to-ai-on-azure/"

def collect_ai900_unit_links_selenium():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    print(f"Chargement de la page (Selenium) : {URL}")
    driver = webdriver.Chrome(options=options)
    driver.get(URL)
    time.sleep(5)  # attendre que la page charge le contenu dynamique

    # Trouver tous les liens avec la classe "unit-title"
    elements = driver.find_elements(By.CSS_SELECTOR, "a.unit-title")

    data = []
    for el in elements:
        title = el.text.strip()
        href = el.get_attribute("href")
        if title and href:
            data.append({
                "module_name": "Microsoft Azure AI - Notions fondamentales",
                "unit_name": title,
                "unit_url": href
            })
            print(f"✔️  {title} → {href}")

    driver.quit()

    output_file = "ai900_units_list.csv"
    with open(output_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["module_name", "unit_name", "unit_url"])
        writer.writeheader()
        writer.writerows(data)

    print(f"\n✅ {len(data)} unités enregistrées dans {output_file}")

if __name__ == "__main__":
    collect_ai900_unit_links_selenium()

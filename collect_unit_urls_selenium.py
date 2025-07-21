from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from urllib.parse import urljoin
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
    time.sleep(5)  # Laisse le temps au contenu de charger

    # 👉 Étape 1 : Cliquer sur tous les boutons d’expansion
    expand_buttons = driver.find_elements(By.CSS_SELECTOR, "button.unit-expander")
    for btn in expand_buttons:
        try:
            if btn.is_displayed() and btn.get_attribute("aria-expanded") == "false":
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(1)  # Attendre le chargement après chaque clic
        except Exception as e:
            print(f"❌ Erreur bouton expand : {e}")

    # 👉 Étape 2 : Récupérer tous les liens d’unité
    elements = driver.find_elements(By.CSS_SELECTOR, "a.unit-title")

    data = []
    for el in elements:
        title = el.text.strip()
        href = el.get_attribute("href")
        if title and href:
            full_url = urljoin(URL, href)
            data.append({
                "module_name": "Microsoft Azure AI - Notions fondamentales",
                "unit_name": title,
                "unit_url": full_url
            })
            print(f"✔️  {title} → {full_url}")

    driver.quit()

    # 👉 Étape 3 : Sauvegarde dans un fichier CSV
    output_file = "ai900_units_list.csv"
    with open(output_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["module_name", "unit_name", "unit_url"])
        writer.writeheader()
        writer.writerows(data)

    print(f"\n✅ {len(data)} unités enregistrées dans {output_file}")

if __name__ == "__main__":
    collect_ai900_unit_links_selenium()

import pandas as pd
from bs4 import BeautifulSoup
import time
import re
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv(".env")

# Chargement du tokenizer Mistral via HuggingFace
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1",
                                          use_auth_token=os.getenv("HF_TOKEN"))

def count_tokens(text):
    return len(tokenizer.encode(text))

def chunk_text(text, max_tokens=512):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def extract_text_from_url_selenium(url, driver):
    try:
        driver.get(url)
        time.sleep(3)  # temps pour chargement JS

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        content_divs = soup.find_all("div", class_="module-unit")
        if not content_divs:
            content_divs = soup.find_all("main")

        raw_text = " ".join(div.get_text(separator=" ", strip=True) for div in content_divs)
        text = re.sub(r'\s+', ' ', raw_text)
        return text.strip()
    except Exception as e:
        print(f"Erreur sur {url} : {e}")
        return ""

def main():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=chrome_options)  # chromedriver doit être accessible

    df_units = pd.read_csv("ai900_units_list.csv")

    rows = []

    for _, row in df_units.iterrows():
        print(f"🔍 Traitement de : {row['unit_name']}")
        text = extract_text_from_url_selenium(row['unit_url'], driver)
        if not text:
            continue
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            rows.append({
                "module_name": row["module_name"],
                "unit_name": row["unit_name"],
                "unit_url": row["unit_url"],
                "chunk_id": idx,
                "text_chunk": chunk,
                "token_count": count_tokens(chunk)
            })
        time.sleep(1.5)

    driver.quit()

    df_chunks = pd.DataFrame(rows)
    df_chunks.to_csv("azure_learning_chunks.csv", index=False)
    print("✅ Fichier enregistré : azure_learning_chunks.csv")

if __name__ == "__main__":
    main()

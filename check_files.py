import os

print("--- Contenu de app.py ---")
try:
    with open("app.py", "r", encoding="utf-8") as f:
        print(f.read())
except FileNotFoundError:
    print("app.py non trouvé dans le répertoire actuel.")
except Exception as e:
    print(f"Erreur de lecture de app.py : {e}")

print("\n--- Contenu de prompts.yaml ---")
try:
    with open("prompts.yaml", "r", encoding="utf-8") as f:
        print(f.read())
except FileNotFoundError:
    print("prompts.yaml non trouvé dans le répertoire actuel.")
except Exception as e:
    print(f"Erreur de lecture de prompts.yaml : {e}")

print("\n--- Contenu de .env (partiel pour la clé Mistral) ---")
try:
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            if "MISTRAL_API_KEY" in line:
                print(line.strip())
                break
        else:
            print("MISTRAL_API_KEY non trouvée dans .env")
except FileNotFoundError:
    print(".env non trouvé dans le répertoire actuel.")
except Exception as e:
    print(f"Erreur de lecture de .env : {e}")